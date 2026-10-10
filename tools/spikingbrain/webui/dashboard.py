import time
import torch
import gradio as gr
import pandas as pd
from typing import Optional, Tuple, Generator

from spikingbrain.config import SpikingBrainConfig
from spikingbrain.core.modeling_spikingbrain import SpikingBrain7BForCausalLM
from spikingbrain.inference.hymeta_engine import HyMetaInferenceEngine
from spikingbrain.inference.sparsity_hooks import SparsityTelemetry


def build_webui(
    model: Optional[SpikingBrain7BForCausalLM] = None,
    config: Optional[SpikingBrainConfig] = None,
) -> gr.Blocks:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model is None:
        cfg = config or SpikingBrainConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            v_threshold=0.8,
            leak_factor=0.9,
        )
        model = SpikingBrain7BForCausalLM(cfg).to(device)
    else:
        model = model.to(device)

    engine = HyMetaInferenceEngine(model)

    def encode_text(text: str) -> torch.LongTensor:
        token_ids = [(ord(c) % 900) + 50 for c in text[:64]]
        if not token_ids:
            token_ids = [100]
        return torch.tensor([token_ids], dtype=torch.long, device=device)

    def decode_token_id(token_id: int) -> str:
        return chr((token_id % 70) + 32)

    def chat_stream(
        message: str,
        v_threshold: float,
        leak_factor: float,
        temperature: float,
        max_tokens: int,
    ):
        # Update neuromorphic parameters dynamically
        model.config.v_threshold = v_threshold
        model.config.leak_factor = leak_factor
        for module in model.modules():
            if hasattr(module, "v_threshold"):
                module.v_threshold = float(v_threshold)
            if hasattr(module, "leak_factor"):
                module.leak_factor = float(leak_factor)

        prompt_tensor = encode_text(message)
        tokens_out = []
        total_lat_ms = 0.0

        stream = engine.generate_stream(
            prompt_tensor,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
        )

        for step_data in stream:
            if step_data["phase"] == "decode":
                tok_char = decode_token_id(step_data["token_id"])
                tokens_out.append(tok_char)
                total_lat_ms += step_data["latency_ms"]
                partial_text = "".join(tokens_out)

                sparsity_str = f"{step_data['step_sparsity_percent']:.2f}%"
                sops_str = f"{step_data['step_sops']:,}"
                eer_str = f"{step_data['step_energy_efficiency']:.2f}x"
                tok_per_sec = f"{(len(tokens_out) / (total_lat_ms / 1000.0)):.2f} tok/s" if total_lat_ms > 0 else "0 tok/s"

                layer_df = pd.DataFrame([
                    {"Layer Component": "Attention Q-Neuron", "Spike Rate": f"{step_data['step_spike_rate']:.4f}", "Sparsity": sparsity_str},
                    {"Layer Component": "Attention K-Neuron", "Spike Rate": f"{step_data['step_spike_rate']:.4f}", "Sparsity": sparsity_str},
                    {"Layer Component": "Feed-Forward LIF", "Spike Rate": f"{step_data['step_spike_rate']:.4f}", "Sparsity": sparsity_str},
                ])

                yield (
                    partial_text,
                    sparsity_str,
                    sops_str,
                    eer_str,
                    tok_per_sec,
                    layer_df,
                )

    with gr.Blocks(title="SpikingBrain-7B Playground") as demo:
        gr.Markdown(
            """
            # ⚡ SpikingBrain-7B Neuromorphic Playground
            ### High-Throughput Event-Driven LLM Inference & Live Sparsity Dashboard
            """
        )

        with gr.Row():
            # Left: Chat and controls
            with gr.Column(scale=5):
                chatbot = gr.Chatbot(label="Conversation", height=420)
                msg_input = gr.Textbox(placeholder="Ask SpikingBrain something...", label="Input Prompt")
                with gr.Row():
                    send_btn = gr.Button("Generate ⚡", variant="primary")
                    clear_btn = gr.Button("Clear")

                with gr.Accordion("⚙️ Neuromorphic & Sampling Hyperparameters", open=False):
                    v_th_slider = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Membrane Threshold (V_th)")
                    leak_slider = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Membrane Leak Factor (β)")
                    temp_slider = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
                    max_tok_slider = gr.Slider(8, 64, value=32, step=4, label="Max Generated Tokens")

            # Right: Live Diagnostics
            with gr.Column(scale=5):
                gr.Markdown("### 📊 Live Neuromorphic Telemetry")
                with gr.Row():
                    sparsity_card = gr.Textbox(label="Global Sparsity", value="-- %")
                    sops_card = gr.Textbox(label="Synaptic Operations", value="-- SOPs")
                with gr.Row():
                    eer_card = gr.Textbox(label="Energy Savings Factor", value="-- x")
                    speed_card = gr.Textbox(label="Throughput", value="-- tok/s")

                layer_table = gr.Dataframe(
                    headers=["Layer Component", "Spike Rate", "Sparsity"],
                    label="Layer-wise Event Density Breakdown",
                )

        def user_submit(user_msg, chat_hist):
            chat_hist = chat_hist or []
            return "", chat_hist + [{"role": "user", "content": user_msg}]

        def bot_stream(chat_hist, v_th, leak, temp, max_tok):
            user_msg = chat_hist[-1]["content"]
            assistant_entry = {"role": "assistant", "content": ""}
            chat_hist.append(assistant_entry)
            for (part_text, sp, sops, eer, spd, l_df) in chat_stream(user_msg, v_th, leak, temp, max_tok):
                chat_hist[-1]["content"] = part_text
                yield chat_hist, sp, sops, eer, spd, l_df

        msg_input.submit(
            user_submit, [msg_input, chatbot], [msg_input, chatbot], queue=False
        ).then(
            bot_stream, [chatbot, v_th_slider, leak_slider, temp_slider, max_tok_slider],
            [chatbot, sparsity_card, sops_card, eer_card, speed_card, layer_table]
        )

        send_btn.click(
            user_submit, [msg_input, chatbot], [msg_input, chatbot], queue=False
        ).then(
            bot_stream, [chatbot, v_th_slider, leak_slider, temp_slider, max_tok_slider],
            [chatbot, sparsity_card, sops_card, eer_card, speed_card, layer_table]
        )

        clear_btn.click(lambda: [], None, chatbot, queue=False)

    return demo

import json
import time
import torch
from typing import Optional, Generator
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from spikingbrain.config import SpikingBrainConfig
from spikingbrain.core.modeling_spikingbrain import SpikingBrain7BForCausalLM
from spikingbrain.inference.hymeta_engine import HyMetaInferenceEngine
from spikingbrain.inference.sparsity_hooks import SparsityTelemetry
from spikingbrain.server.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    UsageInfo,
    NeuromorphicMetrics,
    ChatCompletionChunk,
    ChunkChoice,
    DeltaMessage,
    ModelList,
    ModelCard,
)


def create_app(
    model: Optional[SpikingBrain7BForCausalLM] = None,
    config: Optional[SpikingBrainConfig] = None,
) -> FastAPI:
    app = FastAPI(
        title="SpikingBrain-7B OpenAI-Compatible API",
        description="High-throughput event-driven inference server for SpikingBrain hybrid SNN foundation model",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize model if not passed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        cfg = config or SpikingBrainConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
        )
        model = SpikingBrain7BForCausalLM(cfg).to(device)
    else:
        model = model.to(device)

    engine = HyMetaInferenceEngine(model)

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "model": "spikingbrain-7b",
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
        }

    @app.get("/v1/models", response_model=ModelList)
    async def list_models():
        return ModelList(data=[ModelCard(id="spikingbrain-7b")])

    def encode_messages(messages) -> torch.LongTensor:
        # Synthetic character/word hash encoding for prototype tokenization
        text = " ".join([m.content for m in messages])
        token_ids = [(ord(c) % 900) + 50 for c in text[:64]]
        if not token_ids:
            token_ids = [100]
        return torch.tensor([token_ids], dtype=torch.long, device=device)

    def decode_token_id(token_id: int) -> str:
        # Simple printable token mapping
        char_val = (token_id % 70) + 32
        return chr(char_val)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        prompt_tensor = encode_messages(request.messages)
        max_tokens = request.max_tokens or 32
        temperature = request.temperature or 0.7

        if request.stream:
            def event_stream() -> Generator[str, None, None]:
                stream = engine.generate_stream(
                    prompt_tensor,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
                yield f"data: {json.dumps(ChatCompletionChunk(choices=[ChunkChoice(index=0, delta=DeltaMessage(role='assistant', content=''))]).model_dump())}\n\n"

                for step_data in stream:
                    if step_data["phase"] == "decode":
                        token_str = decode_token_id(step_data["token_id"])
                        chunk = ChatCompletionChunk(
                            choices=[ChunkChoice(index=0, delta=DeltaMessage(content=token_str))],
                            neuromorphic_step_telemetry={
                                "step": step_data["step"],
                                "sparsity_percent": step_data["step_sparsity_percent"],
                                "sops": step_data["step_sops"],
                                "energy_savings_ratio": step_data["step_energy_efficiency"],
                                "latency_ms": step_data["latency_ms"],
                            },
                        )
                        yield f"data: {json.dumps(chunk.model_dump())}\n\n"

                # Send completion stop chunk
                end_chunk = ChatCompletionChunk(choices=[ChunkChoice(index=0, delta=DeltaMessage(), finish_reason="stop")])
                yield f"data: {json.dumps(end_chunk.model_dump())}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Neuromorphic-Model": "spikingbrain-7b",
                },
            )

        # Non-streaming synchronous path
        tokens_out = []
        with SparsityTelemetry(model) as telemetry:
            stream = engine.generate_stream(prompt_tensor, max_new_tokens=max_tokens, temperature=temperature)
            for step_data in stream:
                if step_data["phase"] == "decode":
                    tokens_out.append(decode_token_id(step_data["token_id"]))

        summary = telemetry.get_summary()
        em = summary["energy_metrics"]
        response_text = "".join(tokens_out)

        usage = UsageInfo(
            prompt_tokens=prompt_tensor.shape[1],
            completion_tokens=len(tokens_out),
            total_tokens=prompt_tensor.shape[1] + len(tokens_out),
            neuromorphic_telemetry=NeuromorphicMetrics(
                global_spike_rate=summary["global_spike_rate"],
                global_sparsity_percent=summary["global_sparsity_percent"],
                dense_baseline_flops=em["dense_flops"],
                synaptic_operations_sops=em["sops"],
                dense_energy_joules=em["dense_energy_joules"],
                snn_energy_joules=em["snn_energy_joules"],
                energy_savings_ratio=em["energy_efficiency_ratio"],
            ),
        )

        resp = ChatCompletionResponse(
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )

        headers = {
            "X-Neuromorphic-Sparsity": f"{summary['global_sparsity_percent']:.2f}%",
            "X-Synaptic-Operations-SOPs": str(em["sops"]),
            "X-Energy-Savings-Ratio": f"{em['energy_efficiency_ratio']:.2f}x",
        }
        return JSONResponse(content=resp.model_dump(), headers=headers)

    return app

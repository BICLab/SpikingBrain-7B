import sys
import time
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from spikingbrain.config import SpikingBrainConfig
from spikingbrain.core.modeling_spikingbrain import SpikingBrain7BForCausalLM
from spikingbrain.inference.hymeta_engine import HyMetaInferenceEngine
from spikingbrain.profiler.reporter import SpikeProfilerReporter


def run_standalone_evaluation():
    print("=" * 75)
    print("[*] INITIATING STANDALONE SPIKINGBRAIN-7B EVALUATION & BENCHMARK [*]")
    print("=" * 75)

    # 1. Instantiate benchmark model topology
    config = SpikingBrainConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        v_threshold=0.8,
        leak_factor=0.9,
    )
    print(f"[*] Initializing model with {config.num_hidden_layers} spiking layers...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")
    
    model = SpikingBrain7BForCausalLM(config).to(device)
    engine = HyMetaInferenceEngine(model)

    # 2. Benchmark prompt
    prompt_ids = torch.randint(10, 500, (1, 16)).to(device)
    print(f"[*] Prompt length: {prompt_ids.shape[1]} tokens. Generating 16 new tokens...")
    print("-" * 75)

    stream = engine.generate_stream(prompt_ids, max_new_tokens=16)

    # 3. Stream output & collect metrics
    last_stats = {}
    tokens_generated = 0
    total_latency_ms = 0.0

    for step_data in stream:
        phase = step_data["phase"]
        step = step_data["step"]
        token_id = step_data["token_id"]

        if phase == "prefill":
            print(f"[Prefill] Prompt processed | Sparsity: {step_data['prefill_sparsity']:.2f}% | Spike Rate: {step_data['prefill_spike_rate']:.4f}")
        else:
            lat = step_data["latency_ms"]
            sparsity = step_data["step_sparsity_percent"]
            sops = step_data["step_sops"]
            eer = step_data["step_energy_efficiency"]
            total_latency_ms += lat
            tokens_generated += 1
            print(f"[Decode Step {step:02d}] Token: {token_id:04d} | Latency: {lat:.2f}ms | Sparsity: {sparsity:.2f}% | SOPs: {sops:,} | Energy Savings: {eer:.2f}x")

    # 4. Final Telemetry Profile Report
    avg_latency = total_latency_ms / max(tokens_generated, 1)
    tokens_per_sec = (tokens_generated / (total_latency_ms / 1000.0)) if total_latency_ms > 0 else 0.0
    print("-" * 75)
    print(f"[*] Generation Complete: {tokens_generated} tokens | Avg Latency: {avg_latency:.2f}ms/token | Throughput: {tokens_per_sec:.2f} tok/s")
    
    # 5. Full profile display
    from spikingbrain.inference.sparsity_hooks import SparsityTelemetry
    with SparsityTelemetry(model) as profiler:
        _ = model(prompt_ids)
        telemetry = profiler.get_summary()

    SpikeProfilerReporter.print_terminal_report(telemetry)
    print("[+] Standalone evaluation executed successfully!")


if __name__ == "__main__":
    run_standalone_evaluation()

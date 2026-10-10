import time
import torch
from typing import Generator, Dict, Any, Optional
from spikingbrain.core.modeling_spikingbrain import SpikingBrain7BForCausalLM
from spikingbrain.core.cache import SpikingStateCache
from spikingbrain.inference.sparsity_hooks import SparsityTelemetry


class HyMetaInferenceEngine:
    """
    Event-driven high-throughput inference engine for SpikingBrain.
    Streams output tokens while concurrently measuring micro-level sparsity telemetry.
    """
    def __init__(self, model: SpikingBrain7BForCausalLM):
        self.model = model
        self.model.eval()

    @torch.inference_mode()
    def generate_stream(
        self,
        prompt_ids: torch.LongTensor,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Autoregressive generation stream yielding generated token IDs and telemetry per step.
        """
        device = next(self.model.parameters()).device
        curr_input_ids = prompt_ids.to(device)
        state_cache = SpikingStateCache()

        # Phase 1: Prefill prompt
        start_time = time.perf_counter()
        with SparsityTelemetry(self.model) as prefill_telemetry:
            outputs = self.model(curr_input_ids, state_cache=state_cache)
            next_token_logits = outputs.logits[:, -1, :]

        prefill_stats = prefill_telemetry.get_summary()

        # Sample initial next token
        if temperature > 0:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        yield {
            "phase": "prefill",
            "token_id": next_token.item(),
            "step": 0,
            "prefill_sparsity": prefill_stats["global_sparsity_percent"],
            "prefill_spike_rate": prefill_stats["global_spike_rate"],
        }

        # Phase 2: Autoregressive decoding (T=1 steps with state cache)
        for step in range(1, max_new_tokens):
            decode_input = next_token
            step_start = time.perf_counter()

            with SparsityTelemetry(self.model) as step_telemetry:
                outputs = self.model(decode_input, state_cache=state_cache)
                next_token_logits = outputs.logits[:, -1, :]

            step_time = time.perf_counter() - step_start
            step_stats = step_telemetry.get_summary()

            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            yield {
                "phase": "decode",
                "token_id": next_token.item(),
                "step": step,
                "latency_ms": step_time * 1000.0,
                "step_spike_rate": step_stats["global_spike_rate"],
                "step_sparsity_percent": step_stats["global_sparsity_percent"],
                "step_sops": step_stats["energy_metrics"]["sops"],
                "step_energy_efficiency": step_stats["energy_metrics"]["energy_efficiency_ratio"],
            }

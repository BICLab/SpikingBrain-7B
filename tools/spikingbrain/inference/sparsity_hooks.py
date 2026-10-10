import torch
import torch.nn as nn
from typing import Dict, List, Any
from collections import defaultdict
from spikingbrain.core.neurons import LIFNeuron, ALIFNeuron
from spikingbrain.inference.energy_model import NeuromorphicEnergyModel


class SparsityTelemetry:
    """
    Context manager and forward-hook dispatcher that profiles layer-wise
    spike density, synaptic operations (SOPs), and dynamic energy consumption in real time.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.energy_model = NeuromorphicEnergyModel()
        self.hooks = []
        self.layer_spike_records: Dict[str, List[float]] = defaultdict(list)
        self.total_elements: int = 0
        self.total_spikes: int = 0

    def _hook_fn(self, name: str):
        def hook(module, input_tensor, output):
            # output is (spikes, v_mem) or (spikes, v_mem, adapt)
            if isinstance(output, tuple):
                spikes = output[0]
            else:
                spikes = output

            if isinstance(spikes, torch.Tensor):
                with torch.no_grad():
                    num_spikes = spikes.sum().item()
                    numel = spikes.numel()
                    rate = num_spikes / max(numel, 1)
                    self.layer_spike_records[name].append(rate)
                    self.total_spikes += int(num_spikes)
                    self.total_elements += numel
        return hook

    def attach(self):
        self.reset()
        for name, module in self.model.named_modules():
            if isinstance(module, (LIFNeuron, ALIFNeuron)):
                handle = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(handle)

    def remove(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    def reset(self):
        self.layer_spike_records.clear()
        self.total_elements = 0
        self.total_spikes = 0

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()

    def get_summary(self) -> Dict[str, Any]:
        global_rate = (self.total_spikes / self.total_elements) if self.total_elements > 0 else 0.0
        layer_averages = {
            name: (sum(rates) / len(rates)) if rates else 0.0
            for name, rates in self.layer_spike_records.items()
        }

        # Energy projection based on total elements
        energy_stats = self.energy_model.compute_energy(
            dense_flops=max(self.total_elements, 1),
            spike_rate=global_rate,
        )

        return {
            "global_spike_rate": global_rate,
            "global_sparsity_percent": (1.0 - global_rate) * 100.0,
            "total_spikes": self.total_spikes,
            "total_synaptic_elements": self.total_elements,
            "layer_spike_rates": layer_averages,
            "energy_metrics": energy_stats,
        }

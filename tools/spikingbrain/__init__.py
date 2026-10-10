"""
SpikingBrain-7B Developer Ecosystem Toolkit
High-throughput inference, neuromorphic profiling, and PEFT toolkit for hybrid spiking transformers.
"""

__version__ = "0.1.0"

from spikingbrain.config import SpikingBrainConfig
from spikingbrain.core.modeling_spikingbrain import (
    SpikingBrain7BPreTrainedModel,
    SpikingBrain7BModel,
    SpikingBrain7BForCausalLM,
)
from spikingbrain.inference.sparsity_hooks import SparsityTelemetry
from spikingbrain.inference.hymeta_engine import HyMetaInferenceEngine
from spikingbrain.inference.energy_model import HardwareEnergySpec, NeuromorphicEnergyModel
from spikingbrain.peft.spike_lora import (
    SpikeLoRALinear,
    inject_spike_lora,
    merge_spike_lora,
)
from spikingbrain.profiler.reporter import SpikeProfilerReporter
from spikingbrain.server.app import create_app
from spikingbrain.webui.dashboard import build_webui

__all__ = [
    "SpikingBrainConfig",
    "SpikingBrain7BPreTrainedModel",
    "SpikingBrain7BModel",
    "SpikingBrain7BForCausalLM",
    "SparsityTelemetry",
    "HyMetaInferenceEngine",
    "HardwareEnergySpec",
    "NeuromorphicEnergyModel",
    "SpikeLoRALinear",
    "inject_spike_lora",
    "merge_spike_lora",
    "SpikeProfilerReporter",
    "create_app",
    "build_webui",
    "__version__",
]

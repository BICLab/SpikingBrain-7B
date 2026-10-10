import torch
import torch.nn as nn
from typing import Optional, Tuple
from spikingbrain.config import SpikingBrainConfig
from spikingbrain.core.neurons import LIFNeuron
from spikingbrain.core.attention import LinearSpikeAttention
from spikingbrain.core.cache import SpikingStateCache


class SpikingRMSNorm(nn.Module):
    """
    Root Mean Square Normalization adapted for spiking representations.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class SpikingMLP(nn.Module):
    """
    Event-driven Feed-Forward Network where non-linear activation is replaced
    by a Leaky Integrate-and-Fire (LIF) spike neuron, achieving micro-level sparsity in FFN layers.
    """
    def __init__(self, config: SpikingBrainConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        # Spiking neuron replacing traditional continuous activation
        self.act_neuron = LIFNeuron(
            v_threshold=config.v_threshold,
            leak_factor=config.leak_factor,
            surrogate_alpha=config.surrogate_alpha,
            surrogate_function=config.surrogate_function,
            reset_mode=config.spike_reset_mode,
        )

    def forward(
        self,
        x: torch.Tensor,
        state_cache: Optional[SpikingStateCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vmem_key = f"layer_{self.layer_idx}_mlp_vmem"
        vmem = state_cache.get_membrane(vmem_key) if state_cache else None

        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # Spike generation on pre-synaptic gate
        spikes, vmem = self.act_neuron(gate, vmem)
        if state_cache is not None:
            state_cache.set_membrane(vmem_key, vmem[:, -1:, :])

        # Event-sparse gated multiplication
        activated = spikes * up
        out = self.down_proj(activated)

        return out, spikes


class SpikingBrainDecoderLayer(nn.Module):
    """
    Unified Spiking Transformer Decoder Layer:
    RMSNorm -> LinearSpikeAttention -> Residual -> RMSNorm -> SpikingMLP -> Residual
    """
    def __init__(self, config: SpikingBrainConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = SpikingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LinearSpikeAttention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = SpikingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SpikingMLP(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        state_cache: Optional[SpikingStateCache] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Attention block
        normed_attn_in = self.input_layernorm(hidden_states)
        attn_out, q_spikes, k_spikes = self.self_attn(normed_attn_in, state_cache=state_cache)
        hidden_states = hidden_states + attn_out

        # MLP block
        normed_mlp_in = self.post_attention_layernorm(hidden_states)
        mlp_out, ffn_spikes = self.mlp(normed_mlp_in, state_cache=state_cache)
        hidden_states = hidden_states + mlp_out

        return hidden_states, (q_spikes, k_spikes, ffn_spikes)

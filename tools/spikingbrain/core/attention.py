import torch
import torch.nn as nn
from typing import Optional, Tuple
from spikingbrain.config import SpikingBrainConfig
from spikingbrain.core.neurons import LIFNeuron
from spikingbrain.core.cache import SpikingStateCache


class LinearSpikeAttention(nn.Module):
    """
    Hybrid Linear Spike Attention (LSA) layer.
    - Projects queries and keys into sparse binary spikes via LIF neurons.
    - Projects continuous values.
    - Replaces O(N^2) Softmax attention with O(N) associative linear attention.
    - Provides O(1) stateful decoding per token via associative recurrent state.
    """
    def __init__(self, config: SpikingBrainConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.eps = config.linear_attention_eps

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Spiking gating neurons for Q and K
        self.q_neuron = LIFNeuron(
            v_threshold=config.v_threshold,
            leak_factor=config.leak_factor,
            surrogate_alpha=config.surrogate_alpha,
            surrogate_function=config.surrogate_function,
            reset_mode=config.spike_reset_mode,
        )
        self.k_neuron = LIFNeuron(
            v_threshold=config.v_threshold,
            leak_factor=config.leak_factor,
            surrogate_alpha=config.surrogate_alpha,
            surrogate_function=config.surrogate_function,
            reset_mode=config.spike_reset_mode,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        state_cache: Optional[SpikingStateCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        hidden_states: [B, T, D]
        Returns:
            context: [B, T, D]
            q_spikes: [B, H, T, HeadDim]
            k_spikes: [B, H, T, HeadDim]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Linear projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Retrieve cached membrane potentials if applicable
        q_vmem_key = f"layer_{self.layer_idx}_q_vmem"
        k_vmem_key = f"layer_{self.layer_idx}_k_vmem"
        q_vmem = state_cache.get_membrane(q_vmem_key) if state_cache else None
        k_vmem = state_cache.get_membrane(k_vmem_key) if state_cache else None

        # Convert to spikes
        q_spikes, q_vmem = self.q_neuron(q, q_vmem)
        k_spikes, k_vmem = self.k_neuron(k, k_vmem)

        if state_cache is not None:
            state_cache.set_membrane(q_vmem_key, q_vmem[:, -1:, :])
            state_cache.set_membrane(k_vmem_key, k_vmem[:, -1:, :])

        # Reshape to multi-head: [B, H, T, HeadDim]
        q_spikes_h = q_spikes.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_spikes_h = k_spikes.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_h = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Decoding step (T == 1 with cache) vs Prefill chunk (T > 1)
        if seq_len == 1 and state_cache is not None:
            m_cached, z_cached = state_cache.get_attn_state(self.layer_idx)
            if m_cached is None:
                m_cached = torch.zeros(batch_size, self.num_heads, self.head_dim, self.head_dim, device=hidden_states.device, dtype=v.dtype)
                z_cached = torch.zeros(batch_size, self.num_heads, self.head_dim, device=hidden_states.device, dtype=v.dtype)

            # k_t: [B, H, 1, Dk], v_t: [B, H, 1, Dv]
            kt = k_spikes_h.squeeze(2)  # [B, H, Dk]
            vt = v_h.squeeze(2)         # [B, H, Dv]
            qt = q_spikes_h.squeeze(2)  # [B, H, Dk]

            # Stateful update: M += kt.unsqueeze(-1) * vt.unsqueeze(-2)
            m_cached = m_cached + torch.einsum("bhi,bhj->bhij", kt, vt)
            z_cached = z_cached + kt

            state_cache.set_attn_state(self.layer_idx, m_cached, z_cached)

            # Output: (qt @ M) / (qt @ z + eps)
            num = torch.einsum("bhi,bhij->bhj", qt, m_cached)  # [B, H, Dv]
            denom = torch.einsum("bhi,bhi->bh", qt, z_cached).unsqueeze(-1) + self.eps
            out = (num / denom).unsqueeze(2)  # [B, H, 1, Dv]
        else:
            # Full sequence prefill associative scan:
            # M_t = cumsum(K_t^T @ V_t), Z_t = cumsum(K_t)
            kv = torch.einsum("bhti,bhtj->bhtij", k_spikes_h, v_h)
            m_cum = torch.cumsum(kv, dim=2)
            z_cum = torch.cumsum(k_spikes_h, dim=2)

            num = torch.einsum("bhti,bhtij->bhtj", q_spikes_h, m_cum)
            denom = torch.sum(q_spikes_h * z_cum, dim=-1, keepdim=True) + self.eps
            out = num / denom

            if state_cache is not None:
                # Store final timestep accumulator state
                state_cache.set_attn_state(self.layer_idx, m_cum[:, :, -1], z_cum[:, :, -1])

        # Concatenate heads: [B, T, D]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        context = self.out_proj(out)

        return context, q_spikes, k_spikes

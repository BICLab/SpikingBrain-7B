# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

from transformers.modeling_flash_attention_utils import _upad_input
from transformers.cache_utils import Cache

from transformers.utils import is_flash_attn_2_available
if is_flash_attn_2_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_func, flash_attn_func
    from flash_attn.bert_padding import pad_input

from .cache import Cache
from .quant_linear import QuantLinear, dynamic_spikes

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)    


class FlashAttention(nn.Module):
    def __init__(
            self,
            hidden_size: int = 1024,
            num_heads: Optional[int] = None,
            num_key_value_heads: Optional[int] = None,
            max_position_embeddings: int = 8192, 
            sliding_window: Optional[int] = None,
            rope_theta: float = 10000.0,
            attention_dropout: float = 0.0,
            layer_idx: Optional[int] = None,
        ) -> FlashAttention:
        super().__init__()
        self.layer_idx = layer_idx

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.sliding_window = sliding_window
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        self.q_proj = QuantLinear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = QuantLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = QuantLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = QuantLinear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        
        self.apply(self._initialize_weights)
    
    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            cache_kwargs = {"sin": sin, "cos": cos, "window_size": self.sliding_window}
            key_cached, value_cached = past_key_values.update(
                attn_state=(key_states, value_states),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=cache_kwargs,
            )['attn_state']
            if cache_has_content:
                key_states, value_states = key_cached, value_cached

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        query_states_int, q_vth = dynamic_spikes(query_states, 3.0)
        query_states = (query_states_int * q_vth).to(query_states.dtype)
        key_states = quantize_sym(key_states)
        value_states = quantize_sym(value_states)

        if self.training:
            attn_output = flash_attn_func(
                q=query_states,
                k=key_states,
                v=value_states,
                dropout_p=self.attention_dropout if self.training else 0.0,
                softmax_scale=key_states.shape[-1] ** (-0.5),
                causal=True,
                window_size=(self.sliding_window, 0),
            )
        else:
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
                query_states, key_states, value_states, \
                attention_mask[:, -key_states.shape[1]:] if attention_mask is not None \
                    else torch.ones(key_states.shape[:2]).to(key_states), q_len
            )
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=self.attention_dropout if self.training else 0.0,
                softmax_scale=key_states.shape[-1] ** (-0.5),
                causal=True,
                window_size=(self.sliding_window, 0),
            )
            attn_output = pad_input(attn_output_unpad, indices_q, bsz, q_len)

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_values

def quantize_sym(x, bit=8):
    q_max = (1 << (bit - 1)) - 1
    q_min = -(1 << (bit - 1))
    org_dtype = x.dtype
    scale = x.abs().amax([-1], keepdim=True).float() / (2**(bit-1)-1)
    scale = scale.clamp(min=1e-5, max=1e4)

    x_int = (x / scale).round().clamp(q_min, q_max)
    x = x_int * scale
    
    return x.to(org_dtype)
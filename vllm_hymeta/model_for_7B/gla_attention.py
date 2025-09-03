
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange, repeat
from transformers.activations import ACT2FN

from vllm.forward_context import get_forward_context
from vllm.attention import AttentionMetadata
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size
)
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm.model_executor.layers.linear import (
    # ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig
)

from fla.ops.gla import fused_chunk_gla, chunk_gla

from .gla_cache import GLACacheParams
from .configuration_gla_swa import GLAswaConfig
from .my_fused_recurrent import my_fused_recurrent_gla


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    if hidden_states.dim == 4:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape 
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
    # [num_tokens, num_heads, head_dim]
    assert hidden_states.dim() == 3, \
        "hidden_states should be 3D tensor, but got {}".format(hidden_states.dim())
    num_tokens, num_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :].expand(num_tokens, num_heads, n_rep, head_dim)
    return hidden_states.reshape(num_tokens, num_heads * n_rep, head_dim)

class LowRankSequentialLinear(nn.Module):

    def __init__(
        self,
        input_size: int = 1024,
        low_rank_dim: int = 16,
        output_size: int = 1024,
        bias0: bool = True,
        bias1: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "low_rank_sequential_linear"
    ):
        super().__init__()
        self.input_size = input_size
        self.low_rank_dim = low_rank_dim
        self.output_size = output_size
        # 因为是 low rank 所以就不 parallel 而是直接使用 ReplicatedLinear
        self.linear0 = ReplicatedLinear(
            input_size,
            low_rank_dim,
            bias=bias0,
            quant_config=quant_config,
            prefix=f"{prefix}.0"
        )
        self.linear1 = ReplicatedLinear(
            low_rank_dim,
            output_size,
            bias=bias1,
            quant_config=quant_config,
            prefix=f"{prefix}.1"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.linear0(x)
        x, _ = self.linear1(x)
        return x

class GatedLinearAttention(nn.Module):

    def __init__(
        self,
        config: GLAswaConfig,
        hidden_size: int = 1024,
        num_heads: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        gate_logit_normalizer: int = 16,
        gate_low_rank_dim: int = 16,
        layer_idx: int = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "gla_attn",
        **kwargs
    ) -> None:
        super().__init__()
        assert use_short_conv is False, "GatedLinearAttention does not support short conv yet."

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        self.total_num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // num_key_value_heads

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = self.hidden_size // num_heads
        self.layer_idx = layer_idx

        assert self.hidden_size % num_heads == 0, f"hidden dim must be divisible by num_heads of {num_heads}"

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        assert self.num_heads % self.tp_size == 0, f"num_heads {self.num_heads} must be divisible by tensor model parallel size {self.tp_size}"
        self.tp_heads = self.num_heads // self.tp_size # 每个 GPU 负责的头数

        # https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py
        if self.total_num_key_value_heads >= self.tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            # 即每个 GPU 都有多个 q heads 和多个 kv heads
            assert self.total_num_key_value_heads % self.tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            # 即每个 GPU 有多个 q heads，每个 GPU 只负责一个 kv head 但是一个 kv head 会被多个 GPU 负责
            assert self.tp_size % self.total_num_key_value_heads == 0

        self.tp_kv_heads = max(1, self.total_num_key_value_heads // self.tp_size)
        self.q_size = self.tp_heads * self.head_dim
        self.kv_size = self.tp_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.total_num_key_value_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj"
        )
        
        self.gk_proj = LowRankSequentialLinear(
            input_size=hidden_size,
            low_rank_dim=gate_low_rank_dim,
            output_size=self.total_num_key_value_heads * self.head_dim,
            bias0=False,
            bias1=True,
            quant_config=None,
            # quant_config=quant_config,
            prefix=f"{prefix}.gk_proj"
        )
        self.gate_logit_normalizer = gate_logit_normalizer
        self.feature_map = ACT2FN['relu']
        
        self.o_proj = RowParallelLinear(
            input_size=self.num_heads * self.head_dim,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj"
        )
        self.g_norm = RMSNorm(hidden_size=self.head_dim, eps=config.norm_eps)

    def _prefill_and_mix_infer(self, q, k, v, gk, kv_cache, 
                               state_indices_tensor, attn_metadata):
        hidden = []
        for _prefill_idx in range(getattr(attn_metadata, "num_prefills", 0)):
            if _prefill_idx >= len(attn_metadata.query_start_loc):
                break
            if _prefill_idx >= len(state_indices_tensor):
                break
            
            _start = attn_metadata.query_start_loc[_prefill_idx]
            _end = attn_metadata.query_start_loc[_prefill_idx + 1]
            slot_id = state_indices_tensor[_prefill_idx]

            q_slice = q[_start:_end].transpose(0, 1).contiguous() # [num_heads, num_tokens, head_dim]
            k_slice = k[_start:_end].transpose(0, 1).contiguous()
            v_slice = v[_start:_end].transpose(0, 1).contiguous()
            gk_slice = gk[_start:_end].transpose(0, 1).contiguous()
            initial_state = kv_cache[slot_id, ...].unsqueeze(0) # [1, num_heads, head_dim, head_dim]
            if initial_state.isnan().any(): # 未初始化的 kv_cache 可能有 nan
                initial_state = None
            
            should_pad_dim = q_slice.dim() == 3
            if should_pad_dim:
                q_slice = q_slice.unsqueeze(0) # [1, num_heads, num_tokens, head_dim]
                k_slice = k_slice.unsqueeze(0)
                v_slice = v_slice.unsqueeze(0)
                gk_slice = gk_slice.unsqueeze(0)
            o, recurrent_state = fused_chunk_gla(
                q_slice, k_slice, v_slice, gk_slice, initial_state=initial_state, output_final_state=True)
            # 因为我自定义的 cache 是 kv, 但是 gla 的 recurrent 是 vk，但是 gla 读的时候又本来就反过来读的，所以传的时候不用 transpose
            kv_cache[slot_id].copy_(recurrent_state.squeeze(0)) # [1, num_heads, head_dim, head_dim]
            o = self.g_norm(o)
            hidden.append(rearrange(o.squeeze(0), 'h n d -> n (h d)').contiguous())
        if attn_metadata.num_decode_tokens > 0: # 有需要解码的 tokens
            hidden.append(
                self._decode(q, k, v, gk, kv_cache, 
                    state_indices_tensor, attn_metadata)
            )
        if not hidden:
            return torch.empty((0, q.size(-1)), device=q.device, dtype=q.dtype)
        
        hidden = torch.concat(hidden, dim=0).contiguous() # [num_tokens, h*d]
        return hidden
    
    def _decode(self, q, k, v, gk, kv_cache, state_indices_tensor, attn_metadata):
        # 因为解码的请求的长度都是 1， 这里的 num_tokens 维度等效于 batch 维度(同时处理多个解码请求)
        _start = attn_metadata.num_prefill_tokens
        q = q[_start:].unsqueeze(2).contiguous() # [batch=num_tokens, num_heads, seqlen=1, head_dim]
        k = k[_start:].unsqueeze(2).contiguous()
        v = v[_start:].unsqueeze(2).contiguous()
        gk = gk[_start:].unsqueeze(2).contiguous()
        slot_ids = state_indices_tensor[getattr(attn_metadata, "num_prefills", 0):]
        
        o = my_fused_recurrent_gla(q, k, v, gk, kv_caches=kv_cache, slot_idx=slot_ids) # [num_tokens, num_heads, head_dim]
        o = self.g_norm(o)
        o = rearrange(o, 'b h d -> b (h d)').contiguous() # [num_tokens, h*d]
        return o

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_caches: Optional[GLACacheParams] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        **kwargs
    ):
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        qkv, _ = self.qkv_proj(hidden_states)

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        gk = self.gk_proj(hidden_states)
        
        q = self.feature_map(q)
        k = self.feature_map(k)

        q = q.reshape(q.shape[0], -1, self.head_dim) # [num_tokens, num_heads, head_dim]
        k = k.reshape(k.shape[0], -1, self.head_dim) # [num_tokens, num_heads, head_dim]
        v = v.reshape(v.shape[0], -1, self.head_dim) # [num_tokens, num_heads, head_dim]
        gk = gk.reshape(gk.shape[0], -1, self.head_dim)
        
        start_head = self.tp_kv_heads * self.tp_rank
        end_head = start_head + self.tp_kv_heads
        gk = gk[:, start_head:end_head, :]
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer

        # 这里之后可以优化不 repeat_kv 
        k = repeat_kv(k, self.num_key_value_groups) # [num_tokens, num_heads, head_dim]
        v = repeat_kv(v, self.num_key_value_groups)
        gk = repeat_kv(gk, self.num_key_value_groups)

        kv_cache = kv_caches.gla_cache
        state_indices_tensor = kv_caches.state_indices_tensor

        decode_only = getattr(attn_metadata, "num_prefills", 0) == 0
        if not decode_only:
            outputs = self._prefill_and_mix_infer(
                q, k, v, gk, kv_cache, state_indices_tensor, attn_metadata
            )
        else:
            outputs = self._decode(
                q, k, v, gk, kv_cache, state_indices_tensor, attn_metadata
            )
        
        outputs, _ = self.o_proj(outputs) # [num_tokens, hidden_size]
        return outputs
        

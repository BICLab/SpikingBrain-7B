from typing import Optional, Tuple
import warnings

import torch
import torch.nn.functional as F
from flash_attn.flash_attn_interface import _flash_attn_forward
# from flash_attn.flash_attn_interface import flash_attn_with_kvcache
from flash_attn.flash_attn_interface import _flash_attn_varlen_forward

from einops import rearrange, einsum

#################

from typing import Optional, Union

import os

# isort: off
# We need to import the CUDA kernels after importing torch
import flash_attn_2_cuda as flash_attn_cuda

# isort: on
if 'MHA_INPUT_COLLECTION' in os.environ:
    from flash_attn.tuning.modules.yaml_writer import YamlWriter

def my_flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    softcap=0.0,  # 0.0 means deactivated
):
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    maybe_contiguous = lambda x: x.contiguous() if x is not None and x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)
    cache_batch_idx = maybe_contiguous(cache_batch_idx)
    block_table = maybe_contiguous(block_table)

    if 'MHA_INPUT_COLLECTION' in os.environ:
        writer = YamlWriter()
        if writer:
            problem = dict(
                q = q,
                k_cache = k_cache,
                v_cache = v_cache,
                k = k,
                v = v,
                seqlens_k = cache_seqlens,
                rotary_cose = rotary_cos,
                rotary_sin = rotary_sin,
                cache_batch_idx = cache_batch_idx,
                block_table = block_table,
                alibi_slopes = alibi_slopes,
                softmax_scale = softmax_scale,
                causal = causal,
                window_size_left = window_size[0],
                window_size_right = window_size[1],
                rotary_interleaved = rotary_interleaved,
                num_splits = num_splits,
                softcap = softcap
            )
            writer.write("flash_attn_with_kvcache", q.dtype == torch.bfloat16, problem)
    out, softmax_lse = flash_attn_cuda.fwd_kvcache(
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_seqlens,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        cache_leftpad,
        block_table,
        alibi_slopes,
        None,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        rotary_interleaved,
        num_splits,
    )
    return out, softmax_lse

##################

@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor, # [batch_size, seq_len, num_heads, head_dim]
    lse: torch.Tensor, # [batch_size, num_heads, seq_len]
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        out_dtype = out.dtype
        out, block_out = out.to(torch.float32), block_out.to(torch.float32)
        new_lse = lse - F.logsigmoid(lse - block_lse)
        out = out - F.sigmoid(block_lse - lse).transpose(1, 2)[..., None] * (out - block_out)
    return out.to(out_dtype), new_lse

class FlashAttentionWithMetaToken(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q1,
        q2,
        k2,
        v2,
        key_cache,
        value_cache,
        cache_seqlens,
        num_meta_tokens,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        block_tables,
        softcap,
        attn_mask,
        deterministic,
        return_attn_probs,
    ):
        Lq, Lk = q1.shape[0], key_cache.shape[0]
        if softmax_scale is None:
            softmax_scale = q1.shape[-1] ** (-0.5)
        assert causal and window_size[1] in [0, -1]
        assert num_meta_tokens == k2.shape[0]
        if q2 is not None:
            assert Lq == Lk, "meta_tokens' query doesn't support decoding"
            assert num_meta_tokens == q2.shape[0]

        out1 = my_flash_attn_with_kvcache(
            q=q1.unsqueeze(1), # q must have shape (batch_size, seqlen_q, num_heads, head_size_og)
            k_cache=key_cache, # key_cache.shape: torch.Size([32092, 16, 2, 128]
            v_cache=value_cache,
            block_table=block_tables,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=(window_size[0], -1),
            alibi_slopes=alibi_slopes,
            softcap=softcap,
        )

        out1, lse1 = out1[0], out1[1] # [batch_size=decode_one_nums, seq_len=1, num_heads, head_dim], [batch_size=decode_one_nums, num_heads, seq_len=1]
        # warnings.warn(f"Shape of out1: {out1.shape}, lse1: {lse1.shape}, q1: {q1.shape}, k2: {k2.shape}, v2: {v2.shape}")
        q = torch.cat((q1, q2), dim=0) if q2 is not None else q1
        q = q.to(torch.bfloat16)
        out2 = _flash_attn_forward(
            q.unsqueeze(0),
            k2.unsqueeze(0),
            v2.unsqueeze(0),
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            # window_size=(-1, -1),
            alibi_slopes=alibi_slopes,
            # attn_mask=attn_mask,
            return_softmax=return_attn_probs,
            softcap=softcap,
        )
        out2, lse2 = out2[0], out2[1] # [seq_len=1, batch_size=decode_one_nums, num_heads, head_dim], [seq_len=1, num_heads, batch_size=decode_one_nums]
        out2 = out2.transpose(0, 1) # [batch_size=decode_one_nums, seq_len=1, num_heads, head_dim]
        lse2 = lse2.transpose(0, 2) # [batch_size=decode_one_nums, num_heads, seq_len=1]
        out, lse = _update_out_and_lse(out1, lse1, out2[:,:Lq], lse2[:,:,:Lq])
        out = torch.cat((out2[:, Lq:], out), dim=1) if q2 is not None else out
        return out



class FlashAttentionVarlenWithMetaToken(torch.autograd.Function):

    @staticmethod
    def forward( 
        ctx,
        q1,
        q2,
        k1,
        k2,
        v1,
        v2,
        cu_seqlen_q,
        cu_seqlen_k,
        max_seqlen_q,
        max_seqlen_k,
        num_meta_tokens,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        block_tables,
        softcap,
        attn_mask,
        deterministic,
        return_attn_probs,
    ):
        Lq, Lk = q1.shape[0], k1.shape[0] # [num_tokens, num_heads, head_dim]
        if softmax_scale is None:
            softmax_scale = q1.shape[-1] ** (-0.5)
        assert causal and window_size[1] in [0, -1]
        if k2 is not None:
            assert num_meta_tokens == k2.shape[0]
        if q2 is not None:
            assert Lq == Lk, "meta_tokens' query doesn't support decoding"
            assert num_meta_tokens == q2.shape[0]
        
        out1 = _flash_attn_varlen_forward(
            q1,
            k1,
            v1,
            cu_seqlens_q=cu_seqlen_q,
            cu_seqlens_k=cu_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=True,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            # window_size=(window_size[0], -1), #  window_size[0] already minus 1 at MetaAttentionImpl
            alibi_slopes=alibi_slopes,
            return_softmax=return_attn_probs,
            softcap=softcap,
            block_table=block_tables,
        )
        out1, lse1 = out1[0].unsqueeze(0), out1[1] # flash_attn 2.7.3 out[1], 2.6.3 out[5]
        # lse1 = rearrange(lse1, "b h l -> 1 h (b l)").contiguous()
        lse1 = lse1.unsqueeze(0).contiguous()  # [1, num_heads, num_tokens]
 
        q = torch.cat((q1, q2), dim=0) if q2 is not None else q1 # [num_tokens, num_heads, head_dim]
        q = q.to(torch.bfloat16)
        
        out2 = _flash_attn_forward(
            q.unsqueeze(0), # q must have shape (batch_size, seqlen_q, num_heads, head_size_og)
            k2.unsqueeze(0),
            v2.unsqueeze(0),
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=False, # ? 官方实现不一样 https://github.com/NVlabs/hymba/blob/main/barebones_hymba/barebones_hymba_block.py
            window_size_left=-1,
            window_size_right=-1,
            # window_size=(-1, -1),
            alibi_slopes=alibi_slopes,
            # attn_mask=attn_mask,
            return_softmax=return_attn_probs,
            softcap=softcap,
        )
        out2, lse2 = out2[0], out2[1] # [1, num_tokens, num_heads, head_dim], [1, num_heads, num_tokens]
        out, lse = _update_out_and_lse(out1, lse1, out2[:,:Lq], lse2[:,:,:Lq])
        # 注意这里要把 meta token 即 out2[:, Lq:] 输出放前面
        out = torch.cat((out2[:, Lq:], out), dim=1) if q2 is not None else out
        return out

def metatoken_flash_attn_with_kvcache(
    q1,
    q2,
    k2,
    v2,
    key_cache=None,
    value_cache=None,
    cache_seqlens=None,
    num_meta_tokens=128,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    block_tables=None,
    softcap=0.0,
    deterministic=False,
    return_attn_probs=False,
    attn_mask=None,
):
    return FlashAttentionWithMetaToken.apply(
        q1,
        q2,
        k2,
        v2,
        key_cache,
        value_cache,
        cache_seqlens,
        num_meta_tokens,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        block_tables,
        softcap,
        attn_mask,
        deterministic,
        return_attn_probs,
    )


def metatoken_flash_attn_varlen_func(
    q1,
    q2,
    k1,
    k2,
    v1,
    v2,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    num_meta_tokens=128,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    block_tables=None,
    softcap=0.0,
    deterministic=False,
    return_attn_probs=False,
    attn_mask=None,
):
    return FlashAttentionVarlenWithMetaToken.apply(
        q1,
        q2,
        k1,
        k2,
        v1,
        v2,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_meta_tokens,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        block_tables,
        softcap,
        attn_mask,
        deterministic,
        return_attn_probs,
    )


def naive_metatoken_flash_attn(
    q1,
    q2,
    k1,
    k2,
    v1,
    v2,
    softmax_scale=None,
    num_meta_tokens=128,
    sliding_window=(-1, 0),
):
    Lq, Lk = q1.shape[1], k1.shape[1]
    q_id = Lk - Lq + torch.arange(0, Lq)
    k_id = torch.arange(0, Lk)
    causal_mask = q_id[:, None] >= k_id[None, :]
    sliding_mask = (q_id[:, None] - k_id[None, :]) < sliding_window[0]
    mask = torch.ones((Lq, Lk + num_meta_tokens), dtype=torch.bool)
    mask[:, num_meta_tokens:] = causal_mask & sliding_mask

    if q2 is not None:
        q = torch.cat((q1, q2), dim=1)
        mask_meta = torch.cat((torch.ones(num_meta_tokens, num_meta_tokens, dtype=torch.bool), \
                               torch.zeros(num_meta_tokens, Lk, dtype=torch.bool)), dim=1)
        mask = torch.cat((mask, mask_meta), dim=0)
    else:
        q = q1

    k = torch.cat([k2, k1], 1)
    v = torch.cat([v2, v1], 1)
    qk = einsum(q, k, "b q h d, b k h d -> b h q k") * softmax_scale
    qk = torch.where(rearrange(mask.cuda(), "q k -> 1 1 q k"), qk, -float("inf"))
    weight = torch.softmax(qk, dim=-1)
    out = einsum(weight, v, "b h q k, b k h d -> b q h d")
    return out

import torch
import triton
from typing import Tuple
import triton.language as tl
from einops import rearrange

from fla.utils import autocast_custom_fwd, contiguous

@triton.jit
def my_fused_recurrent_fwd_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr,
    output_ptr, 
    D: tl.constexpr,
    kv_cache_ptr, 
    slot_idx,
    scale, # D ** -0.5
    qkv_b_stride, qkv_h_stride,
    cache_b_stride, cache_h_stride,
    cache_d0_stride, cache_d1_stride,
    BLOCK_SIZE: tl.constexpr
):
    """
    Kernel for linear attention decoding with KV cache.
    
    This kernel computes attention for a single token using the KV cache.
    """
    pid_b = tl.program_id(0)  # batch index
    pid_h = tl.program_id(1)  # head index
    pid_d = tl.program_id(2)  # dimension block index

    # Load slot index for the current batch
    slot_idx = tl.load(slot_idx + pid_b).to(tl.int64)

    # Skip if slot_idx is -1 (indicating no cache)
    if slot_idx == -1:
        return
    
    batch_id = pid_b
    head_id = pid_h

    # Load decay factor g

    # Caculate offsets for dimensions
    qk_d_offsets = tl.arange(0, D) # [D]
    v_d_offsets = tl.arange(0, BLOCK_SIZE) + pid_d * BLOCK_SIZE # [BLOCK_SIZE]
    cache_d_offsets = qk_d_offsets[:, None] * cache_d0_stride + v_d_offsets[None, :] * cache_d1_stride # [D, BLOCK_SIZE]

    # Caculate offsets for the current batch and head
    q_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride
    k_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride
    v_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride

    cache_offset = slot_idx * cache_b_stride + head_id * cache_h_stride

    # Create masks for loading tensors
    qk_mask = qk_d_offsets < D
    v_mask = v_d_offsets < D

    # Load query, key, value, and decay factor
    q = tl.load(q_ptr + q_offset + qk_d_offsets, mask=qk_mask, other=0.0) * scale
    k = tl.load(k_ptr + k_offset + qk_d_offsets, mask=qk_mask, other=0.0)
    v = tl.load(v_ptr + v_offset + v_d_offsets, mask=v_mask, other=0.0)
    g = tl.load(g_ptr + k_offset + qk_d_offsets, mask=qk_mask, other=0.0).to(tl.float32)
    # Compute key-value outer product
    kv_outer = k[:, None] * v[None, :] # [D, BLOCK_SIZE]
    kv_mask = qk_mask[:, None] & v_mask[None, :]

    # Apply decay factor to previous KV cache
    ratio = tl.exp(g[:, None])
    kv_ptr = kv_cache_ptr + cache_offset + cache_d_offsets
    kv_cache_old = tl.load(kv_ptr, mask=kv_mask, other=0.0)
    kv_cache_new = kv_outer + ratio * kv_cache_old # [D, BLOCK_SIZE]

    # Compute attention output
    output = q[:, None].to(tl.float32) * kv_cache_new
    output = tl.sum(output, axis=0) # [BLOCK_SIZE]

    # Update KV cache and store output
    tl.store(kv_ptr, kv_cache_new.to(kv_ptr.dtype.element_ty), mask=kv_mask)
    tl.store(output_ptr + q_offset + v_d_offsets, output.to(output_ptr.dtype.element_ty), mask=v_mask)


class MyFusedRecurrentFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, g, kv_caches, slot_idx, scale=None, initial_state=None, output_final_state=False, reverse=False):
        B, H, _, D = q.shape

        # used for decoding, so we assume the sequence length is 1
        assert k.shape == (B, H, 1, D)
        assert v.shape == (B, H, 1, D)

        # default scale
        if scale is None:
            scale = D ** -0.5

        output = torch.empty_like(q)

        BLOCK_SIZE = 64
        grid = (B, H, D // BLOCK_SIZE)

        qkv_b_stride, qkv_h_stride = q.stride(0), q.stride(1)

        cache_b_stride, cache_h_stride = kv_caches.stride(0), kv_caches.stride(1)
        cache_d0_stride, cache_d1_stride = kv_caches.stride(2), kv_caches.stride(3)

        my_fused_recurrent_fwd_kernel[grid](
            q, k, v, g, output, D,
            kv_caches, slot_idx, scale,
            qkv_b_stride, qkv_h_stride,
            cache_b_stride, cache_h_stride,
            cache_d0_stride, cache_d1_stride,
            BLOCK_SIZE,
        )
        output = output.transpose(1, 2).contiguous() # [b, h, l, d] --> [b, l, h, d]
        # output = rearrange(output, 'b h l d -> b l (h d)')
        return output.squeeze(1).contiguous() # (b, h, d), remove the sequence length dimension because it is 1

        

def my_fused_recurrent(
    q,
    k,
    v,
    g=None,
    kv_caches: torch.Tensor = None,
    slot_idx: torch.Tensor = None,
    scale=None,
    initial_state=None,
    output_final_state=False,
    reverse=False
):
    return MyFusedRecurrentFunction.apply(q, k, v, g, kv_caches, slot_idx, scale, initial_state, output_final_state, reverse)


def my_fused_recurrent_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor = None,
    kv_caches: torch.Tensor = None,
    slot_idx: torch.Tensor = None,
    scale: int = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    reverse: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    o = my_fused_recurrent(q, k, v, g, kv_caches, slot_idx, scale, initial_state, output_final_state, reverse)
    return o
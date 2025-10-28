
    
# """
# Linear attention classes
# """
import sys
from typing import List, Tuple, Optional
import copy
import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
import math
from fla.modules.activations import ACT2FN
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn, fused_recurrent_linear_attn
from fla.ops.gla import fused_chunk_gla, fused_recurrent_gla, chunk_gla



def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    直接在 heads 维度上重复键值对，避免了额外的转置操作。
    输入张量形状: (batch, seq_len, num_key_value_heads, head_dim)
    输出张量形状: (batch, seq_len, num_attention_heads, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    
    # 1. 在 heads 维度后增加一个新维度 -> (batch, slen, num_key_value_heads, 1, head_dim)
    hidden_states = hidden_states.unsqueeze(3)
    
    # 2. 扩展新维度 n_rep 次 -> (batch, slen, num_key_value_heads, n_rep, head_dim)
    hidden_states = hidden_states.expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    
    # 3. 将 num_key_value_heads 和 n_rep 维度合并 -> (batch, slen, num_key_value_heads * n_rep, head_dim)
    #    num_key_value_heads * n_rep 等于 num_attention_heads
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


class GLA(nn.Module):
    def __init__(self, hidden_size, num_heads, num_key_value_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads  # Assuming same heads for simplicity
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.norm = nn.LayerNorm(self.head_dim)
        self.do_feature_map_norm = True
        self.g_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.g_norm = nn.LayerNorm(self.head_dim)
        self.gate_fn = torch.nn.SiLU()
        gate_low_rank_dim = 16
        self.gk_proj = nn.Sequential(nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
                                     nn.Linear(gate_low_rank_dim, self.num_key_value_groups, bias=True))
    def forward(self, q,k,v,hidden_states,initial_state):
        b, l, _,_ =q.shape

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        gk = self.gk_proj(hidden_states)
        gk = gk.unsqueeze(2) 
        # -> (b, l, 1, num_key_value_groups, 1)
        gk = gk.unsqueeze(-1)
        
        # 3. 使用 expand 将其高效广播到目标形状
        # 目标形状: (b, l, num_key_value_heads, num_key_value_groups, head_dim)
        gk = gk.expand(b, l, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
        
        # 4. 合并最后两个分组维度，得到最终形状
        # -> (b, l, num_heads, head_dim)
        gk = gk.reshape(b, l, self.num_heads, self.head_dim)
        gk = F.logsigmoid(gk) / 16
        # print(q.shape,k.shape,v.shape,gk.shape)
        is_recurrent_mode = l == 1 and initial_state is not None

        if is_recurrent_mode:
            o, final_state = fused_recurrent_gla(
                q=q,
                k=k,
                v=v,
                g=gk,
                initial_state=initial_state,
                output_final_state=True
                )

        else:
            o, final_state = chunk_gla(
                q=q,
                k=k,
                v=v,
                g=gk,
                initial_state=initial_state,
                output_final_state=True
                )
        g = self.g_proj(hidden_states)
        # o, final_state = chunk_linear_attn(
        #         q=q,
        #         k=k,
        #         v=v,
        #         normalize=self.do_feature_map_norm,
        #     )
        o = self.g_norm(o)
        g = rearrange(g, 'b l (h d) -> b l h d', h=self.num_heads)
        o = o * self.gate_fn(g) 
        # o = self.norm(o)
        o = o.view(b, l, -1)
       
        return o, final_state


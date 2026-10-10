import torch
from typing import Dict, Optional, Tuple


class SpikingStateCache:
    """
    State cache storing layer-wise recurrent states for autoregressive decoding:
    1. Membrane potentials (V_mem) for attention & MLP neurons
    2. Associative linear attention state accumulator M and normalizer z
    """
    def __init__(self):
        self.membrane_states: Dict[str, torch.Tensor] = {}
        self.linear_attn_m: Dict[int, torch.Tensor] = {}  # Layer index -> [B, H, Dk, Dv]
        self.linear_attn_z: Dict[int, torch.Tensor] = {}  # Layer index -> [B, H, Dk]

    def get_membrane(self, key: str) -> Optional[torch.Tensor]:
        return self.membrane_states.get(key)

    def set_membrane(self, key: str, value: torch.Tensor):
        self.membrane_states[key] = value.detach()

    def get_attn_state(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.linear_attn_m.get(layer_idx), self.linear_attn_z.get(layer_idx)

    def set_attn_state(self, layer_idx: int, m: torch.Tensor, z: torch.Tensor):
        self.linear_attn_m[layer_idx] = m.detach()
        self.linear_attn_z[layer_idx] = z.detach()

    def reset(self):
        self.membrane_states.clear()
        self.linear_attn_m.clear()
        self.linear_attn_z.clear()

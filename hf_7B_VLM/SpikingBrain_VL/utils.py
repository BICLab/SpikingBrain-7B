# state_holder.py

import torch
from typing import Optional, Tuple, Dict

class RecurrentStateHolder:
    def __init__(self):
        self.states: Dict[int, Optional[Tuple[torch.Tensor, ...]]] = {}
        print("RecurrentStateHolder initialized.")

    def set(self, layer_idx: int, state: Tuple[torch.Tensor, ...]):
        
        self.states[layer_idx] = state

    def get(self, layer_idx: int, cache_position: torch.LongTensor) -> Optional[Tuple[torch.Tensor, ...]]:
        
        if cache_position is not None and cache_position[0] == 0:
            if self.states:
                # print(f"New sequence detected. Resetting {len(self.states)} stored states.")
                self.states.clear()
            return None
        return self.states.get(layer_idx, None)

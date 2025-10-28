# state_holder.py

import torch
from typing import Optional, Tuple, Dict

class RecurrentStateHolder:
    """
    一个用于在模型各层之间携带和管理循环状态的容器。
    它通过 `cache_position` 自动检测新序列的开始，并重置状态。
    """
    def __init__(self):
        # 使用字典来存储每一层的状态，键是 layer_idx
        self.states: Dict[int, Optional[Tuple[torch.Tensor, ...]]] = {}
        print("RecurrentStateHolder initialized.")

    def set(self, layer_idx: int, state: Tuple[torch.Tensor, ...]):
        """为指定层设置新的状态。"""
        self.states[layer_idx] = state

    def get(self, layer_idx: int, cache_position: torch.LongTensor) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        为指定层获取状态。
        如果 cache_position 表明这是序列的第一个token，则自动重置所有状态。
        """
        # `cache_position[0] == 0` 是一个明确的信号，表示预填充（prefill）阶段或新序列的开始
        if cache_position is not None and cache_position[0] == 0:
            if self.states:
                # print(f"New sequence detected. Resetting {len(self.states)} stored states.")
                self.states.clear()
            return None
        
        # 返回已存储的该层状态，如果不存在则返回 None
        return self.states.get(layer_idx, None)
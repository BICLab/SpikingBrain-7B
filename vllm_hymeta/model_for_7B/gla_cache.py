
from dataclasses import dataclass

import torch

from vllm.model_executor.models.constant_size_cache import ConstantSizeCache

@dataclass
class GLACacheParams:
    gla_cache: torch.Tensor = torch.Tensor()
    state_indices_tensor: torch.Tensor = torch.Tensor()

    def at_layer_idx(self, layer_idx):
        return GLACacheParams(self.gla_cache[layer_idx, ...],
                              self.state_indices_tensor)
    
class GLACacheManager(ConstantSizeCache):

    def __init__(self, dtype, cache_shape):
        super().__init__(cache_shape[1]) # max_batch_size is cache_shape[1]
        self._gla_cache = torch.empty(size=cache_shape,
                                      dtype=dtype,
                                      device="cuda")
    
    @property
    def cache(self):
        return self._gla_cache
    
    def _copy_cache(self, from_index: int, to_index: int):
        assert len(self.cache) > 0
        for cache_t in self.cache:
            cache_t[:, to_index].copy_(cache_t[:, from_index],
                                       non_blocking=True)
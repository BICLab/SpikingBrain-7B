import math
import torch
import torch.nn as nn
from typing import List, Optional


class SpikeLoRALinear(nn.Module):
    """
    Spike-compatible Low-Rank Adaptation (LoRA) layer.
    W = W_0 + (alpha / r) * (B @ A)
    
    Operates in the pre-synaptic continuous potential domain prior to membrane thresholding,
    enabling surrogate gradient backpropagation to seamlessly adjust low-rank matrices
    without destabilizing discrete Heaviside spike firing.
    """
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.merged = False

        # Freeze base parameters
        for param in self.base_layer.parameters():
            param.requires_grad = False

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
            self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
            self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform for A, zeros for B (starts as exact identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self.r == 0 or self.merged:
            return base_out

        # Adapter path: (x @ A^T) @ B^T * scaling
        adapter_out = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + adapter_out

    def merge(self):
        """Merges LoRA adapter weights directly into base linear weights for zero-overhead inference."""
        if not self.merged and self.r > 0:
            weight_delta = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += weight_delta
            self.merged = True

    def unmerge(self):
        """Unmerges adapter weights from base linear weights."""
        if self.merged and self.r > 0:
            weight_delta = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data -= weight_delta
            self.merged = False


def inject_spike_lora(
    model: nn.Module,
    r: int = 8,
    lora_alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
) -> int:
    """
    Recursively replaces target linear modules with SpikeLoRALinear layers.
    Returns the count of injected adapter modules.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    injected_count = 0
    for name, module in model.named_children():
        for target_name in target_modules:
            if hasattr(module, target_name):
                target_linear = getattr(module, target_name)
                if isinstance(target_linear, nn.Linear):
                    lora_layer = SpikeLoRALinear(target_linear, r=r, lora_alpha=lora_alpha)
                    setattr(module, target_name, lora_layer)
                    injected_count += 1
        # Recurse down children
        injected_count += inject_spike_lora(module, r=r, lora_alpha=lora_alpha, target_modules=target_modules)
    return injected_count


def merge_spike_lora(model: nn.Module):
    """Recursively merges all SpikeLoRALinear layers throughout the model."""
    for module in model.modules():
        if isinstance(module, SpikeLoRALinear):
            module.merge()

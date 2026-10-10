import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class FastSigmoidSurrogate(torch.autograd.Function):
    """
    Fast Sigmoid surrogate gradient function for non-differentiable Heaviside step:
    sigma'(x) = alpha / (2 * (1 + |alpha * x|)^2)
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        denominator = 1.0 + torch.abs(alpha * x)
        surrogate_grad = alpha / (2.0 * (denominator * denominator))
        return grad_output * surrogate_grad, None


class ATanSurrogate(torch.autograd.Function):
    """
    Arctan surrogate gradient function:
    sigma'(x) = 1 / (pi * (1 + (alpha * x)^2))
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        ax = alpha * x
        surrogate_grad = alpha / (math.pi * (1.0 + ax * ax))
        return grad_output * surrogate_grad, None


def get_surrogate(name: str = "fast_sigmoid"):
    if name == "atan":
        return ATanSurrogate.apply
    return FastSigmoidSurrogate.apply


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) Neuron with surrogate gradient backprop
    and stateful autoregressive membrane potential tracking.
    """
    def __init__(
        self,
        v_threshold: float = 1.0,
        leak_factor: float = 0.9,
        surrogate_alpha: float = 2.0,
        surrogate_function: str = "fast_sigmoid",
        reset_mode: str = "hard",
    ):
        super().__init__()
        self.v_threshold = float(v_threshold)
        self.leak_factor = float(leak_factor)
        self.surrogate_alpha = float(surrogate_alpha)
        self.reset_mode = reset_mode
        self.surrogate_fn = get_surrogate(surrogate_function)

    def forward(
        self,
        current: torch.Tensor,
        v_mem: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for either single token step (T=1) or sequential chunk.
        current: Tensor of shape (..., D)
        v_mem: Initial membrane potential of matching shape (..., D)
        Returns:
            spikes: Binary spike activation tensor {0, 1}
            v_mem: Updated membrane potential tensor
        """
        if v_mem is None:
            v_mem = torch.zeros_like(current)

        # Membrane charging: V[t] = leak * V[t-1] + I[t]
        v_next = self.leak_factor * v_mem + current

        # Spike firing: Heaviside step via surrogate
        delta_v = v_next - self.v_threshold
        spikes = self.surrogate_fn(delta_v, self.surrogate_alpha)

        # Membrane reset: hard or soft
        if self.reset_mode == "hard":
            v_mem_out = v_next * (1.0 - spikes)
        else:
            v_mem_out = v_next - (self.v_threshold * spikes)

        return spikes, v_mem_out


class ALIFNeuron(nn.Module):
    """
    Adaptive Leaky Integrate-and-Fire (ALIF) Neuron.
    Emulates biological spike-frequency adaptation by dynamically raising threshold
    upon firing, stabilizing long-sequence attention representations.
    """
    def __init__(
        self,
        v_threshold: float = 1.0,
        leak_factor: float = 0.9,
        adaptation_leak: float = 0.95,
        adaptation_scale: float = 0.1,
        surrogate_alpha: float = 2.0,
        surrogate_function: str = "fast_sigmoid",
        reset_mode: str = "hard",
    ):
        super().__init__()
        self.v_threshold = float(v_threshold)
        self.leak_factor = float(leak_factor)
        self.adaptation_leak = float(adaptation_leak)
        self.adaptation_scale = float(adaptation_scale)
        self.surrogate_alpha = float(surrogate_alpha)
        self.reset_mode = reset_mode
        self.surrogate_fn = get_surrogate(surrogate_function)

    def forward(
        self,
        current: torch.Tensor,
        v_mem: Optional[torch.Tensor] = None,
        adaptation_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if v_mem is None:
            v_mem = torch.zeros_like(current)
        if adaptation_state is None:
            adaptation_state = torch.zeros_like(current)

        # Effective adaptive threshold
        effective_threshold = self.v_threshold + (self.adaptation_scale * adaptation_state)

        # Membrane integration
        v_next = self.leak_factor * v_mem + current

        # Spike generation
        delta_v = v_next - effective_threshold
        spikes = self.surrogate_fn(delta_v, self.surrogate_alpha)

        # Reset & Adaptation update
        if self.reset_mode == "hard":
            v_mem_out = v_next * (1.0 - spikes)
        else:
            v_mem_out = v_next - (effective_threshold * spikes)

        next_adaptation = self.adaptation_leak * adaptation_state + spikes

        return spikes, v_mem_out, next_adaptation

import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.ticker import MaxNLocator
class SpikeCountBaseLIFNode(nn.Module):
    def __init__(self,):
        super().__init__()
        self.T = None # Number of timesteps (int)
        self.x_remain = None 
        self.spike_seq = None # Generated spike sequence, shape: (T, *input_shape)
        self.is_bidirectional = False # True if coding is bidirectional (-1/0/1)
        self.is_bitwise_coding = False # True if coding is bitwise representation
        self.is_two_complement = False
        
    def forward(self,):
        """
        Forward pass placeholder for subclasses.

        Args:
            x (torch.Tensor): Input spike count tensor.
            T (int | None): Optional number of timesteps. If None, determine from x.

        Returns:
            torch.Tensor: Spike sequence of shape [T, *x.shape].
        """
        raise NotImplementedError
    
    def neuronal_charge(self,):
        raise NotImplementedError

    def neuronal_fire(self,):
        raise NotImplementedError

    def neuronal_reset(self,):
        raise NotImplementedError

    def firing_rate(self,):
        """
        Calculate the average firing rate of the spike sequence.

        For bidirectional coding: use absolute spike values.
        For unidirectional coding: use raw spike values.

        Returns:
            torch.Tensor: scalar firing rate = total_spikes / (num_elements * T)

        Raises:
            ValueError: if spike_seq or T is not set before calling.
        """
        if self.spike_seq is None:
            raise ValueError("spike_seq is None. Run forward() before calling firing_rate().")
        if self.T is None:
            raise ValueError("T is None. Set self.T before calling firing_rate().")
     
        if self.is_bidirectional:
            return self.spike_seq.abs().sum()/ self.spike_seq.numel() 
        else:
            return self.spike_seq.sum()/ self.spike_seq.numel()     
    
    def visualize_spike(self, max_neurons=30, max_token=20, filename="sample.png", title="", seed=42):
        plt.rcParams.update({'font.size': 10}) 
        if seed is not None:
            random.seed(seed)

        if self.spike_seq is None:
            raise ValueError("spike_seq is None. Run forward() before visualization.")
        if self.spike_seq.dim() != 3:
            raise ValueError("Expected spike_seq of shape [T, N, D]")

        T, N, D = self.spike_seq.shape  # Time, Tokens, Neurons
        seq = self.spike_seq  # [T, N, D]
        seq = seq.permute(1, 0, 2)  # [N, T, D]
        seq = seq.reshape(N * T, D)  # Flatten to [time_steps, neurons]

        total_neurons = D
        sampled = random.sample(range(total_neurons), min(max_neurons, total_neurons))

        max_time = min(N * T, max_token * T)
        plt.figure(figsize=(6, 3))

        for idx, i in enumerate(sampled):
            spikes = (seq[:max_time, i].abs() > 0).nonzero(as_tuple=True)[0]
            plt.scatter(spikes.cpu().numpy(), [idx] * len(spikes), s=2, marker='|', color='black')

        plt.xlabel("Time")
        plt.ylabel("Neuron")
        plt.title(title, fontsize=10)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
        plt.tight_layout()
        plt.savefig(f"png/{filename}")
    

class SpikeCountBinaryLIFNode(SpikeCountBaseLIFNode):
    """
    Spike count to binary spike sequence (0/1) over T timesteps.
    Emits 1 each step until count reaches zero.
    Input x must be a non-negative integer tensor representing spike counts (spike counts ≥ 0).
    """
    def __init__(self,):
        super().__init__()
        self.T = None
        self.x_remain = None
        self.spike_seq = None
        self.is_bidirectional = False   
        self.is_bitwise_coding = False 
        self.is_two_complement = False
    
    def forward(self, x: torch.Tensor, T: int | None = None):
        if not torch.allclose(x, x.round()):
            raise ValueError("Input x must be integer-valued (all elements must be whole numbers).")
        
        if not torch.all(x >= 0):
            raise ValueError("Input x must be non-negative (no values below 0).")

        if T is not None:
            if not isinstance(T, int) or T < 0:
                raise ValueError("T must be a non-negative integer.")
            self.T = T

        if self.T is None:
            self.T = int(x.max().item())

        self.neuronal_charge(x)

        self.spike_seq = torch.zeros((self.T,) + x.shape, dtype=torch.float32, device=x.device)

        for t in range(self.T):
            self.spike_seq[t] = self.neuronal_fire()
            self.neuronal_reset(self.spike_seq[t])

        return self.spike_seq # shape: [T, *x.shape]
    
    def neuronal_charge(self, x: torch.Tensor):
        self.x_remain = x

    def neuronal_fire(self,):
        # Emit 1 if there is remaining count, else 0
        return (self.x_remain > 0).to(torch.float32)
    
    def neuronal_reset(self, spike):
        self.x_remain = self.x_remain - spike

class SpikeCountTernaryLIFNode(SpikeCountBaseLIFNode):
    """
    Spike count to ternary spike sequence (-1/0/1) over T timesteps.
    Positive count emits +1, negative count emits -1.
    Input x must be an integer tensor representing spike counts, and may contain both positive and negative values.
    """
    def __init__(self,):
        super().__init__()
        self.T = None
        self.x_remain = None
        self.spike_seq = None
        self.is_bidirectional = True  
        self.is_bitwise_coding = False 
        self.is_two_complement = False

    def forward(self, x: torch.Tensor, T: int | None = None):
        if not torch.allclose(x, x.round()):
            raise ValueError("Input x must be integer-valued (all elements must be whole numbers).")

        if T is not None:
            if not isinstance(T, int) or T < 0:
                raise ValueError("T must be a non-negative integer.")
            self.T = T

        if self.T is None:
            self.T = int(torch.abs(x).max().item())
         
        self.neuronal_charge(x)

        self.spike_seq = torch.zeros((self.T,) + x.shape, dtype=torch.float32, device=x.device)
        
        for t in range(self.T):
            self.spike_seq[t] = self.neuronal_fire()
            self.neuronal_reset(self.spike_seq[t])

        return self.spike_seq # shape: [T, *x.shape]
    
    def neuronal_charge(self, x: torch.Tensor):
        self.x_remain = x

    def neuronal_fire(self,):
        return torch.sign(self.x_remain) 
        
    def neuronal_reset(self, spike):
        self.x_remain = self.x_remain - spike

class SpikeCountBitwiseNode(SpikeCountBaseLIFNode):
    """
    Spike count to bitwise-coded spike sequence.
    Emits one bit per timestep from the binary representation of the count.
    Supports non-negative integers or signed integers in two's complement or non-complementary binary representation 
    if is_bidirectional=True. 

    In **non-complementary binary representation**:
    - For a positive integer `x`, its binary form is used directly (e.g., `+5 -> [1, 0, 1]`).
    - For a negative integer `-x`, the binary form of the absolute value `x` is used, with `1`s replaced by `-1`s (e.g., `-5 -> [-1, 0, -1]`).

    In **two's complement binary representation** (if `is_two_complement=True`):
    - For positive integers, the binary representation is the same as regular binary (e.g., `+5 -> [0, 1, 0, 1]`).
    - For negative integers, the two's complement representation is used (e.g., `-5 -> [1, 0, 1, 1]`).
      In this case, the most significant bit (MSB) indicates the sign (0 for positive, 1 for negative), and the value is adjusted to reflect the two's complement format.
    """
    def __init__(self, is_bidirectional: bool = False, is_two_complement: bool = False):
        super().__init__()
        self.T = None
        self.x_remain = None
        self.spike_seq = None
        self._bit_idx = 0
        self.is_bidirectional = is_bidirectional  
        self.is_bitwise_coding = True
        self.is_two_complement = is_two_complement

    def forward(self, x: torch.Tensor, T: int | None = None):
        if not torch.allclose(x, x.round()):
            raise ValueError("Input x must be integer-valued (whole numbers).")
        
        x = x.to(torch.int64)

        if self.is_bidirectional:
            x_min = x.min().item()
            x_max = x.max().item()
            x_abs_max = max(abs(x_min), abs(x_max))
        else:
            if not torch.all(x >= 0):
                raise ValueError("Input x must be non-negative when is_bidirectional=False.")
            x_abs_max = x.max().item()

        if T is not None:
            if not isinstance(T, int) or T <= 0:
                raise ValueError("T must be a positive integer.")
            self.T = T
        else:
            if self.is_bidirectional and self.is_two_complement:
                # Use two's complement to represent signed integers, hence +1 for sign bit
                self.T = max(2, math.ceil(math.log2(x_abs_max + 1)) + 1)
            else:
                self.T = max(1, math.ceil(math.log2(x_abs_max + 1)))

        if self.is_bidirectional and not self.is_two_complement:
            negative_mask = x < 0

        self.neuronal_charge(x)

        self.spike_seq = torch.zeros((self.T,) + x.shape, dtype=torch.float32, device=x.device)

        for t in range(self.T):
            spike = self.neuronal_fire()
            self.neuronal_reset(spike)
            if self.is_bidirectional and not self.is_two_complement:
                self.spike_seq[t] = torch.where(negative_mask, -spike, spike)
            else:
                self.spike_seq[t] = spike

        return self.spike_seq  # shape: [T, *x.shape]

    def neuronal_charge(self, x: torch.Tensor):
        if self.is_bidirectional and self.is_two_complement:
            # Convert signed int to two's complement representation within T bits
            mask = (1 << self.T) - 1
            self.x_remain = (x & mask).to(torch.long)
        else:
            self.x_remain = torch.abs(x).to(torch.long)
        self._bit_idx = 0

    def neuronal_fire(self):
        mask = 1 << (self.T - 1 - self._bit_idx)
        return ((self.x_remain & mask) != 0).to(torch.float32)

    def neuronal_reset(self, spike):
        self._bit_idx += 1


def spike_quant(
    x: torch.Tensor, 
    lif_quantizer: SpikeCountBaseLIFNode, 
    x_zero: torch.Tensor = None
) -> torch.Tensor:
    """
    Quantize an integer tensor into spike sequences using a spike encoder.
    Supports optional zero-point shifting for bidirectional or unipolar modes.

    Args:
        x (torch.Tensor): Input integer tensor.
        lif_quantizer (SpikeCountBaseLIFNode): A spike encoder instance.
        x_zero (torch.Tensor, optional): Zero-point tensor, will be modified in-place if required.

    Returns:
        torch.Tensor: Spike sequence tensor with shape [T, *x.shape]
    """
    if not isinstance(lif_quantizer, SpikeCountBaseLIFNode):
        raise TypeError("lif_quantizer must be a subclass of SpikeCountBaseLIFNode.")

    if not torch.allclose(x, x.round()):
        raise ValueError("Input x must be integer-valued (or floats with no fraction).")

    if lif_quantizer.is_bidirectional and torch.all(x >= 0):
        if x_zero is None:
            raise ValueError("x_zero is required for bidirectional mode when x is non-negative.")
        half_up = torch.ceil(x.to(torch.float32) / 2.0).to(x.dtype)
        x = x - half_up
        x_zero.sub_(half_up.to(x_zero.dtype))

    elif not lif_quantizer.is_bidirectional and torch.any(x < 0):
        if x_zero is None:
            raise ValueError("x_zero is required for non-bidirectional mode with negative x.")
        x_min = x.min()
        x = x - x_min
        x_zero.sub_(x_min.to(x_zero.dtype))

    x = x.to(torch.float32)
    spike = lif_quantizer(x)

    if lif_quantizer.is_two_complement:
        T = spike.shape[0]
        msb_mask = spike[0] >= 0.5 
        x_zero[msb_mask] += 2 ** T

    return spike

def spike_dequant(
    spike: torch.Tensor, 
    lif_quantizer: SpikeCountBaseLIFNode,
    x_zero: torch.Tensor = None
) -> torch.Tensor:
    """
    Decode a spike sequence back to integer values based on encoding type.

    Args:
        spike (torch.Tensor): Spike sequence of shape [T, *x.shape].
        lif_quantizer (SpikeCountBaseLIFNode): The encoder instance that generated the spike sequence.

    Returns:
        torch.Tensor: Reconstructed integer tensor of shape [*x.shape].
    """
    if not isinstance(lif_quantizer, SpikeCountBaseLIFNode):
        raise TypeError("lif_quantizer must be a subclass of SpikeCountBaseLIFNode.")

    if lif_quantizer.is_bitwise_coding:
        T = spike.shape[0]
        pow2 = 2 ** torch.arange(T - 1, -1, -1, device=spike.device, dtype=spike.dtype)
        weights = pow2.view(T, *([1] * (spike.dim() - 1)))
        decoded = (spike * weights).sum(dim=0)

        # if lif_quantizer.is_two_complement:
        #     msb_mask = spike[0] >= 0.5 
        #     x_zero[msb_mask] += 2 ** T
    else:
        # Binary and ternary modes simply sum all spikes (±1)
        decoded = spike.sum(dim=0)

    return decoded

def spike_fake_quant(
    x: torch.Tensor, 
    lif_quantizer: SpikeCountBaseLIFNode, 
    x_zero: torch.Tensor = None
) -> torch.Tensor:
    """
    Simulate spike-based quantization and dequantization (fake quantization),
    with optional zero-point shifting.

    Args:
        x (torch.Tensor): Input integer tensor.
        lif_quantizer (SpikeCountBaseLIFNode): Spike encoder (Binary / Ternary / Bitwise).
        x_zero (torch.Tensor, optional): Zero-point tensor, will be modified in-place if needed.

    Returns:
        torch.Tensor: Reconstructed tensor after spike quant + dequant.
    """
    spike_seq = spike_quant(x, lif_quantizer, x_zero)
    x_reconstructed = spike_dequant(spike_seq, lif_quantizer, x_zero)
    return x_reconstructed

def quant(
    x: torch.Tensor, 
    quantizer, 
    lif_quantizer: SpikeCountBaseLIFNode, 
    x_zero: torch.Tensor = None
) -> torch.Tensor:
    """
    Convert float tensor to spike sequence via integer quantizer + LIF encoder.

    Args:
        x (torch.Tensor): Float input tensor.
        quantizer (Callable): Float-to-int quantization function.
        lif_quantizer (SpikeCountBaseLIFNode): Spike encoder.
        x_zero (torch.Tensor, optional): Optional zero-point tensor (modified in-place if needed).

    Returns:
        torch.Tensor: Spike sequence with shape [T, *x.shape]
    """
    if quantizer is None:
        raise ValueError("A quantizer must be provided to map float → int.")
    
    spike_count = quantizer(x)
    return spike_quant(spike_count, lif_quantizer, x_zero)

def dequant(
    spike: torch.Tensor, 
    lif_quantizer: SpikeCountBaseLIFNode, 
    dequantizer
) -> torch.Tensor:
    """
    Full dequantization: decode spike sequence to pulse count, then map to float using dequantizer.

    Args:
        spike (torch.Tensor): Spike sequence of shape [T, *x.shape].
        lif_quantizer (SpikeCountBaseLIFNode): Spike encoder instance used during quantization.
        dequantizer (Callable): Function to map pulse count (int) → float.

    Returns:
        torch.Tensor: Final float-valued tensor.
    """
    if dequantizer is None:
        raise ValueError("A dequantizer must be provided for spike → count → float decoding.")

    spike_count = dequantizer(spike, lif_quantizer)  
    float_value = dequantizer(spike_count)           
    return float_value

def fake_quant(
    x: torch.Tensor,
    quantizer,
    lif_quantizer: SpikeCountBaseLIFNode,
    dequantizer,
    x_zero: torch.Tensor = None
) -> torch.Tensor:
    """
    Simulate spike-based quantization-dequantization (fake quantization) on float input.

    Args:
        x (torch.Tensor): Input float tensor.
        quantizer (Callable): Function mapping float → integer spike count.
        lif_quantizer (SpikeCountBaseLIFNode): Spike encoder (Binary / Ternary / Bitwise).
        dequantizer (Callable): Function mapping spike count → float.
        x_zero (torch.Tensor, optional): Zero-point tensor, modified in-place if needed.

    Returns:
        torch.Tensor: Float tensor after fake spike quantization + dequantization.
    """
    spike_seq = quant(x, quantizer, lif_quantizer, x_zero)
    x_reconstructed = dequant(spike_seq, lif_quantizer, dequantizer)
    return x_reconstructed

def spike_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    x_zero: torch.Tensor = None,
    lif_quantizer: SpikeCountBaseLIFNode = None,
    w_zero: torch.Tensor = None,
    weight_bitwise: bool = False
) -> torch.Tensor:
    """
    Perform matrix multiplication for spike-based inputs with optional spike sequence
    conversion (LIF nodes) and optional bitwise processing.

    Args:
        x (torch.Tensor): Input spike count tensor (must be integer-valued).
        w (torch.Tensor): Weight matrix (can be float or integer, depending on bitwise mode).
        x_zero (torch.Tensor, optional): Zero-point tensor for x, used in spike decomposition.
        lif_quantizer (SpikeCountBaseLIFNode, optional): Converts spike counts to spike sequences.
        w_zero (torch.Tensor, optional): Zero-point tensor for weights, used in bitwise quantization.
        weight_bitwise (bool): Whether to treat weights as bitwise-encoded values.

    Returns:
        torch.Tensor: Output after spike processing and matrix multiplication.
    """
    if not torch.allclose(x, x.round()):
        raise ValueError("Input x must be integer-valued.")

    # Use spike_quant for encoding + optional x_zero adjustment
    if lif_quantizer is not None:
        x = spike_quant(x, lif_quantizer, x_zero)
        

    # Bitwise weight quantization
    if weight_bitwise:
        if not torch.allclose(w, w.round()):
            raise ValueError("Weights must be integer-valued for bitwise mode.")
        w = weight_to_bitwise(w, w_zero)
        y = torch.stack([x @ t for t in w], dim=0)
    else:
        y = x @ w

    # Bitwise decoding
    x_is_bitwise = bool(getattr(lif_quantizer, "is_bitwise_coding", False))
    w_is_bitwise = bool(weight_bitwise)

    if x_is_bitwise and w_is_bitwise:
        Tx, Tw = x.shape[0], w.shape[0]
        pow2x = 2 ** torch.arange(Tx - 1, -1, -1, device=y.device, dtype=y.dtype)
        pow2w = 2 ** torch.arange(Tw - 1, -1, -1, device=y.device, dtype=y.dtype)
        wx = pow2x.view(1, Tx, *([1] * (y.dim() - 2)))
        ww = pow2w.view(Tw, 1, *([1] * (y.dim() - 2)))
        weights = wx * ww
        out = (y * weights).sum(dim=(0, 1))

        if lif_quantizer.is_bidirectional:
            msb_mask = y[0] >= 0.5  
            out[msb_mask] -= 2 ** Tx * w

    elif w_is_bitwise:
        Tw = w.shape[0]
        pow2w = 2 ** torch.arange(Tw - 1, -1, -1, device=y.device, dtype=y.dtype)
        weights = pow2w.view(Tw, *([1] * (y.dim() - 1)))
        out = (y * weights).sum(dim=0 if x_is_bitwise else (0, 1))

    elif x_is_bitwise:
        Tx = x.shape[0]
        pow2x = 2 ** torch.arange(Tx - 1, -1, -1, device=y.device, dtype=y.dtype)
        weights = pow2x.view(Tx, *([1] * (y.dim() - 1)))
        out = (y * weights).sum(dim=0)

    else:
        out = y.sum(dim=0) if lif_quantizer is not None else y

    return out

def weight_to_bitwise(
    w: torch.Tensor, 
    w_zero: torch.Tensor
):
    """
    Convert integer weight tensor into its bitwise representation.

    - Ensures `w` is integer-valued.
    - If `w` contains negative values, shifts it into a non-negative range (adjusting `w_zero` accordingly).
    - Computes the minimum number of bits T required to represent the largest value in `w`.
    - Returns a binary sequence tensor of shape [T, *w.shape], where each slice along dim 0 is one bit plane.

    Args:
        w (torch.Tensor): Integer weight tensor.
        w_zero (torch.Tensor): Zero-point tensor (must match `w` shape).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            bit_seq: Tensor of shape [T, *w.shape], containing 0/1 bit planes.
            w_zero: Adjusted zero-point tensor.
    """
    if not torch.allclose(w, w.round()):
        raise ValueError("w must be integer-valued.")
    if not isinstance(w_zero, torch.Tensor):
        raise TypeError(f"w_zero must be a torch.Tensor, but got {type(w_zero).__name__}")

    if torch.any(w < 0):
        w_min = w.min()
        w = w - w_min
        w_zero.sub_(w_min.to(w_zero.dtype))

    w_max = int(w.max().item())
    T = max(1, math.ceil(math.log2(w_max + 1)))

    masks = 2 ** torch.arange(T - 1, -1, -1, device=w.device)
    bit_seq = ((w.unsqueeze(0).long() & masks.view(-1, *[1]*w.dim())) != 0).float()

    return bit_seq

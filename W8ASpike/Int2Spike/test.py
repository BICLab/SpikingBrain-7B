import torch
from neuron import(
    SpikeCountBaseLIFNode,
    SpikeCountBinaryLIFNode,
    SpikeCountTernaryLIFNode,
    SpikeCountBitwiseNode,
    spike_fake_quant,
    spike_matmul,
    weight_to_bitwise,
)

def test_spike_count_binary_lif_node(
    x: torch.Tensor = None, 
    low: int = 0, 
    high: int = 15, 
    size: tuple = (12, 1024, 2048)
) -> bool:
    """
    Test function for SpikeCountBinaryLIFNode.

    Args:
        x (torch.Tensor, optional): Input tensor with non-negative integer values.
                                    If None, a random tensor will be generated.
        high (int): Upper bound for random generation (exclusive).
        size (tuple): Shape of randomly generated tensor.

    Returns:
        bool: True if the spike sequence sums match the original input; False otherwise.
    """
    if x is None:
        x = torch.randint(low=low, high=high+1, size=size)
    else:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a torch.Tensor")

        if not torch.all(x >= 0):
            raise ValueError("Input x must be non-negative")

        if not torch.allclose(x, x.round()):
            raise ValueError("Input x must contain integer values only")
   
    x_zero = torch.zeros_like(x, dtype=torch.float32)
    
    lif = SpikeCountBinaryLIFNode()
    spike_sum = spike_fake_quant(x, lif, x_zero)

    match = torch.allclose((x + x_zero).to(dtype=spike_sum.dtype), spike_sum, rtol=1e-3, atol=1e-3)
    
    if match:
        print(f"\u2714 Numerical match: {lif.__class__.__name__} spike sum matches input spike counts.")
    else:
        print(f"\u2718 Mismatch: {lif.__class__.__name__} spike sum does not match input spike counts.")

    return match

def test_spike_count_ternary_lif_node(
    x: torch.Tensor = None, 
    low: int = 0, 
    high: int = 15, 
    size: tuple = (12, 1024, 2048)
) -> bool:
    """
    Test function for SpikeCountTernaryLIFNode.

    Args:
        x (torch.Tensor, optional): Input tensor with integer values.
                                    If None, a random tensor will be generated.
        low (int): Lower bound for random input generation (inclusive).
        high (int): Upper bound for random input generation (exclusive).
        size (tuple): Shape of the input tensor.

    Returns:
        bool: True if spike sequence sums match the original input.
    """
    if x is None:
        x = torch.randint(low=low, high=high+1, size=size)
    else:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a torch.Tensor")

        if not torch.allclose(x, x.round()):
            raise ValueError("Input x must contain integer values only")

    x_zero = torch.zeros_like(x, dtype=torch.float32)
    
    lif = SpikeCountTernaryLIFNode()
    spike_sum = spike_fake_quant(x, lif, x_zero)

    match = torch.allclose((x + x_zero).to(dtype=spike_sum.dtype), spike_sum, rtol=1e-3, atol=1e-3)

    if match:
        print(f"\u2714 Numerical match: {lif.__class__.__name__} spike sum matches input spike counts.")
    else:
        print(f"\u2718 Mismatch: {lif.__class__.__name__} spike sum does not match input spike counts.")

    return match

def test_spike_count_bitwise_node(
    x: torch.Tensor = None,
    low: int = 0,
    high: int = 15,
    size: tuple = (12, 1024, 2048),
    is_bidirectional: bool = False,
    is_two_complement: bool = False
) -> bool:
    """
    Test function for SpikeCountBitwiseNode using spike_fake_quant().

    Args:
        x (torch.Tensor, optional): Input tensor with integer values.
                                    If None, a random tensor will be generated.
        low (int): Lower bound (inclusive) for random input generation.
        high (int): Upper bound (inclusive) for random input generation.
        size (tuple): Shape of the input tensor.
        is_bidirectional (bool): Whether to use signed (two's complement) encoding.

    Returns:
        bool: True if the reconstructed value equals the input.
    """
    if x is None:
        x = torch.randint(low=low, high=high + 1, size=size)
    else:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a torch.Tensor")

        if not torch.allclose(x, x.round()):
            raise ValueError("Input x must contain integer values only")

        if not is_bidirectional and not torch.all(x >= 0):
            raise ValueError("Input x must be non-negative for unsigned mode.")
        
    # Initialize x_zero (zero point for the input)
    x_zero = torch.zeros_like(x, dtype=torch.float32)
    
    lif = SpikeCountBitwiseNode(is_bidirectional = is_bidirectional, is_two_complement = is_two_complement)

    spike_sum = spike_fake_quant(x, lif, x_zero)

    match = torch.allclose((x + x_zero).to(dtype=spike_sum.dtype), spike_sum, rtol=1e-3, atol=1e-3)

    if match:
        print(f"✓ Match: {lif.__class__.__name__} decoded output matches input.")
    else:
        print(f"✘ Mismatch: {lif.__class__.__name__} decoding failed.")

    return match


def test_spike_matmul_equivalence(
    x: torch.Tensor = None, 
    x_low: int = 0, 
    x_high: int = 15, 
    x_size: tuple = (12, 2048, 2048), 
    w: torch.Tensor = None, 
    w_size: tuple = (2048, 2048), 
    lif_quantizer: SpikeCountBaseLIFNode = None, 
    weight_bitwise: bool = False
) -> bool:
    """
    General test for verifying spike_matmul matches dense matmul output
    when using a given SpikeCount LIF quantizer.

    Args:
        x (torch.Tensor, optional): Integer input tensor. Randomly generated if None.
        w (torch.Tensor, optional): Weight tensor. Randomly generated if None.
        lif_quantizer (SpikeCountBaseLIFNode): Quantizer to convert spike count to spike sequences.
        high (int): Max value (inclusive) for input generation.
        size (tuple): (B, K, N), where x.shape=(B,K), w.shape=(K,N)
        rtol (float): Relative tolerance.
        atol (float): Absolute tolerance.

    Returns:    
        bool: True if spike_matmul output is close to x @ w.
    """
    if x is None:
        x = torch.randint(low=x_low, high=x_high + 1, size=x_size).float()
        
        x_zero = torch.zeros_like(x)
    else:
        if not torch.allclose(x, x.round()):
            raise ValueError("x must be integer-valued.")
        x = x.to(torch.float32)

    if w is None:
        w = torch.randint(low=-8, high=7, size=w_size).float()
        w_zero = torch.zeros_like(w)

    if lif_quantizer is None:
        raise ValueError("lif_quantizer must be provided (e.g., SpikeCountBinaryLIFNode()).")

    y_spike = spike_matmul(x, w, x_zero=x_zero, lif_quantizer=lif_quantizer, w_zero=w_zero, weight_bitwise=weight_bitwise)
    y_ref = (x + x_zero) @ (w + w_zero)

    match = torch.allclose(y_ref, y_spike, rtol=1e-3, atol=1e-3)

    if match:
        if isinstance(lif_quantizer, SpikeCountBinaryLIFNode):
            print(f"\u2714 Numerical match: Binary spike sequence (0/1) matmul with {lif_quantizer.__class__.__name__} matches spike count matmul.")
        elif isinstance(lif_quantizer, SpikeCountTernaryLIFNode):
            print(f"\u2714 Numerical match: Ternary spike sequence (-1/0/1) matmul with {lif_quantizer.__class__.__name__} matches spike count matmul.")
        elif isinstance(lif_quantizer, SpikeCountBitwiseNode):
            print(f"\u2714 Numerical match: Bitwise spike sequence matmul with {lif_quantizer.__class__.__name__} matches spike count matmul.")
    else:
        if isinstance(lif_quantizer, SpikeCountBinaryLIFNode):
            print(f"\u2718 Mismatch: Binary spike sequence (0/1) matmul with {lif_quantizer.__class__.__name__} does not match spike count matmul.")
        elif isinstance(lif_quantizer, SpikeCountTernaryLIFNode):
            print(f"\u2718 Mismatch: Ternary spike sequence (-1/0/1) matmul with {lif_quantizer.__class__.__name__} does not match spike count matmul.")
        elif isinstance(lif_quantizer, SpikeCountBitwiseNode):
            print(f"\u2718 Mismatch: Bitwise spike sequence matmul with {lif_quantizer.__class__.__name__} does not match spike count matmul.")

    return match

def test_weight_to_bitwise(
    w: torch.Tensor = None, 
    low: int = -8, 
    high: int = 7, 
    size: tuple = (1024, 1024)) -> bool:
    w = torch.randint(low=low, high=high, size=size).float()
    w_zero = torch.zeros_like(w)
    bit_seq= weight_to_bitwise(w, w_zero)

    T = bit_seq.shape[0]
    powers = 2 ** torch.arange(T - 1, -1, -1).view(T, *[1] * (bit_seq.dim() - 1))
    recovered = (bit_seq * powers).sum(dim=0) 

    w = w + w_zero

    match = torch.allclose(recovered, w, rtol=1e-3, atol=1e-3)

    if match:
        print(f"\u2714 Passed: Recovered weights match original weights. shape={w.shape}")
    else:
        print(f"\u2718 Failed: Recovered weights do not match original. shape={w.shape}")

  
# --- Binary LIF Node: Numerical match test for spike count decoding ---
test_spike_count_binary_lif_node(high=15, size=(12, 2048, 2048))    # 4-bit unsigned
test_spike_count_binary_lif_node(high=31, size=(12, 2048, 2048))    # 5-bit unsigned
test_spike_count_binary_lif_node(high=63, size=(12, 2048, 2048))    # 6-bit unsigned
test_spike_count_binary_lif_node(high=255, size=(12, 2048, 2048))   # 8-bit unsigned

test_spike_count_binary_lif_node(low=-8, high=7, size=(12, 2048, 2048))       # 4-bit signed, zero-point adjustment
test_spike_count_binary_lif_node(low=-16, high=15, size=(12, 2048, 2048))     # 5-bit signed
test_spike_count_binary_lif_node(low=-32, high=31, size=(12, 2048, 2048))     # 6-bit signed
test_spike_count_binary_lif_node(low=-128, high=127, size=(12, 2048, 2048))   # 8-bit signed

# --- Binary LIF Node: Numerical match test for matmul ---
test_spike_matmul_equivalence(x_high=15, lif_quantizer=SpikeCountBinaryLIFNode())     # 4-bit unsigned
test_spike_matmul_equivalence(x_high=31, lif_quantizer=SpikeCountBinaryLIFNode())     # 5-bit unsigned
test_spike_matmul_equivalence(x_high=63, lif_quantizer=SpikeCountBinaryLIFNode())     # 6-bit unsigned
test_spike_matmul_equivalence(x_high=255, lif_quantizer=SpikeCountBinaryLIFNode())    # 8-bit unsigned

test_spike_matmul_equivalence(x_low=-8, x_high=7, lif_quantizer=SpikeCountBinaryLIFNode())     # 4-bit signed, zero-point adjustment
test_spike_matmul_equivalence(x_low=-16, x_high=15, lif_quantizer=SpikeCountBinaryLIFNode())   # 5-bit signed
test_spike_matmul_equivalence(x_low=-32, x_high=31, lif_quantizer=SpikeCountBinaryLIFNode())   # 6-bit signed
test_spike_matmul_equivalence(x_low=-128, x_high=127, lif_quantizer=SpikeCountBinaryLIFNode()) # 8-bit signed

# --- Ternary LIF Node: Numerical match test for spike count decoding ---
test_spike_count_ternary_lif_node(high=15, size=(12, 2048, 2048))    # 4-bit unsigned, zero-point adjustment
test_spike_count_ternary_lif_node(high=31, size=(12, 2048, 2048))    # 5-bit unsigned
test_spike_count_ternary_lif_node(high=63, size=(12, 2048, 2048))    # 6-bit unsigned
test_spike_count_ternary_lif_node(high=255, size=(12, 2048, 2048))   # 8-bit unsigned

test_spike_count_ternary_lif_node(low=-8, high=7, size=(12, 2048, 2048))      # 4-bit signed 
test_spike_count_ternary_lif_node(low=-16, high=15, size=(12, 2048, 2048))    # 5-bit signed 
test_spike_count_ternary_lif_node(low=-32, high=31, size=(12, 2048, 2048))    # 6-bit signed 
test_spike_count_ternary_lif_node(low=-128, high=127, size=(12, 2048, 2048))  # 8-bit signed 

# --- Ternary LIF Node: Numerical match test for matmul ---
test_spike_matmul_equivalence(x_high=15, lif_quantizer=SpikeCountTernaryLIFNode())     # 4-bit unsigned, zero-point adjustment
test_spike_matmul_equivalence(x_high=31, lif_quantizer=SpikeCountTernaryLIFNode())     # 5-bit unsigned
test_spike_matmul_equivalence(x_high=63, lif_quantizer=SpikeCountTernaryLIFNode())     # 6-bit unsigned
test_spike_matmul_equivalence(x_high=255, lif_quantizer=SpikeCountTernaryLIFNode())    # 8-bit unsigned

test_spike_matmul_equivalence(x_low=-8, x_high=7, lif_quantizer=SpikeCountTernaryLIFNode())     # 4-bit signed
test_spike_matmul_equivalence(x_low=-16, x_high=15, lif_quantizer=SpikeCountTernaryLIFNode())   # 5-bit signed
test_spike_matmul_equivalence(x_low=-32, x_high=31, lif_quantizer=SpikeCountTernaryLIFNode())   # 6-bit signed
test_spike_matmul_equivalence(x_low=-128, x_high=127, lif_quantizer=SpikeCountTernaryLIFNode()) # 8-bit signed

# --- Bitwise LIF Node: Numerical match test for spike count decoding ---
test_spike_count_bitwise_node(high=15, size=(12, 2048, 2048))    # 4-bit unsigned 
test_spike_count_bitwise_node(high=31, size=(12, 2048, 2048))    # 5-bit unsigned 
test_spike_count_bitwise_node(high=63, size=(12, 2048, 2048))    # 6-bit unsigned 
test_spike_count_bitwise_node(high=255, size=(12, 2048, 2048))   # 8-bit unsigned 

test_spike_count_bitwise_node(low=-8, high=7, size=(12, 2048, 2048))      # 4-bit signed, zero-point adjustment
test_spike_count_bitwise_node(low=-16, high=15, size=(12, 2048, 2048))    # 5-bit signed 
test_spike_count_bitwise_node(low=-32, high=31, size=(12, 2048, 2048))    # 6-bit signed 
test_spike_count_bitwise_node(low=-128, high=127, size=(12, 2048, 2048))  # 8-bit signed 

test_spike_count_bitwise_node(high=15, size=(12, 2048, 2048), is_bidirectional=True)  # 4-bit unsigned, zero-point adjustment
test_spike_count_bitwise_node(high=31, size=(12, 2048, 2048), is_bidirectional=True)  # 5-bit unsigned 
test_spike_count_bitwise_node(high=63, size=(12, 2048, 2048), is_bidirectional=True)  # 6-bit unsigned 
test_spike_count_bitwise_node(high=255, size=(12, 2048, 2048), is_bidirectional=True) # 8-bit unsigned 

test_spike_count_bitwise_node(low=-8, high=0, size=(12, 2048, 2048), is_bidirectional=True)      # 4-bit signed
test_spike_count_bitwise_node(low=-16, high=15, size=(12, 2048, 2048), is_bidirectional=True)    # 5-bit signed 
test_spike_count_bitwise_node(low=-32, high=31, size=(12, 2048, 2048), is_bidirectional=True)    # 6-bit signed 
test_spike_count_bitwise_node(low=-128, high=127, size=(12, 2048, 2048), is_bidirectional=True)  # 8-bit signed 

test_spike_count_bitwise_node(high=15, size=(12, 2048, 2048), is_bidirectional=True, is_two_complement=True) # 4-bit unsigned, zero-point adjustment
test_spike_count_bitwise_node(high=31, size=(12, 2048, 2048), is_bidirectional=True, is_two_complement=True) # 5-bit unsigned 
test_spike_count_bitwise_node(high=63, size=(12, 2048, 2048), is_bidirectional=True, is_two_complement=True) # 6-bit unsigned 
test_spike_count_bitwise_node(high=255, size=(12, 2048, 2048), is_bidirectional=True, is_two_complement=True)# 8-bit unsigned 

test_spike_count_bitwise_node(low=-8, high=0, size=(12, 2048, 2048), is_bidirectional=True, is_two_complement=True)    # 4-bit signed, zero-point adjustment
test_spike_count_bitwise_node(low=-16, high=15, size=(12, 2048, 2048), is_bidirectional=True, is_two_complement=True)  # 5-bit signed
test_spike_count_bitwise_node(low=-32, high=31, size=(12, 2048, 2048), is_bidirectional=True, is_two_complement=True)  # 6-bit signed
test_spike_count_bitwise_node(low=-128, high=127, size=(12, 2048, 2048), is_bidirectional=True, is_two_complement=True)# 8-bit signed

# --- Bitwise LIF Node: Numerical match test for matmul ---
test_spike_matmul_equivalence(x_high=15, lif_quantizer=SpikeCountBitwiseNode())     # 4-bit unsigned
test_spike_matmul_equivalence(x_high=31, lif_quantizer=SpikeCountBitwiseNode())     # 5-bit unsigned
test_spike_matmul_equivalence(x_high=63, lif_quantizer=SpikeCountBitwiseNode())     # 6-bit unsigned
test_spike_matmul_equivalence(x_high=255, lif_quantizer=SpikeCountBitwiseNode())    # 8-bit unsigned

test_spike_matmul_equivalence(x_low=-8, x_high=7, lif_quantizer=SpikeCountBitwiseNode())     # 4-bit signed, zero-point adjustment
test_spike_matmul_equivalence(x_low=-16, x_high=15, lif_quantizer=SpikeCountBitwiseNode())   # 5-bit signed
test_spike_matmul_equivalence(x_low=-32, x_high=31, lif_quantizer=SpikeCountBitwiseNode())   # 6-bit signed
test_spike_matmul_equivalence(x_low=-128, x_high=127, lif_quantizer=SpikeCountBitwiseNode()) # 8-bit signed

test_spike_matmul_equivalence(x_high=15, lif_quantizer=SpikeCountBitwiseNode(is_bidirectional=True))  # 4-bit unsigned, zero-point adjustment
test_spike_matmul_equivalence(x_high=31, lif_quantizer=SpikeCountBitwiseNode(is_bidirectional=True))  # 5-bit unsigned
test_spike_matmul_equivalence(x_high=63, lif_quantizer=SpikeCountBitwiseNode(is_bidirectional=True))  # 6-bit unsigned
test_spike_matmul_equivalence(x_high=255, lif_quantizer=SpikeCountBitwiseNode(is_bidirectional=True)) # 8-bit unsigned

test_spike_matmul_equivalence(x_low=-8, x_high=7, lif_quantizer=SpikeCountBitwiseNode(is_bidirectional=True))     # 4-bit signed
test_spike_matmul_equivalence(x_low=-16, x_high=15, lif_quantizer=SpikeCountBitwiseNode(is_bidirectional=True))   # 5-bit signed
test_spike_matmul_equivalence(x_low=-32, x_high=31, lif_quantizer=SpikeCountBitwiseNode(is_bidirectional=True))   # 6-bit signed
test_spike_matmul_equivalence(x_low=-128, x_high=127, lif_quantizer=SpikeCountBitwiseNode(is_bidirectional=True)) # 8-bit signed

test_spike_matmul_equivalence(x_high=15, lif_quantizer=SpikeCountBitwiseNode(is_bidirectional=True, is_two_complement=True))  # 4-bit unsigned, zero-point adjustment 
test_spike_matmul_equivalence(x_high=31, lif_quantizer=SpikeCountBitwiseNode(is_bidirectional=True, is_two_complement=True))  # 5-bit unsigned
test_spike_matmul_equivalence(x_high=63, lif_quantizer=SpikeCountBitwiseNode(is_bidirectional=True, is_two_complement=True))  # 6-bit unsigned
test_spike_matmul_equivalence(x_high=255, lif_quantizer=SpikeCountBitwiseNode(is_bidirectional=True, is_two_complement=True)) # 8-bit unsigned

test_spike_matmul_equivalence(x_low=-8, x_high=7, lif_quantizer=SpikeCountBitwiseNode(is_bidirectional=True, is_two_complement=True))     # 4-bit signed, zero-point adjustment
test_spike_matmul_equivalence(x_low=-16, x_high=15, lif_quantizer=SpikeCountBitwiseNode(is_bidirectional=True, is_two_complement=True))   # 5-bit signed
test_spike_matmul_equivalence(x_low=-32, x_high=31, lif_quantizer=SpikeCountBitwiseNode(is_bidirectional=True, is_two_complement=True))   # 6-bit signed
test_spike_matmul_equivalence(x_low=-128, x_high=127, lif_quantizer=SpikeCountBitwiseNode(is_bidirectional=True, is_two_complement=True)) # 8-bit signed

# --- Bitwise LIF Node and Bitwise Weight: Numerical match test for matmul ---
test_spike_matmul_equivalence(x_high=15, lif_quantizer=SpikeCountBitwiseNode(), weight_bitwise=True)     # 4-bit unsigned
test_spike_matmul_equivalence(x_high=31, lif_quantizer=SpikeCountBitwiseNode(), weight_bitwise=True)     # 5-bit unsigned
test_spike_matmul_equivalence(x_high=63, lif_quantizer=SpikeCountBitwiseNode(), weight_bitwise=True)     # 6-bit unsigned
test_spike_matmul_equivalence(x_high=255, lif_quantizer=SpikeCountBitwiseNode(), weight_bitwise=True)    # 8-bit unsigned

test_spike_matmul_equivalence(x_low=-8, x_high=7, lif_quantizer=SpikeCountBitwiseNode(), weight_bitwise=True)     # 4-bit signed, zero-point adjustment
test_spike_matmul_equivalence(x_low=-16, x_high=15, lif_quantizer=SpikeCountBitwiseNode(), weight_bitwise=True)   # 5-bit signed
test_spike_matmul_equivalence(x_low=-32, x_high=31, lif_quantizer=SpikeCountBitwiseNode(), weight_bitwise=True)   # 6-bit signed
test_spike_matmul_equivalence(x_low=-128, x_high=127, lif_quantizer=SpikeCountBitwiseNode(), weight_bitwise=True) # 8-bit signed

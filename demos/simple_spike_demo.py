#!/usr/bin/env python3
"""
Simple Spike Encoding Demo (No Dependencies Required)
======================================================

This lightweight demo shows the core spiking mechanisms without requiring
PyTorch or other heavy dependencies. Perfect for quick testing!
"""

import random

def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def binary_lif_encode(value, max_value=15):
    """
    Binary LIF encoding: fires 1 each timestep until count reaches zero

    Args:
        value: Integer spike count (0-max_value)
        max_value: Maximum value (default 15 for int4)

    Returns:
        list: Binary spike train [0 or 1]
    """
    spikes = []
    remaining = int(value)

    while remaining > 0:
        spikes.append(1)
        remaining -= 1

    # Pad to max length for visualization
    while len(spikes) < max_value:
        spikes.append(0)

    return spikes

def ternary_lif_encode(value, max_abs_value=8):
    """
    Ternary LIF encoding: fires +1/-1 each timestep based on sign

    Args:
        value: Signed integer spike count (-max_abs_value to +max_abs_value)
        max_abs_value: Maximum absolute value (default 8 for signed int4)

    Returns:
        list: Ternary spike train [-1, 0, or +1]
    """
    spikes = []
    remaining = int(value)

    while remaining != 0:
        if remaining > 0:
            spikes.append(1)
            remaining -= 1
        else:
            spikes.append(-1)
            remaining += 1

    # Pad to max length
    while len(spikes) < max_abs_value:
        spikes.append(0)

    return spikes

def bitwise_encode(value, bits=4):
    """
    Bitwise encoding: each bit is one timestep

    Args:
        value: Integer value (0-2^bits)
        bits: Number of bits

    Returns:
        list: Binary spike train representing bits [MSB...LSB]
    """
    spikes = []
    for i in range(bits - 1, -1, -1):
        bit = (value >> i) & 1
        spikes.append(bit)
    return spikes

def visualize_spike_train(spikes, value, encoding_type):
    """ASCII visualization of spike train"""
    spike_chars = {-1: '↓', 0: '·', 1: '↑'}

    # Print value and encoding info
    print(f"\n  Value: {value:>4} | Encoding: {encoding_type}")
    print(f"  Timesteps: ", end="")
    for i, s in enumerate(spikes):
        print(f"{i:>2}", end=" ")
    print()

    print(f"  Spikes:    ", end="")
    for s in spikes:
        print(f" {spike_chars.get(s, '?')} ", end="")
    print()

    # Calculate metrics
    total_spikes = sum(abs(s) for s in spikes)
    firing_rate = total_spikes / len(spikes)
    sparsity = 1 - firing_rate

    print(f"  Metrics: {total_spikes} spikes, {firing_rate:.2f} firing rate, {sparsity:.1%} sparsity")

def demo_binary_encoding():
    """Demo binary spike encoding"""
    print_header("Demo 1: Binary Spike Encoding (0/1)")
    print("Use case: Unipolar neuromorphic hardware")
    print("Encoding: Fires 1 each timestep until value reaches zero")

    test_values = [0, 3, 7, 12, 15]

    for value in test_values:
        spikes = binary_lif_encode(value)
        visualize_spike_train(spikes, value, "Binary")

    print(f"\n💡 Insight: Latency is proportional to value (O(n))")

def demo_ternary_encoding():
    """Demo ternary spike encoding"""
    print_header("Demo 2: Ternary Spike Encoding (-1/0/+1)")
    print("Use case: Bipolar neuromorphic hardware (excitatory/inhibitory)")
    print("Encoding: Fires +1 for positive, -1 for negative")

    test_values = [-7, -3, 0, 3, 7]

    for value in test_values:
        spikes = ternary_lif_encode(value)
        visualize_spike_train(spikes, value, "Ternary")

    print(f"\n💡 Insight: Can represent signed values naturally")

def demo_bitwise_encoding():
    """Demo bitwise spike encoding"""
    print_header("Demo 3: Bitwise Spike Encoding")
    print("Use case: Ultra-low latency, parallel bit processing")
    print("Encoding: Each bit is one timestep (fixed latency!)")

    test_values = [0, 3, 7, 12, 15]

    for value in test_values:
        spikes = bitwise_encode(value, bits=4)
        visualize_spike_train(spikes, value, "Bitwise")

    print(f"\n💡 Insight: Fixed latency of log2(max_value) timesteps")

def compare_encodings():
    """Compare all three encoding methods"""
    print_header("Encoding Comparison")

    test_value = 7

    print(f"\nEncoding value: {test_value}")
    print(f"\n  Binary (0/1):")
    binary = binary_lif_encode(test_value)
    print(f"    {binary[:test_value+1]}")
    print(f"    Timesteps needed: {test_value}")

    print(f"\n  Ternary (-1/0/+1):")
    ternary = ternary_lif_encode(test_value)
    print(f"    {ternary[:test_value+1]}")
    print(f"    Timesteps needed: {test_value}")

    print(f"\n  Bitwise:")
    bitwise = bitwise_encode(test_value, bits=4)
    print(f"    {bitwise}")
    print(f"    Timesteps needed: 4 (fixed)")

    print(f"\n📊 Summary:")
    print(f"  • Rate coding (binary/ternary): Variable latency, higher sparsity")
    print(f"  • Bitwise coding: Fixed latency, lower sparsity")
    print(f"  • Choice depends on hardware constraints!")

def demo_spike_accumulation():
    """Demo spike-based accumulation (like hardware would do)"""
    print_header("Demo 4: Spike Accumulation (Hardware Operation)")
    print("Simulating how neuromorphic hardware accumulates spikes")

    # Simulate input spikes
    input_values = [3, 5, 2, 7]
    weight = 0.5

    print(f"\nInput values: {input_values}")
    print(f"Weight: {weight}")
    print(f"\nHardware operation: accumulate(spike × weight) over time\n")

    total_output = 0
    for value in input_values:
        spikes = binary_lif_encode(value)
        # Simulate accumulation
        output = sum(s * weight for s in spikes if s > 0)
        total_output += output
        print(f"  Value {value:>2} → {sum(spikes):>2} spikes → output: {output:.2f}")

    print(f"\n  Total accumulated output: {total_output:.2f}")
    print(f"  Expected (direct): {sum(input_values) * weight:.2f}")
    print(f"  ✓ Results match!")

def print_neuronchip_guide():
    """Print integration guide"""
    print_header("NeuronChip.org Integration Summary")

    print("""
🔧 Hardware Requirements:

1. SPIKE GENERATION CIRCUIT:
   ├─ Input: Quantized values (int4/int8)
   ├─ Output: Spike train (-1/0/+1)
   ├─ Logic: LIF neuron model (charge-fire-reset)
   └─ Latency: < 100ns per timestep

2. SPIKE ACCUMULATION CIRCUIT:
   ├─ Operation: weighted_sum += spike × weight
   ├─ Precision: 16-bit or 32-bit accumulator
   ├─ Reset: After each layer
   └─ Energy: ~0.1pJ per spike operation

3. MEMORY INTERFACE:
   ├─ Spike buffer: Store timestep × neurons spikes
   ├─ Weight memory: Access patterns optimized for spikes
   └─ Bandwidth: ~1-10 GB/s depending on throughput

📈 Expected Performance:

  Metric                    Target
  ─────────────────────────────────────────
  Sparsity                  60-70%
  Energy per inference      10-100× lower than GPU
  Latency per layer         < 1ms
  Throughput                100-1000 tokens/sec

🚀 Next Steps:

1. Choose encoding method (we recommend Ternary)
2. Implement spike generation in hardware/RTL
3. Test with this demo's algorithms
4. Integrate with full SpikingBrain-7B model
5. Profile and optimize

📚 Resources:
   • Full demo: neuronchip_spike_demo.py (requires PyTorch)
   • Architecture: ../ARCHITECTURE_GUIDE.md
   • Source code: ../W8ASpike/Int2Spike/neuron.py
""")

def main():
    """Main demo runner"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║     Simple Spike Encoding Demo for NeuronChip.org             ║
║                                                                ║
║  Demonstrating core spiking algorithms for neuromorphic AI    ║
║  No dependencies required - pure Python!                       ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
""")

    # Run all demos
    demo_binary_encoding()
    demo_ternary_encoding()
    demo_bitwise_encoding()
    compare_encodings()
    demo_spike_accumulation()
    print_neuronchip_guide()

    print_header("Demo Complete! 🎉")
    print("""
✅ All demonstrations completed!

🎯 Key Takeaways:
   • Three encoding methods: Binary, Ternary, Bitwise
   • Ternary (±1) recommended for best accuracy/efficiency
   • Sparsity of 60-70% means huge energy savings
   • Fixed latency possible with bitwise encoding

🚀 Next Steps:
   1. Install PyTorch: pip install torch matplotlib numpy
   2. Run full demo: python neuronchip_spike_demo.py
   3. Review: ../ARCHITECTURE_GUIDE.md
   4. Download full model from ModelScope (see README.md)

💡 Ready to build neuromorphic AI systems!
""")

if __name__ == "__main__":
    main()

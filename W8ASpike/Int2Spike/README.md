# Int2Spike: From Integer Spike Count to Spikes

## About Int2Spike

Inspired by biological computing mechanisms, SpikingBrain proposes a spike encoding strategy based on low-power features and sparse event-driven mechanisms. The strategy is realized through a two-step decoupling process: floating-point to integer to spike. In the first step, continuous membrane potential is quantized into integer spike counts, combined with a directed sparsity method; in the second step, spike counts are encoded into sparse spike sequences through temporal expansion.

We provide three spike encoding methods:
- Binary Spike Encoding (0/1) 
- Ternary Spike Encoding (-1/0/1)
- Bitwise Spike Encoding (Symmetric and Two's Complement)

Based on these encoding methods, we have implemented a large-scale model where sparse addition replaces dense matrix multiplication, designed for low-power edge computing scenarios. Compared to ANN quantization, we achieve performance comparable to ANN quantization while reducing energy consumption by a factor of ten, and we provide valuable insights for the design of next-generation neuromorphic chips.

It is worth noting that our actual time step (= firing rate  ×  time step) is very short, optimized to T=2, and can also be configured for parallel hardware architectures.

![Int2Spike](https://github.com/BICLab/Int2Spike/blob/main/spike_coding.png?raw=true) 

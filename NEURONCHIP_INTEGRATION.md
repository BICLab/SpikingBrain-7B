# SpikingBrain-7B Integration Guide for NeuronChip.org

## Executive Summary

This document provides a complete guide for integrating **SpikingBrain-7B** with neuromorphic hardware at **https://neuronchip.org**. SpikingBrain-7B is a 7-billion parameter large language model that incorporates spiking neural network (SNN) quantization, achieving **69% sparsity** at the micro-level and **100× speedup** for long-context inference.

---

## Quick Start Demo

We've created working demos that showcase the spiking mechanisms:

### 1. Simple Demo (No Dependencies)
```bash
cd demos
python3 simple_spike_demo.py
```

This demo shows:
- ✅ Binary spike encoding (0/1)
- ✅ Ternary spike encoding (-1/0/+1) **← RECOMMENDED**
- ✅ Bitwise spike encoding (fixed latency)
- ✅ Spike accumulation (hardware operation simulation)
- ✅ ASCII visualizations of spike trains

**Output:**
```
  Value:    7 | Encoding: Ternary
  Timesteps:  0  1  2  3  4  5  6  7
  Spikes:     ↑  ↑  ↑  ↑  ↑  ↑  ↑  ·
  Metrics: 7 spikes, 0.88 firing rate, 12.5% sparsity
```

### 2. Full Demo (Requires PyTorch)
```bash
# Install dependencies
pip install torch matplotlib numpy einops

# Run full demo with visualizations
python3 neuronchip_spike_demo.py
```

Generates:
- 📊 Spike raster plots
- 📈 Encoding comparison charts
- 📉 Sparsity analysis
- 🔬 Reconstruction accuracy metrics

---

## System Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    NeuronChip.org System                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Input Tokens    │
                    └────────┬─────────┘
                             │
                             ▼
            ┌─────────────────────────────────┐
            │  SpikingBrain-7B Transformer    │
            │  • 28 hybrid attention layers   │
            │  • GLA + Flash Attention        │
            │  • 3584 hidden dimensions       │
            └────────┬────────────────────────┘
                     │
                     ▼
            ┌─────────────────────────────────┐
            │  W8ASpike Quantization          │
            │  • Float32 → Int8               │
            │  • LIF neuron encoding          │
            │  • Spike train generation       │
            └────────┬────────────────────────┘
                     │
                     ▼
            ┌─────────────────────────────────┐
            │  NeuronChip Hardware Interface  │
            │  • Spike transmission           │
            │  • Accumulate-and-fire          │
            │  • Output reconstruction        │
            └─────────────────────────────────┘
```

### Spike Encoding Flow

```
Activation (Float32)
      │
      ▼
[ Quantization ]  →  Int8 value (-128 to 127)
      │
      ▼
[ LIF Encoding ]  →  Spike train [T, neurons]
      │                T = timesteps
      ▼
[ Transmission ]  →  Send to hardware
      │
      ▼
[ Accumulation ]  →  weighted_sum += spike × weight
      │
      ▼
[ Reconstruction ] →  Output activation
```

---

## Spike Encoding Methods

### 🥇 Ternary Encoding (RECOMMENDED)

**Best for:** Most neuromorphic hardware

**Advantages:**
- Bipolar spiking (-1/0/+1) matches biological neurons
- 60-70% sparsity achievable
- Lossless reconstruction for quantized values
- Natural signed value representation

**Example:**
```python
# Input: Quantized activation value = 7
# Output spike train: [+1, +1, +1, +1, +1, +1, +1, 0, 0, ...]
#
# For negative value = -5:
# Output: [-1, -1, -1, -1, -1, 0, 0, ...]
```

**Hardware Requirements:**
- Bipolar spike generation circuit
- Signed accumulator (16-bit or 32-bit)
- Positive/negative spike routing

### 🥈 Binary Encoding

**Best for:** Simple unipolar hardware

**Advantages:**
- Simplest hardware implementation
- Unipolar (0/1) only
- Good sparsity for small values

**Example:**
```python
# Input: value = 5
# Output: [1, 1, 1, 1, 1, 0, 0, ...]
```

**Hardware Requirements:**
- Unipolar spike generation
- Unsigned accumulator
- Simpler routing

### 🥉 Bitwise Encoding

**Best for:** Ultra-low latency applications

**Advantages:**
- Fixed latency: log₂(max_value) timesteps
- Parallel bit processing possible
- Deterministic timing

**Example:**
```python
# Input: value = 7 (int4)
# Binary: 0111
# Output: [0, 1, 1, 1]  (4 timesteps fixed)
```

**Hardware Requirements:**
- Bit-serial transmission
- Bit-parallel weight processing (optional)
- Binary weighted accumulation

---

## Hardware Integration Checklist

### Phase 1: Circuit Design

- [ ] **Spike Generation Circuit**
  - Input: Quantized values (int8 or int4)
  - Output: Spike trains (±1 or 0/1)
  - Logic: LIF neuron model (charge-fire-reset)
  - Target latency: < 100ns per timestep

- [ ] **Spike Accumulation Circuit**
  - Operation: `acc += spike × weight`
  - Accumulator width: 16-bit (minimum), 32-bit (recommended)
  - Reset mechanism: Per-layer reset
  - Energy target: ~0.1 pJ per spike operation

- [ ] **Memory Architecture**
  - Spike buffer: [T × batch × neurons]
  - Weight memory: Optimized access patterns
  - Bandwidth: 1-10 GB/s (depends on throughput goal)

### Phase 2: Software Interface

- [ ] **Driver Development**
  - Implement spike I/O protocol
  - Handle timestep synchronization
  - Buffer management for spike trains

- [ ] **Adapter Implementation**
  - Use provided `NeuronChipAdapter` class (see ARCHITECTURE_GUIDE.md)
  - Customize for your hardware interface (PCIe/custom)
  - Test with demo spike patterns

- [ ] **Integration Testing**
  - Verify encoding/decoding accuracy
  - Measure end-to-end latency
  - Profile power consumption
  - Compare with GPU baseline

### Phase 3: System Integration

- [ ] **Model Deployment**
  - Download SpikingBrain-7B weights (see below)
  - Load model with transformers library
  - Hook hardware adapter into inference pipeline

- [ ] **Performance Optimization**
  - Optimize spike buffer sizes
  - Tune batch sizes for throughput
  - Implement pipelining if possible

- [ ] **Validation**
  - Accuracy testing on benchmarks
  - Latency profiling per layer
  - Power measurement vs. GPU
  - Throughput stress testing

---

## Model Weights & Deployment

### Download Pre-trained Models

Choose the model variant for your use case:

```bash
# Install ModelScope SDK
pip install modelscope

# Download base model (7B parameters)
from modelscope import snapshot_download
model_dir = snapshot_download('Panyuqi/V1-7B-base')

# OR download chat model (fine-tuned)
model_dir = snapshot_download('Panyuqi/V1-7B-sft-s3-reasoning')

# OR download vision-language model
model_dir = snapshot_download('sherry12334/SpikingBrain-7B-VL')

# OR download quantized model (W8ASpike)
model_dir = snapshot_download('Abel2076/SpikingBrain-7B-W8ASpike')
```

### Model Sizes

| Model Variant | Size | Use Case |
|---------------|------|----------|
| Base (7B) | ~14 GB | Pre-training, research |
| Chat (7B-SFT) | ~14 GB | Conversational AI |
| VLM (7B-VL) | ~15 GB | Vision-language tasks |
| W8ASpike (quantized) | ~7 GB | Neuromorphic deployment |

---

## Performance Targets

Based on the SpikingBrain technical report:

| Metric | Target | Notes |
|--------|--------|-------|
| **Sparsity** | 69%+ | Micro-level (spiking) |
| **TTFT Speedup** | 100× | For 4M token sequences |
| **Memory Reduction** | 40% | vs. standard attention |
| **Accuracy** | ≈ Llama-7B | Comparable to baselines |
| **Latency per Layer** | < 1ms | Neuromorphic hardware goal |
| **Energy per Inference** | 10-100× lower | vs. GPU baseline |
| **Throughput** | 100-1000 tok/s | Depends on hardware |

---

## Code Examples

### Example 1: Basic Spike Encoding

```python
from W8ASpike.Int2Spike.neuron import SpikeCountTernaryLIFNode
import torch

# Simulate transformer activation
activation = torch.randn(8, 512) * 0.5  # 8 tokens, 512 dims

# Quantize to int8
quantized = torch.round(activation * 127).clamp(-128, 127)

# Encode to ternary spikes
lif = SpikeCountTernaryLIFNode()
spike_train = lif(quantized.float())

print(f"Input shape: {activation.shape}")
print(f"Spike train shape: {spike_train.shape}")
print(f"Timesteps: {spike_train.shape[0]}")
print(f"Firing rate: {lif.firing_rate():.3f}")
print(f"Sparsity: {1 - lif.firing_rate():.1%}")
```

### Example 2: Hardware Adapter Template

```python
class NeuronChipAdapter:
    """Adapt SpikingBrain spikes to your hardware"""

    def __init__(self, hardware_interface):
        self.hardware = hardware_interface
        self.lif_encoder = SpikeCountTernaryLIFNode()

    def process_layer(self, activations, weights):
        """Process one transformer layer on hardware"""

        # 1. Quantize activations
        quantized = torch.round(activations * 127).clamp(-128, 127)

        # 2. Encode to spikes
        spike_train = self.lif_encoder(quantized.float())

        # 3. Send to hardware
        for t in range(spike_train.shape[0]):
            self.hardware.send_spikes(spike_train[t])

        # 4. Get accumulated output
        output = self.hardware.get_output()

        return output
```

### Example 3: Full Inference Pipeline

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained(
    'V1-7B-base',
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained('V1-7B-base')

# Inference
prompt = "What is neuromorphic computing?"
inputs = tokenizer(prompt, return_tensors='pt')

outputs = model.generate(
    inputs.input_ids,
    max_new_tokens=128,
    temperature=0.7,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

---

## Bandwidth & Latency Calculations

### Per-Layer Spike Data

For a single transformer layer:

```
Configuration:
- Hidden size: 3584
- Batch size: 1
- Average timesteps (ternary): ~8
- Bits per spike: 2 (for ±1/0)

Data per layer:
= 3584 neurons × 1 batch × 8 timesteps × 2 bits
= 57,344 bits
= 7.2 KB per layer
```

### Full Model Forward Pass

```
Total layers: 28
Data per forward pass:
= 28 layers × 7.2 KB
= 201.6 KB per token

For batch_size = 32:
= 201.6 KB × 32
= 6.45 MB per forward pass
```

### Bandwidth Requirements

```
Target throughput: 1000 tokens/sec

Required bandwidth:
= 1000 tokens/sec × 201.6 KB/token
= 201.6 MB/sec
= ~1.6 Gbps

With 70% sparsity:
= 1.6 Gbps × 0.3 (active spikes)
= ~0.5 Gbps (effective)
```

---

## Testing & Validation

### Unit Tests

1. **Spike Encoding Accuracy**
   ```bash
   python W8ASpike/Int2Spike/test.py
   ```

2. **Reconstruction Error**
   ```python
   # Should be zero for quantized integers
   assert torch.abs(original - reconstructed).sum() < 1e-5
   ```

3. **Hardware Interface**
   ```python
   # Test spike I/O
   test_spikes = torch.randint(-1, 2, (10, 512))
   adapter.send_spikes(test_spikes)
   received = adapter.receive_spikes()
   assert torch.equal(test_spikes, received)
   ```

### Integration Tests

1. **End-to-End Latency**
   - Target: < 28ms for 28 layers (< 1ms per layer)
   - Measure: Time from input → final output

2. **Accuracy Validation**
   - Run standard LLM benchmarks (MMLU, C-Eval, etc.)
   - Compare with GPU baseline
   - Acceptable degradation: < 1%

3. **Power Profiling**
   - Measure power per forward pass
   - Compare with GPU (A100 baseline)
   - Target: 10-100× improvement

---

## Directory Structure

```
SpikingBrain-7B/
├── ARCHITECTURE_GUIDE.md          # Detailed architecture explanation
├── NEURONCHIP_INTEGRATION.md      # This file
├── README.md                       # Main repository README
│
├── demos/                          # Integration demos
│   ├── README.md                   # Demo documentation
│   ├── simple_spike_demo.py       # No-dependency demo ✓
│   ├── neuronchip_spike_demo.py   # Full PyTorch demo
│   └── png/                        # Generated visualizations
│
├── W8ASpike/                       # Spiking quantization
│   ├── Int2Spike/
│   │   ├── neuron.py              # LIF neuron implementations
│   │   ├── demo.py                # Spike encoding demos
│   │   └── test.py                # Unit tests
│   ├── quant_linear.py            # Quantized linear layers
│   └── README.md
│
├── hf_7B_model/                    # HuggingFace model
│   ├── modeling_gla_swa.py        # Main model architecture
│   ├── gla_attention.py           # Gated Linear Attention
│   ├── window_attention.py        # Sliding Window Attention
│   └── configuration_gla_swa.py   # Model config
│
├── run_model/                      # Inference examples
│   ├── run_model_hf.py            # HuggingFace inference
│   └── run_model_vllm.py          # vLLM inference
│
└── vllm_hymeta/                    # vLLM plugin for deployment
```

---

## Frequently Asked Questions

### Q: Which spike encoding should I use?

**A:** We recommend **Ternary encoding (-1/0/+1)** for most cases:
- Best accuracy/efficiency tradeoff
- Natural signed value representation
- 60-70% sparsity
- Standard in neuromorphic hardware

Use **Bitwise** only if you need fixed, ultra-low latency.

### Q: What hardware interface is needed?

**A:** Options:
1. **PCIe** - Standard, high bandwidth
2. **Custom interconnect** - If integrating on same die
3. **Network** - For distributed systems
4. **USB/Serial** - For prototyping

Minimum bandwidth: ~0.5 Gbps (with sparsity)

### Q: Can I run without downloading the full model?

**A:** Yes! The demos work standalone:
```bash
python3 demos/simple_spike_demo.py  # No model needed
```

For full inference, you need model weights (~14 GB).

### Q: What's the energy improvement?

**A:** Expected **10-100× better** than GPU:
- Sparsity: 70% fewer operations
- Low-precision: int8 vs. float16/32
- Event-driven: Power only on spikes
- Simpler circuits: No FPU needed

Actual improvement depends on your hardware implementation.

### Q: Is the quantization lossless?

**A:** Yes, for quantized integer values:
- Spike encoding/decoding is exact
- Quantization (float → int8) has standard quantization error
- Overall accuracy ≈ Llama-7B/Qwen-7B

### Q: How do I integrate with existing transformers code?

**A:** Two approaches:

1. **Post-processing**: Add spike encoding after each layer
2. **Custom kernels**: Replace linear layers with spike-based ops

See `NeuronChipAdapter` in ARCHITECTURE_GUIDE.md.

---

## Next Steps

1. ✅ **Run the demos**
   ```bash
   cd demos
   python3 simple_spike_demo.py
   ```

2. 📚 **Read the architecture guide**
   ```bash
   less ARCHITECTURE_GUIDE.md
   ```

3. 🔬 **Test spike encoding**
   ```bash
   python W8ASpike/Int2Spike/demo.py
   ```

4. 💾 **Download model weights** (optional, for full inference)
   ```python
   from modelscope import snapshot_download
   model_dir = snapshot_download('Panyuqi/V1-7B-base')
   ```

5. 🛠️ **Design hardware interface**
   - Review bandwidth calculations above
   - Choose spike encoding method
   - Design spike I/O circuits

6. 🔌 **Implement adapter**
   - Start with `NeuronChipAdapter` template
   - Add your hardware interface
   - Test with demo spike patterns

7. ✅ **Validate end-to-end**
   - Accuracy on benchmarks
   - Latency profiling
   - Power measurements

---

## Resources

### Documentation
- **Architecture Guide**: `ARCHITECTURE_GUIDE.md`
- **Demo README**: `demos/README.md`
- **Main README**: `README.md`

### Papers
- Technical Report (English): `SpikingBrain_Report_Eng.pdf`
- Technical Report (Chinese): `SpikingBrain_Report_Chi.pdf`
- ArXiv: https://arxiv.org/abs/2509.05276

### Code
- Spike encoding: `W8ASpike/Int2Spike/neuron.py`
- Model architecture: `hf_7B_model/modeling_gla_swa.py`
- Inference examples: `run_model/`

### Model Weights
- Base: https://www.modelscope.cn/models/Panyuqi/V1-7B-base
- Chat: https://www.modelscope.cn/models/Panyuqi/V1-7B-sft-s3-reasoning
- VLM: https://www.modelscope.cn/models/sherry12334/SpikingBrain-7B-VL
- Quantized: https://www.modelscope.cn/models/Abel2076/SpikingBrain-7B-W8ASpike

---

## Support

For questions about neuromorphic integration:
- **Website**: https://neuronchip.org
- **GitHub**: https://github.com/BICLab/SpikingBrain-7B
- **Demo**: https://openbayes.com/console/public/tutorials/eKBhv3jUkWw

---

## License

Apache 2.0 - See LICENSE file

---

**Ready to build the future of neuromorphic AI!** 🧠⚡🚀

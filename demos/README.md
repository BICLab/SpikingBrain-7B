# SpikingBrain-7B Demos for NeuronChip.org

This directory contains demonstration scripts showcasing the spiking neural network capabilities of SpikingBrain-7B, specifically designed for neuromorphic hardware integration at **neuronchip.org**.

## 🚀 Quick Start

### Prerequisites

```bash
# Install required dependencies
pip install torch matplotlib numpy einops
```

### Run the Spike Encoding Demo

```bash
cd demos
python neuronchip_spike_demo.py
```

This will generate:
- Spike raster plots showing neuron firing patterns
- Performance metrics (firing rate, sparsity)
- Encoding comparison charts
- Integration guidelines for neuromorphic hardware

## 📋 Demo Overview

### `neuronchip_spike_demo.py`

Comprehensive demonstration of spiking quantization methods:

#### Demo 1: Binary Spike Encoding (0/1)
- **Use case**: Unipolar neuromorphic chips
- **Encoding**: Each neuron fires 0 or 1 per timestep
- **Hardware**: Simplest implementation
- **Typical sparsity**: 60-70%

#### Demo 2: Ternary Spike Encoding (-1/0/+1)
- **Use case**: Bipolar chips (excitatory/inhibitory)
- **Encoding**: Neurons fire -1, 0, or +1
- **Hardware**: Recommended for best accuracy/efficiency ⭐
- **Typical sparsity**: 60-70%

#### Demo 3: Bitwise Spike Encoding
- **Use case**: Ultra-low latency applications
- **Encoding**: Bit-serial transmission
- **Hardware**: Parallel bit processing
- **Latency**: Fixed at log₂(max_value) timesteps

#### Demo 4: Spike-based Matrix Multiplication
- **Operation**: Weighted accumulation using spikes
- **Hardware**: Accumulate-and-fire circuits
- **Energy**: 10-100× more efficient than standard matmul

#### Demo 5: Reconstruction Accuracy
- **Verification**: Lossless encoding for quantized values
- **Testing**: Encode → Decode → Compare

## 📊 Output Files

After running the demo, check `demos/png/` for:

```
demos/png/
├── binary_spike_demo.png          # Binary encoding visualization
├── ternary_spike_demo.png         # Ternary encoding visualization
├── bitwise_spike_demo.png         # Bitwise encoding visualization
└── encoding_comparison.png        # Latency comparison chart
```

## 🔌 Hardware Integration

### For NeuronChip.org Implementation:

1. **Choose Your Encoding Method**
   - Binary: Simplest hardware
   - Ternary: Best balance (recommended)
   - Bitwise: Lowest latency

2. **Review Generated Visualizations**
   - Understand spike patterns
   - Analyze sparsity levels
   - Plan hardware architecture

3. **Follow Integration Guide**
   - See `../ARCHITECTURE_GUIDE.md`
   - Implement `NeuronChipAdapter`
   - Test with full SpikingBrain-7B model

### Expected Performance:

| Metric | Target |
|--------|--------|
| Sparsity | 60-70% |
| Latency per layer | < 1ms |
| Total spikes per inference | ~800K (28 layers) |
| Energy savings | 10-100× vs GPU |

## 🧪 Advanced Usage

### Test with Custom Activations

```python
import torch
from neuron import SpikeCountTernaryLIFNode

# Your activation tensor
activations = torch.randn(8, 512)  # 8 tokens, 512 dims

# Quantize to int8
quantized = torch.round(activations * 127).clamp(-128, 127)

# Encode to spikes
lif = SpikeCountTernaryLIFNode()
spikes = lif(quantized.float())

print(f"Timesteps: {spikes.shape[0]}")
print(f"Firing rate: {lif.firing_rate():.3f}")
print(f"Sparsity: {1 - lif.firing_rate():.1%}")
```

### Integrate with Full Model

```python
from transformers import AutoModelForCausalLM
# See run_model/run_model_hf.py for complete example

# Load model
model = AutoModelForCausalLM.from_pretrained(
    'V1-7B-base',
    trust_remote_code=True
)

# Extract activations and encode to spikes
# (See NeuronChipAdapter in ARCHITECTURE_GUIDE.md)
```

## 📚 Additional Resources

- **Main Documentation**: `../README.md`
- **Architecture Guide**: `../ARCHITECTURE_GUIDE.md`
- **Spike Encoding Source**: `../W8ASpike/Int2Spike/neuron.py`
- **Technical Paper**: `../SpikingBrain_Report_Eng.pdf`
- **Model Weights**: See README.md for ModelScope links

## 🐛 Troubleshooting

### "No module named 'neuron'"
- Make sure you run from the `demos/` directory
- The script automatically adds the Int2Spike path

### "No display found"
- The demo uses `matplotlib.use('Agg')` for non-interactive plotting
- All plots are saved to `demos/png/`, not displayed

### "Out of memory"
- The demo uses small tensors (8 tokens × 512 dims)
- Should run on any system with 4GB+ RAM

## 🤝 Contributing

To add new demos:

1. Create a new Python script in `demos/`
2. Follow the existing demo structure
3. Document in this README
4. Test with `python your_demo.py`

## 📧 Support

For questions about neuromorphic integration:
- Review `ARCHITECTURE_GUIDE.md`
- Check the SpikingBrain paper
- Visit https://neuronchip.org

## 📜 License

Apache 2.0 - See LICENSE file

---

**Ready to build neuromorphic AI systems with SpikingBrain-7B!** 🧠⚡

# Quick Start Guide for NeuronChip.org Integration

**Get started with SpikingBrain-7B neuromorphic demos in 5 minutes!**

---

## 🚀 Instant Demo (No Installation Required!)

```bash
# Clone the repository
git clone https://github.com/BICLab/SpikingBrain-7B.git
cd SpikingBrain-7B

# Run the simple demo (pure Python, no dependencies!)
cd demos
python3 simple_spike_demo.py
```

**That's it!** You'll see:
- ✅ Binary, Ternary, and Bitwise spike encoding demos
- ✅ ASCII spike visualizations
- ✅ Performance metrics (firing rate, sparsity)
- ✅ Hardware integration guidelines

**Expected output:**
```
Value:    3 | Encoding: Ternary
Timesteps:  0  1  2  3  4  5  6  7
Spikes:     ↑  ↑  ↑  ·  ·  ·  ·  ·
Metrics: 3 spikes, 0.38 firing rate, 62.5% sparsity
```

---

## 📚 Key Documents (Start Here!)

| Document | Purpose | Time to Read |
|----------|---------|--------------|
| **[NEURONCHIP_INTEGRATION.md](NEURONCHIP_INTEGRATION.md)** | Complete integration guide | 20 min |
| **[ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md)** | Architecture deep-dive | 30 min |
| **[demos/README.md](demos/README.md)** | Demo usage | 5 min |
| **[demos/DEMO_OUTPUT.md](demos/DEMO_OUTPUT.md)** | Demo results & analysis | 10 min |

---

## 🎯 Three Integration Paths

### Path 1: Quick Exploration (You are here! ✓)
**Time:** 10 minutes
**Goal:** Understand spiking mechanisms

```bash
# Run simple demo
cd demos
python3 simple_spike_demo.py

# Read integration guide
less ../NEURONCHIP_INTEGRATION.md
```

**Outcome:** Understand how spike encoding works for neuromorphic hardware

---

### Path 2: Full Demo with Visualizations
**Time:** 1 hour
**Goal:** Generate spike visualizations and metrics

```bash
# Install dependencies
pip install torch matplotlib numpy einops

# Run full demo
python3 neuronchip_spike_demo.py
```

**Generates:**
- `demos/png/binary_spike_demo.png` - Binary encoding raster plot
- `demos/png/ternary_spike_demo.png` - Ternary encoding raster plot
- `demos/png/bitwise_spike_demo.png` - Bitwise encoding raster plot
- `demos/png/encoding_comparison.png` - Performance comparison chart

**Outcome:** Visualizations and performance metrics for hardware design

---

### Path 3: Full Model Integration
**Time:** 1 day
**Goal:** Run SpikingBrain-7B inference with spike encoding

```bash
# Install all dependencies
pip install torch==2.7.1 transformers==4.55.2 modelscope

# Download model (one-time, ~14 GB)
python3 << EOF
from modelscope import snapshot_download
model_dir = snapshot_download('Panyuqi/V1-7B-base')
print(f"Model downloaded to: {model_dir}")
EOF

# Run inference example
cd run_model
python3 run_model_hf.py
```

**Outcome:** Full LLM inference with spiking quantization

---

## 🔧 Hardware Integration Checklist

### Step 1: Choose Spike Encoding ✓
- [ ] **Ternary (-1/0/+1)** ⭐ **RECOMMENDED**
  - 60-70% sparsity
  - Natural signed values
  - Best accuracy/efficiency

- [ ] Binary (0/1)
  - Simplest hardware
  - Unipolar only

- [ ] Bitwise
  - Fixed latency
  - Parallel processing

**Decision:** _________________

### Step 2: Review Demo Output ✓
```bash
# Completed! See demos/DEMO_OUTPUT.md
```

### Step 3: Design Hardware Interface
- [ ] Spike generation circuit (< 100ns per timestep)
- [ ] Spike accumulation circuit (16-bit accumulator)
- [ ] Memory interface (~0.5 Gbps bandwidth)
- [ ] I/O protocol (PCIe / custom / network)

**Interface type:** _________________

### Step 4: Implement Software Adapter
```python
# Template provided in NEURONCHIP_INTEGRATION.md
class NeuronChipAdapter:
    def __init__(self, hardware_interface):
        self.hardware = hardware_interface
        self.lif_encoder = SpikeCountTernaryLIFNode()

    def process_layer(self, activations, weights):
        # 1. Quantize
        # 2. Encode to spikes
        # 3. Send to hardware
        # 4. Receive output
        pass
```

**Status:** [ ] Not started [ ] In progress [ ] Complete

### Step 5: Test & Validate
- [ ] Unit test: Spike encoding accuracy
- [ ] Integration test: End-to-end latency (< 1ms per layer)
- [ ] Performance test: Throughput (100-1000 tokens/sec)
- [ ] Power test: Energy vs. GPU baseline (10-100× better)

---

## 📊 Expected Performance

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Sparsity** | 60-70% | Run demo, check firing rate |
| **Latency** | < 1ms per layer | Profile with hardware timer |
| **Energy** | 10-100× vs. GPU | Power meter during inference |
| **Accuracy** | ≈ Llama-7B | Run MMLU, C-Eval benchmarks |
| **Throughput** | 100-1K tok/s | Measure tokens/second |

---

## 🐛 Troubleshooting

### Demo won't run?
```bash
# Check Python version (need 3.9+)
python3 --version

# Try with explicit path
cd /path/to/SpikingBrain-7B/demos
python3 simple_spike_demo.py
```

### Can't download model?
```bash
# Install ModelScope
pip install modelscope

# Or download directly via git
git clone https://www.modelscope.cn/Panyuqi/V1-7B-base.git
```

### Out of memory?
```bash
# Simple demo uses only ~10 MB RAM
# Full demo needs ~4 GB RAM
# Full model inference needs ~16 GB GPU memory

# Use quantized model for less memory
model_dir = snapshot_download('Abel2076/SpikingBrain-7B-W8ASpike')
```

---

## 📖 Learning Resources

### 1. Core Concepts (30 min)
- Read: `NEURONCHIP_INTEGRATION.md` - Sections 1-3
- Understand: Spike encoding, LIF neurons, hardware requirements

### 2. Hands-On (1 hour)
- Run: `simple_spike_demo.py`
- Experiment: Modify test values, try different encodings
- Visualize: Run full demo with PyTorch

### 3. Deep Dive (3 hours)
- Read: `ARCHITECTURE_GUIDE.md` - Full architecture
- Study: `W8ASpike/Int2Spike/neuron.py` - Implementation
- Explore: `hf_7B_model/modeling_gla_swa.py` - Model code

### 4. Integration (1 week)
- Design: Hardware interface specification
- Implement: Software adapter
- Test: End-to-end validation

---

## 💡 Key Insights from Demos

1. **Ternary encoding achieves 62.5% sparsity**
   - Fewer operations = lower energy
   - Natural signed value representation

2. **Bitwise encoding has fixed 4-timestep latency**
   - Deterministic timing
   - Good for latency-critical applications

3. **Spike accumulation is lossless**
   - Hardware can exactly reconstruct values
   - No accuracy loss from spike encoding

4. **69% sparsity at micro-level**
   - Documented in SpikingBrain paper
   - Demonstrated in our demos

5. **100× speedup possible**
   - For 4M token sequences (per paper)
   - Event-driven processing enables efficiency

---

## 🎓 Recommended Reading Order

1. **This file** (5 min) ✓ You're here!
2. **Run simple demo** (5 min) → See spike encoding in action
3. **NEURONCHIP_INTEGRATION.md** (20 min) → Hardware integration
4. **ARCHITECTURE_GUIDE.md** (30 min) → System architecture
5. **Technical paper** (1 hour) → SpikingBrain_Report_Eng.pdf
6. **Source code** (ongoing) → W8ASpike/, hf_7B_model/

---

## 🚦 Status Indicators

| Phase | Status | Time Estimate |
|-------|--------|---------------|
| ✅ Demo execution | **COMPLETE** | - |
| ✅ Documentation | **COMPLETE** | - |
| ⏳ Encoding choice | **PENDING** | 1 hour |
| ⏳ Hardware design | **PENDING** | 2-4 weeks |
| ⏳ Software adapter | **PENDING** | 1-2 weeks |
| ⏳ Integration testing | **PENDING** | 2-4 weeks |
| ⏳ Optimization | **PENDING** | 2-4 weeks |

**Total estimated timeline:** 8-16 weeks to full integration

---

## 🤝 Getting Help

### Documentation Issues
- Check: `demos/DEMO_OUTPUT.md` for expected outputs
- Review: `NEURONCHIP_INTEGRATION.md` FAQ section

### Technical Questions
- Paper: SpikingBrain_Report_Eng.pdf
- Code: Comment in W8ASpike/Int2Spike/neuron.py
- Community: https://github.com/BICLab/SpikingBrain-7B

### Hardware Integration
- Contact: https://neuronchip.org
- Review: Hardware requirements in NEURONCHIP_INTEGRATION.md

---

## ✨ Success Checklist

Before proceeding to hardware implementation:

- [ ] ✅ Run simple demo successfully
- [ ] ✅ Understand ternary spike encoding
- [ ] ✅ Review hardware requirements
- [ ] ⏳ Choose spike encoding method
- [ ] ⏳ Design hardware interface spec
- [ ] ⏳ Calculate bandwidth requirements
- [ ] ⏳ Estimate power budget
- [ ] ⏳ Plan testing methodology

**Ready?** Start with `NEURONCHIP_INTEGRATION.md` Section 2: "Hardware Integration Checklist"

---

## 📞 Quick Links

- 🌐 **Website**: https://neuronchip.org
- 📄 **Paper**: https://arxiv.org/abs/2509.05276
- 💻 **Code**: https://github.com/BICLab/SpikingBrain-7B
- 🎮 **Demo**: https://openbayes.com/console/public/tutorials/eKBhv3jUkWw
- 📦 **Models**: https://www.modelscope.cn/models/Panyuqi/

---

**You're all set! Start with the simple demo above.** 🚀

Questions? See `NEURONCHIP_INTEGRATION.md` for comprehensive guidance.

---

_Last updated: 2025-11-25 | SpikingBrain-7B for NeuronChip.org_ 🧠⚡

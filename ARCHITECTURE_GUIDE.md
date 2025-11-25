# SpikingBrain-7B System Architecture Guide for NeuronChip.org

## System Architecture Overview (ASCII Diagram)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        SpikingBrain-7B Architecture                         │
└────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │ Input Tokens │
                              └──────┬───────┘
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │  Token Embeddings      │
                        │  (vocab_size → 3584)   │
                        └────────────┬───────────┘
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │            28 Hybrid Transformer Blocks                │
        │                                                         │
        │  ┌──────────────────────────────────────────────────┐ │
        │  │  Block Pattern (Alternating):                    │ │
        │  │  • Odd layers (1,3,5...): FlashAttention (SWA)  │ │
        │  │  • Even layers (0,2,4...): GatedLinearAttn (GLA)│ │
        │  └──────────────────────────────────────────────────┘ │
        │                                                         │
        │  Each Block Contains:                                  │
        │  ┌──────────────────────────────────────────────────┐ │
        │  │                                                   │ │
        │  │  1. RMS Normalization                            │ │
        │  │     ↓                                             │ │
        │  │  2. Attention Layer (FlashAttn OR GLA)           │ │
        │  │     │                                             │ │
        │  │     ├─→ FlashAttention (Sliding Window):         │ │
        │  │     │   • Window size: 4096 tokens               │ │
        │  │     │   • Multi-head: 28 heads                   │ │
        │  │     │   • KV heads: 4 (GQA)                      │ │
        │  │     │   • RoPE embeddings                        │ │
        │  │     │                                             │ │
        │  │     └─→ Gated Linear Attention (GLA):            │ │
        │  │         • Linear complexity O(n)                 │ │
        │  │         • Chunk-based processing                 │ │
        │  │         • Short convolution (optional)           │ │
        │  │         • Feature gating mechanism               │ │
        │  │     ↓                                             │ │
        │  │  3. Residual Connection                          │ │
        │  │     ↓                                             │ │
        │  │  4. RMS Normalization (fused)                    │ │
        │  │     ↓                                             │ │
        │  │  5. MLP (SwiGLU):                                │ │
        │  │     • gate_proj: 3584 → 18944                    │ │
        │  │     • up_proj: 3584 → 18944                      │ │
        │  │     • activation: swish                          │ │
        │  │     • down_proj: 18944 → 3584                    │ │
        │  │     ↓                                             │ │
        │  │  6. Residual Connection                          │ │
        │  │                                                   │ │
        │  └──────────────────────────────────────────────────┘ │
        └─────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
                       ┌────────────────────┐
                       │  Final RMS Norm    │
                       └──────────┬─────────┘
                                  │
                                  ▼
                       ┌────────────────────┐
                       │  LM Head           │
                       │  (3584 → 152064)   │
                       └──────────┬─────────┘
                                  │
                                  ▼
                            ┌──────────┐
                            │  Logits  │
                            └──────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│                    Spiking Quantization Layer (W8ASpike)                   │
└────────────────────────────────────────────────────────────────────────────┘

                      Input Activations (Float32)
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Quantizer            │
                    │  (Float → Int8)       │
                    └──────────┬────────────┘
                               │
                               ▼
                    ┌───────────────────────┐
                    │  Spike Encoder        │
                    │  (LIF Neuron Model)   │
                    │                       │
                    │  Options:             │
                    │  • Binary (0/1)       │
                    │  • Ternary (-1/0/+1)  │
                    │  • Bitwise coding     │
                    └──────────┬────────────┘
                               │
                               ▼
                    ┌───────────────────────┐
                    │  Spike Sequences      │
                    │  [T, batch, hidden]   │
                    │  T = time steps       │
                    └──────────┬────────────┘
                               │
                               ▼
                    ┌───────────────────────┐
                    │  Spike Matrix Mult    │
                    │  (Accumulate over T)  │
                    └──────────┬────────────┘
                               │
                               ▼
                    ┌───────────────────────┐
                    │  Spike Decoder        │
                    │  (Reconstruct value)  │
                    └──────────┬────────────┘
                               │
                               ▼
                        Output Activations

┌────────────────────────────────────────────────────────────────────────────┐
│                         Inference Pipeline (vLLM)                          │
└────────────────────────────────────────────────────────────────────────────┘

    User Request → vLLM Server → Model Loading → KV Cache Init
                       │                              │
                       │                              ▼
                       │                    ┌─────────────────┐
                       │                    │  Prefill Phase  │
                       │                    │  (Full Context) │
                       │                    └────────┬────────┘
                       │                             │
                       │                             ▼
                       │                    ┌─────────────────┐
                       │                    │ Decode Phase    │
                       │                    │ (Token by token)│
                       │                    └────────┬────────┘
                       │                             │
                       └─────────────────────────────┴────→ Response

         Performance Metrics:
         • TTFT: 100× faster for 4M token sequences
         • Sparsity: 69% at micro-level (spiking)
         • MoE sparsity: Additional macro-level efficiency
```

---

## Detailed Component Breakdown

### 1. **Hybrid Attention Mechanism**

The system uses TWO types of attention, alternating between layers:

#### A. Flash Attention (Sliding Window Attention - SWA)
- **Location**: Odd layers (1, 3, 5, 7, ...)
- **Purpose**: Captures local dependencies efficiently
- **Key Features**:
  - Sliding window of 4096 tokens
  - Flash Attention 2 implementation
  - Grouped Query Attention (28 Q heads, 4 KV heads)
  - RoPE positional embeddings (theta=1M for long context)
  - Memory efficient O(n) complexity within window

#### B. Gated Linear Attention (GLA)
- **Location**: Even layers (0, 2, 4, 6, ...)
- **Purpose**: Captures global dependencies with linear complexity
- **Key Features**:
  - Linear time complexity O(n)
  - Chunk-based processing for efficiency
  - Optional short convolutions (1D conv, size=4)
  - Feature gating via low-rank projection
  - Three modes: chunk, fused_recurrent, fused_chunk

**File**: `hf_7B_model/gla_attention.py:36-100`

### 2. **Spiking Neural Network Integration (W8ASpike)**

The quantization layer converts activations to spike trains:

#### Spike Encoding Methods:

**a) Binary Encoding (0/1)**
- Input: Non-negative integers
- Output: Binary spike train
- Fires 1 each timestep until count reaches zero
- Implemented in: `W8ASpike/Int2Spike/neuron.py:97-146`

**b) Ternary Encoding (-1/0/+1)**
- Input: Signed integers
- Output: Ternary spike train
- +1 for positive, -1 for negative
- Implemented in: `W8ASpike/Int2Spike/neuron.py:147-192`

**c) Bitwise Encoding**
- Input: Integers
- Output: Bitwise spike representation
- One bit per timestep
- Optional two's complement support
- Implemented in: `W8ASpike/Int2Spike/neuron.py:193-277`

#### LIF Neuron Model Components:
1. **Charge Phase**: Accumulate input spike count
2. **Fire Phase**: Generate spike if threshold exceeded
3. **Reset Phase**: Subtract fired spike from potential

**File**: `W8ASpike/Int2Spike/neuron.py:8-192`

### 3. **Model Configuration**

Key parameters from `hf_7B_model/configuration_gla_swa.py:29-83`:

```python
vocab_size: 152064
hidden_size: 3584
num_hidden_layers: 28
num_attention_heads: 28
num_key_value_heads: 4  # GQA
intermediate_size: 18944  # MLP expansion
max_position_embeddings: 131072  # 4096 * 32
sliding_window: 4096
attn_mode: "chunk"
use_short_conv: False
conv_size: 4
```

---

## How to Build Your NeuronChip.org System

### Phase 1: Core Model Setup

#### Step 1.1: Clone and Install Base System
```bash
git clone https://github.com/BICLab/SpikingBrain-7B.git
cd SpikingBrain-7B

# Install dependencies
pip install torch==2.7.1 transformers==4.55.2
pip install triton==3.3.1 flash-attn==2.7.3
pip install flash-linear-attention==0.1
pip install vllm==0.10.0
pip install scipy pyyaml decorator setuptools setuptools-scm
```

#### Step 1.2: Download Pre-trained Weights
```python
from modelscope import snapshot_download

# Choose your model variant:
# Base model
model_dir = snapshot_download('Panyuqi/V1-7B-base')

# OR Chat model
model_dir = snapshot_download('Panyuqi/V1-7B-sft-s3-reasoning')

# OR Vision-Language model
model_dir = snapshot_download('sherry12334/SpikingBrain-7B-VL')

# OR Quantized model
model_dir = snapshot_download('Abel2076/SpikingBrain-7B-W8ASpike')
```

#### Step 1.3: Test Basic Inference
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Test generation
text = "What is neuromorphic computing?"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

### Phase 2: NeuronChip Integration

#### Step 2.1: Understanding the Data Flow

```
Your NeuronChip Hardware
        │
        ▼
┌───────────────────┐
│  Spike Interface  │  ← You need to implement this
│  (Hardware ↔ SW)  │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ W8ASpike Pipeline │  ← Already implemented
│ (Float → Spikes)  │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ SpikingBrain-7B   │  ← Pre-trained model
│ Transformer Core  │
└───────────────────┘
```

#### Step 2.2: Adapt Spike Encoding for Your Hardware

Create `neuronchip_adapter.py`:

```python
import torch
from W8ASpike.Int2Spike.neuron import (
    SpikeCountBinaryLIFNode,
    SpikeCountTernaryLIFNode,
    spike_quant,
    spike_matmul
)

class NeuronChipAdapter:
    """
    Adapter to interface SpikingBrain with NeuronChip hardware
    """
    def __init__(self, encoding_type='ternary', timesteps=8):
        """
        Args:
            encoding_type: 'binary', 'ternary', or 'bitwise'
            timesteps: Number of time steps for spike encoding
        """
        self.encoding_type = encoding_type
        self.timesteps = timesteps

        # Initialize spike encoder
        if encoding_type == 'binary':
            self.lif_encoder = SpikeCountBinaryLIFNode()
        elif encoding_type == 'ternary':
            self.lif_encoder = SpikeCountTernaryLIFNode()
        else:
            raise ValueError(f"Unsupported encoding: {encoding_type}")

    def float_to_spikes(self, activations, scale=127.0):
        """
        Convert floating point activations to spike trains

        Args:
            activations: torch.Tensor (float32/bfloat16)
            scale: Quantization scale factor

        Returns:
            spike_train: torch.Tensor [T, *activations.shape]
        """
        # Quantize to int8 range
        int_acts = torch.round(activations * scale).to(torch.int8)

        # Convert to spike train
        spike_train = spike_quant(
            int_acts,
            self.lif_encoder,
            x_zero=torch.zeros_like(int_acts)
        )

        return spike_train

    def send_to_hardware(self, spike_train):
        """
        Send spike train to NeuronChip hardware
        TODO: Implement hardware interface

        Args:
            spike_train: [T, batch, neurons] tensor

        Returns:
            Hardware acknowledgment or result
        """
        # YOUR HARDWARE INTERFACE HERE
        # This is where you'll integrate with neuronchip.org
        # Example pseudo-code:
        #
        # neuronchip_api = NeuronChipAPI()
        # for t in range(spike_train.shape[0]):
        #     neuronchip_api.send_spike_timestep(spike_train[t])
        # result = neuronchip_api.get_output()
        # return result

        pass

    def compute_with_spikes(self, x, weight):
        """
        Perform spiking matrix multiplication

        Args:
            x: Input activations (int tensor)
            weight: Weight matrix

        Returns:
            Output after spike-based computation
        """
        return spike_matmul(
            x,
            weight,
            lif_quantizer=self.lif_encoder
        )
```

#### Step 2.3: Create Custom Model Wrapper

Create `neuronchip_model.py`:

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class NeuronChipSpikingModel(nn.Module):
    """
    Wrapper that integrates SpikingBrain with NeuronChip hardware
    """
    def __init__(self, model_path, adapter):
        super().__init__()

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        self.adapter = adapter

    def forward_with_spikes(self, input_ids):
        """
        Forward pass using spike encoding
        """
        # Get embeddings
        embeddings = self.base_model.model.embeddings(input_ids)

        # Convert to spikes
        spike_embeddings = self.adapter.float_to_spikes(embeddings)

        # Process through layers
        hidden_states = spike_embeddings

        for layer in self.base_model.model.layers:
            # Each layer processes spike trains
            hidden_states = self._process_layer_with_spikes(
                layer,
                hidden_states
            )

        # Final norm and output
        hidden_states = self.base_model.model.norm(hidden_states)
        logits = self.base_model.lm_head(hidden_states)

        return logits

    def _process_layer_with_spikes(self, layer, spike_states):
        """
        Process a single transformer layer with spikes
        """
        # YOUR NEURONCHIP PROCESSING HERE
        # This is where you'd interface with hardware

        # For now, use standard computation
        # Replace this with hardware calls
        return layer(spike_states)
```

### Phase 3: vLLM Server Setup for Production

#### Step 3.1: Deploy with Docker

```bash
# Build container
docker build -t neuronchip-spikingbrain:v1.0 .

# Run server
docker run -itd \
    --entrypoint /bin/bash \
    --network host \
    --name neuronchip-server \
    --shm-size 160g \
    --gpus all \
    --privileged \
    -v /path/to/models:/models \
    neuronchip-spikingbrain:v1.0
```

#### Step 3.2: Start vLLM Service

```bash
vllm serve /models/SpikingBrain-7B \
  --served-model-name neuronchip-7b \
  --gpu-memory-utilization 0.9 \
  --block-size 16 \
  --dtype bfloat16 \
  --port 8000 \
  --tensor-parallel-size 1
```

#### Step 3.3: API Client Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123"
)

response = client.chat.completions.create(
    model="neuronchip-7b",
    messages=[
        {"role": "user", "content": "Explain spiking neural networks"}
    ],
    max_tokens=512
)

print(response.choices[0].message.content)
```

---

## Requirements Checklist for NeuronChip.org

### Software Requirements

#### Core Dependencies
```
Python: 3.9+
PyTorch: 2.7.1
Transformers: 4.55.2
Triton: 3.3.1
Flash Attention: 2.7.3
Flash Linear Attention: 0.1
vLLM: 0.10.0
```

#### Additional Libraries
```
scipy
pyyaml
decorator
setuptools
setuptools-scm
einops
modelscope (for downloading models)
```

### Hardware Requirements

#### Minimum for Inference:
- **GPU**: NVIDIA A100 40GB or equivalent
- **RAM**: 64GB system RAM
- **Storage**: 50GB for model weights
- **Network**: High-bandwidth for model download

#### Recommended for Production:
- **GPU**: NVIDIA A100 80GB or H100
- **RAM**: 128GB+ system RAM
- **Storage**: 200GB SSD
- **Network**: 10Gbps+

#### NeuronChip Hardware Integration:
- **Interface**: PCIe or custom interconnect
- **Spike I/O**: High-speed serial/parallel interface
- **Latency**: <1ms per layer for real-time operation
- **Bandwidth**: Calculate based on:
  ```
  Per-layer spike data:
  - Hidden size: 3584
  - Batch size: 1
  - Timesteps: 8
  - Data: 3584 × 1 × 8 = 28,672 spikes per layer
  - For 28 layers: ~800K spikes per forward pass
  ```

### Data Requirements

#### Training Data (if fine-tuning):
- **Format**: JSON/JSONL with text or conversation structure
- **Volume**: Depends on task (2% of original training recommended)
- **Preprocessing**: Tokenization with 152K vocab tokenizer

#### Inference Data:
- **Input**: Text prompts (any length up to 131K tokens)
- **Format**: String or chat messages
- **Batching**: Supported via vLLM

---

## Performance Targets

Based on the paper, you should achieve:

| Metric | Target |
|--------|--------|
| **TTFT (4M tokens)** | 100× faster than baseline |
| **Micro-level sparsity** | 69%+ from spiking |
| **Memory efficiency** | 40% reduction vs. standard attention |
| **Accuracy** | Comparable to Llama-7B/Qwen-7B |

---

## Integration Roadmap for NeuronChip.org

### Week 1-2: Environment Setup
- [ ] Set up development environment
- [ ] Install all dependencies
- [ ] Download pre-trained models
- [ ] Run basic inference tests
- [ ] Verify model outputs

### Week 3-4: Understanding Architecture
- [ ] Study hybrid attention mechanism
- [ ] Analyze spike encoding methods
- [ ] Profile memory and compute usage
- [ ] Identify hardware integration points
- [ ] Design hardware interface API

### Week 5-8: Hardware Integration
- [ ] Implement NeuronChipAdapter
- [ ] Create spike I/O drivers
- [ ] Test spike encoding/decoding
- [ ] Benchmark latency and throughput
- [ ] Optimize data transfer

### Week 9-12: System Integration
- [ ] Integrate adapter with model
- [ ] Deploy vLLM server
- [ ] Build API endpoints
- [ ] Create monitoring dashboard
- [ ] Performance testing

### Week 13-16: Optimization & Validation
- [ ] Profile end-to-end performance
- [ ] Optimize bottlenecks
- [ ] Validate accuracy
- [ ] Load testing
- [ ] Documentation

---

## Key Files Reference

### Model Architecture
- `hf_7B_model/modeling_gla_swa.py:57-125` - HybridBlock (main transformer layer)
- `hf_7B_model/configuration_gla_swa.py:24-83` - Configuration
- `hf_7B_model/gla_attention.py:36` - Gated Linear Attention
- `hf_7B_model/window_attention.py` - Sliding Window Attention

### Spike Encoding
- `W8ASpike/Int2Spike/neuron.py:8` - Base LIF neuron class
- `W8ASpike/Int2Spike/neuron.py:97` - Binary encoding
- `W8ASpike/Int2Spike/neuron.py:147` - Ternary encoding
- `W8ASpike/Int2Spike/neuron.py:279` - spike_quant function
- `W8ASpike/Int2Spike/neuron.py:451` - spike_matmul function

### Inference
- `run_model/run_model_hf.py` - HuggingFace inference
- `run_model/run_model_vllm.py` - vLLM inference
- `vllm_hymeta/` - vLLM plugin for hardware support

---

## Support Resources

1. **Technical Report**:
   - English: `SpikingBrain_Report_Eng.pdf`
   - Chinese: `SpikingBrain_Report_Chi.pdf`

2. **ArXiv Paper**: https://arxiv.org/abs/2509.05276

3. **Demo**: https://openbayes.com/console/public/tutorials/eKBhv3jUkWw

4. **Model Weights**:
   - Base: https://www.modelscope.cn/models/Panyuqi/V1-7B-base
   - Chat: https://www.modelscope.cn/models/Panyuqi/V1-7B-sft-s3-reasoning
   - Vision: https://www.modelscope.cn/models/sherry12334/SpikingBrain-7B-VL
   - Quantized: https://www.modelscope.cn/models/Abel2076/SpikingBrain-7B-W8ASpike

---

## Next Steps for NeuronChip.org

1. **Hardware Specification**: Define spike I/O interface specification
2. **Driver Development**: Build hardware drivers for spike transmission
3. **API Design**: Create REST/gRPC API for neuronchip.org
4. **Testing Framework**: Build comprehensive test suite
5. **Documentation**: Create integration guides for users
6. **Benchmarking**: Establish performance baselines
7. **Community**: Open source integration code (optional)

---

## Questions to Answer Before Starting

1. **What spike encoding will your hardware support?**
   - Binary (0/1)
   - Ternary (-1/0/+1)
   - Bitwise
   - Custom?

2. **What is your hardware interface?**
   - PCIe
   - Custom interconnect
   - Network-based
   - USB/Serial?

3. **What latency can you achieve?**
   - Target: <1ms per layer
   - Measured: ?

4. **What throughput is needed?**
   - Spikes per second
   - Tokens per second
   - Batch size

5. **What precision does your hardware support?**
   - INT8
   - INT4
   - Binary
   - Custom?

Answer these questions to customize the integration for your specific hardware.

# SpikingBrain：Spiking Brain-inspired Large Models

Our technical report: [PDF link]

---

## About SpikingBrain

Inspired by brain mechanisms, **SpikingBrain** integrates **hybrid efficient attention**, **MoE modules**, and **spike encoding** into its architecture, supported by a universal conversion pipeline compatible with the open-source model ecosystem. This enables continual pre-training with less than 2\% of the data while achieving performance comparable to mainstream open-source models. We further adapt frameworks, operators, parallel strategies, and communication primitives for **non-NVIDIA (MetaX) clusters**, ensuring stable large-scale training and inference. SpikingBrain achieves over 100× speedup in TTFT for 4M-token sequences, while spiking delivers over 69\% sparsity at the micro level. Combined with macro-level MoE sparsity, these advances provide valuable guidance for the design of next-generation neuromorphic chips.

![](assets/fig1.png)

This repository provides the full implementation and weights of **SpikingBrain-7B**, including the **HuggingFace version**, **vLLM inference version**, and **quantized version**, enabling flexible deployment and research across different scenarios.

```
SpikingBrain-7B/
├── hf_7B_model/ # HuggingFace version
├── vllm_hymeta/ # vLLM plugins and inference support
├── W8ASpike/    # Quantized inference version
├── setup.py
├── requirements.txt 
└── README.md 
```

The model weights are hosted on **ModelScope**. Please select the appropriate version based on your needs:

> Original weights (pre-traind model): https://www.modelscope.cn/models/Panyuqi/V1-7B-base
> Original weights (chat model): https://www.modelscope.cn/models/Panyuqi/V1-7B-sft-s3-reasoning
> Quantized weights: https://www.modelscope.cn/models/Abel2076/SpikingBrain-7B-W8ASpike

--- 

## vLLM-HyMeta

**vllm-hymeta** is the plugin adaptation of the HyMeta model for the [vLLM inference framework](https://github.com/vllm-project/vllm/tree/main), providing efficient inference support on NVIDIA GPUs.

(HyMeta is an acronym for Hybrid Models built on MetaX GPUs.)

By leveraging the [plugins mechanism](https://blog.vllm.ai/2025/05/12/hardware-plugin.html) in vLLM, hardware backends can be integrated in a modular fashion, bringing the following benefits:

- **Decoupled codebase**: Backend-specific code remains independent, keeping the vLLM core cleaner.

- **Reduced maintenance cost**: vLLM developers can focus on general functionality without being affected by backend-specific implementations.

- **Faster integration**: New backends can be integrated quickly and evolve independently with less engineering effort.

### Container Deployment (NVIDIA)
```bash
sudo docker run -itd \
    --entrypoint /bin/bash \
    --network host \
    --name hymeta-bench \
    --shm-size 160g \
    --gpus all \
    --privileged \
    -v /host_path:/container_path \
    --env "HF_ENDPOINT=https://hf-mirror.com" \
    docker.1ms.run/vllm/vllm-openai:v0.10.0
```

### Plugin Installation
```bash
git clone https://github.com/BICLab/SpikingBrain-7B.git
cd vllm-hymeta
pip install .
```

Recommended environment for installing **vllm-hymeta** on NVIDIA GPUs:

```makefile
decorator
pyyaml
scipy
setuptools
setuptools-scm
flash_attn==2.7.3
flash-linear-attention==0.1
vllm==0.10.0
torch==2.7.1
```

---

## W8ASpike

**W8ASpike** is the quantized inference version of SpikingBrain-7B, aiming to reduce inference cost under low-precision settings and explore the potential of Spiking Neural Networks (SNNs).

The current implementation adopts **pseudo-spiking**, where activations are approximated as spike-like signals at the tensor level, rather than true asynchronous event-driven spiking on neuromorphic hardware.

- **Pseudo-spiking**: Efficient approximation at the tensor level, suitable for prototyping and research.

- **True-spiking**: Requires asynchronous hardware and event-driven operator support, which is beyond the scope of this repository.

The activation spike encoding process here is inspired by the pseudo-spiking interfaces from [BICLab/Int2Spike](https://github.com/BICLab/Int2Spike). For additional PyTorch-based spiking interfaces, please refer to the Int2Spike library.

---

## Available Models

Table 1: **Performance evaluation of the SpikingBrain-7B pre-trained model.** All models are tested with the HuggingFace framework and evaluated using a perplexity-based method. Except for Qwen2.5, the other baselines are trained on limited Chinese data, resulting in clear disadvantages on CMMLU and C-Eval.
![](assets/table1.png)


Table 2: **Performance evaluation of the SpikingBrain-76B pre-trained model.** All models are tested with the vLLM framework and evaluated using a perplexity-based method. Except for Qwen2.5, the other baselines are trained on limited Chinese data, resulting in clear disadvantages on CMMLU and C-Eval.
![](assets/table2.png)

--- 

## Citation

If you find our work useful, please consider citing SpikingBrain:

```py
# todo
```
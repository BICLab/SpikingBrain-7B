FROM docker.1ms.run/vllm/vllm-openai:v0.10.0

WORKDIR /workspace


ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ca-certificates \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace

# install Spiking-Brain v1.0
RUN pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple  --no-build-isolation --verbose


# resolving conflict between transformers and fla
COPY ./docker_build/patch_fla_bitnet.py /tmp/patch_fla_bitnet.py

RUN python3.12 /tmp/patch_fla_bitnet.py



CMD ["/bin/bash"]
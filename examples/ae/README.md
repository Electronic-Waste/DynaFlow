# DynaFlow Artifact Evaluation

This directory contains the scripts and configuration needed to reproduce the
throughput benchmarks for **DynaFlow**.

## Environment Setup

We recommend to use image `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` as the base environment. After creating a container from this image, install the following dependencies:

```bash
apt-get update && \
apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    htop \
    neovim \
    unzip \
    ca-certificates \
    sudo \
    python3 \
    python3-pip \
    gnupg \
    lsb-release \
    openssh-client \
    software-properties-common
pip3 install uv && \
    echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
```

## Build

After setting up the environment, run the following commands to install DynaFlow and the targeted frameworks.

```bash
git clone https://github.com/uw-syfi/DynaFlow.git
cd DynaFlow
git switch ae
cd examples/ae
```

The following operations should be executed from the `examples/ae` directory.

```bash
# Clone and build vLLM
git clone https://github.com/vllm-project/vllm.git
cd vllm
git reset --hard 2dda3e35d054235b0c2170df359b42ec25b4fe2c
git apply ../patch/vllm.patch
uv venv && source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e .
# Build DynaFlow
uv pip install -e "../../../"
# Build Tokenweave kernels
cd ../scheduler/vllm/tokenweave-kernels
mkdir -p build
cd build
cmake .. -DVLLM_PYTHON_EXECUTABLE=$(which python)
make
```

```bash
# Clone and build SGLang
git clone https://github.com/sgl-project/sglang.git
cd sglang
git reset --hard d6fee73d1f593bd6754cd2550775fd2e54aeae60
git apply ../patch/sglang.patch
uv venv && source .venv/bin/activate
uv pip install -e "python"
# Build DynaFlow
uv pip install -e "../../../"
```

After installing the frameworks, you can download the required models in any of the created virtual environments.

```bash
source vllm/.venv/bin/activate
huggingface-cli login # Log in with your HuggingFace account to access the models
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct
huggingface-cli download meta-llama/Meta-Llama-3-70B-Instruct
huggingface-cli download Qwen/Qwen2.5-72B-Instruct
huggingface-cli download deepseek-ai/DeepSeek-V2-Lite
```

## Benchmark

After installing the frameworks, you can run the benchmarks for each target. Each target has a corresponding Makefile rule that runs the benchmarks for all models and datasets.

```bash

make vllm-nanoflow # Figure 9
make sglang-nanoflow # Figure 10
make vllm-dbo # Figure 12
make vllm-tokenweave # Figure 13
```

Results are written under `examples/results/` with the following layout:

```
results/
├── vllm_nanoflow/<model>/
│   ├── <testcase>.json       # Per-iteration throughput metrics
│   └── log/<testcase>.log    # Full vllm bench stdout/stderr
├── vllm_tokenweave/<model>/
├── vllm_ep_dbo/<model>/
└── sglang_nanoflow/<model>/
```

## Plotting

Coming soon.

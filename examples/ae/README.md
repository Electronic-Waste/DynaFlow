# DynaFlow Artifact Evaluation

This directory contains the scripts and configuration needed to reproduce the
throughput benchmarks for **DynaFlow**.

## Build

Both build steps clone the framework, pin to the evaluated commit, apply the
DynaFlow patch, and install into an isolated virtual environment.
Each requires ~20–30 GB of disk space and ~15–30 minutes on a fast connection.

### vLLM

```bash
cd examples/ae
make vllm-build
```

This clones [vllm-project/vllm](https://github.com/vllm-project/vllm),
resets to commit `2dda3e3`, applies `patch/vllm.patch`, and installs with
`VLLM_USE_PRECOMPILED=1` (uses pre-built CUDA kernels).

### SGLang

```bash
cd examples/ae
make sglang-build
```

This clones [sgl-project/sglang](https://github.com/sgl-project/sglang),
resets to commit `d6fee73`, applies `patch/sglang.patch`, and installs
`sglang/python`.

---

## Running Evaluations

Each target loops over all default model/parallelism configurations and writes
one JSON result file plus one log file per iteration to `../results/`.

### Figure 9

```bash
make vllm-nanoflow
```

Default runs: Llama-3-8B (TP=2), Llama-3-70B (TP=8), Qwen2.5-72B (TP=8).

### Figure 10

```bash
make sglang-nanoflow
```

Default runs: Llama-3-8B (TP=2), Llama-3-70B (TP=8), Qwen2.5-72B (TP=8).

### Figure 12

```bash
make vllm-dbo
```

Default runs: DeepSeek-V2-Lite (DP=2)

### Figure 13

```bash
make vllm-tokenweave
```

Default runs: same model/TP combinations as NanoFlow.

## Results

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

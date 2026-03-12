SHELL := /bin/bash

# ── Configurable run sets ─────────────────────────────────────────────────────
# Each entry is MODEL:PARALLELISM_SIZE.  Override on the command line, e.g.:
#   make vllm-nanoflow VLLM_TP_RUNS="meta-llama/Meta-Llama-3-8B-Instruct:2"

VLLM_TP_RUNS ?= \
	meta-llama/Meta-Llama-3-8B-Instruct:2 \
	meta-llama/Meta-Llama-3-70B-Instruct:8 \
	Qwen/Qwen2.5-72B-Instruct:8

VLLM_EP_RUNS ?= \
	deepseek-ai/DeepSeek-V2-Lite:2

SGLANG_TP_RUNS ?= \
	meta-llama/Meta-Llama-3-8B-Instruct:2 \
	meta-llama/Meta-Llama-3-70B-Instruct:8 \
	Qwen/Qwen2.5-72B-Instruct:8

.PHONY: vllm-build vllm-nanoflow vllm-tokenweave vllm-dbo \
        sglang-build sglang-nanoflow

# ── vLLM ──────────────────────────────────────────────────────────────────────
vllm/.build_done:
	git clone https://github.com/vllm-project/vllm.git vllm && \
	cd vllm && \
	git reset --hard 2dda3e35d054235b0c2170df359b42ec25b4fe2c && \
	git apply ../patch/vllm.patch && \
	uv venv .venv && \
	source .venv/bin/activate && \
	VLLM_USE_PRECOMPILED=1 uv pip install -e . && \
	touch .build_done

vllm-build: vllm/.build_done

vllm-nanoflow: vllm/.build_done
	cd vllm && source .venv/bin/activate && \
	for run in $(VLLM_TP_RUNS); do \
	    model=$$(echo $$run | cut -d: -f1); \
	    tp=$$(echo $$run | cut -d: -f2); \
	    python ../vllm-bench-tp.py --model $$model --tp-size $$tp --strategy nanoflow --mode fixed; \
	done

vllm-tokenweave: vllm/.build_done
	cd vllm && source .venv/bin/activate && \
	for run in $(VLLM_TP_RUNS); do \
	    model=$$(echo $$run | cut -d: -f1); \
	    tp=$$(echo $$run | cut -d: -f2); \
	    python ../vllm-bench-tp.py --model $$model --tp-size $$tp --strategy tokenweave --mode fixed; \
	done

vllm-dbo: vllm/.build_done
	cd vllm && source .venv/bin/activate && \
	for run in $(VLLM_EP_RUNS); do \
	    model=$$(echo $$run | cut -d: -f1); \
	    dp=$$(echo $$run | cut -d: -f2); \
	    python ../vllm-bench-ep.py --model $$model --dp-size $$dp --strategy dbo --mode fixed; \
	done

# ── SGLang ────────────────────────────────────────────────────────────────────
sglang/.build_done:
	git clone https://github.com/sgl-project/sglang.git sglang && \
	cd sglang && \
	git reset --hard d6fee73d1f593bd6754cd2550775fd2e54aeae60 && \
	git apply ../patch/sglang.patch && \
	uv venv .venv && \
	source .venv/bin/activate && \
	uv pip install -e "python" && \
	touch .build_done

sglang-build: sglang/.build_done

sglang-nanoflow: sglang/.build_done
	cd sglang && source .venv/bin/activate && \
	for run in $(SGLANG_TP_RUNS); do \
	    model=$$(echo $$run | cut -d: -f1); \
	    tp=$$(echo $$run | cut -d: -f2); \
	    python ../sglang-bench-tp.py --model $$model --tp-size $$tp --strategy nanoflow --mode fixed; \
	done

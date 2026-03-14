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

.PHONY: vllm-nanoflow vllm-tokenweave vllm-dbo sglang-nanoflow

# ── vLLM ──────────────────────────────────────────────────────────────────────
vllm-nanoflow: vllm/.venv/bin/activate
	cd vllm && source .venv/bin/activate && \
	for run in $(VLLM_TP_RUNS); do \
	    model=$$(echo $$run | cut -d: -f1); \
	    tp=$$(echo $$run | cut -d: -f2); \
	    python ../vllm-bench-tp.py --model $$model --tp-size $$tp --strategy nanoflow --mode fixed; \
	done

vllm-tokenweave: vllm/.venv/bin/activate
	cd vllm && source .venv/bin/activate && \
	for run in $(VLLM_TP_RUNS); do \
	    model=$$(echo $$run | cut -d: -f1); \
	    tp=$$(echo $$run | cut -d: -f2); \
	    python ../vllm-bench-tp.py --model $$model --tp-size $$tp --strategy tokenweave --mode fixed; \
	done

vllm-dbo: vllm/.venv/bin/activate
	cd vllm && source .venv/bin/activate && \
	for run in $(VLLM_EP_RUNS); do \
	    model=$$(echo $$run | cut -d: -f1); \
	    dp=$$(echo $$run | cut -d: -f2); \
	    python ../vllm-bench-ep.py --model $$model --dp-size $$dp --strategy dbo --mode fixed; \
	done

# ── SGLang ────────────────────────────────────────────────────────────────────
sglang-nanoflow: sglang/.venv/bin/activate
	cd sglang && source .venv/bin/activate && \
	for run in $(SGLANG_TP_RUNS); do \
	    model=$$(echo $$run | cut -d: -f1); \
	    tp=$$(echo $$run | cut -d: -f2); \
	    python ../sglang-bench-tp.py --model $$model --tp-size $$tp --strategy nanoflow --mode fixed; \
	done

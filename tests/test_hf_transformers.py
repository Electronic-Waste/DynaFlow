import argparse
import os
from collections.abc import Callable
from typing import Any

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from schedflow.config import CUDAGraphConfig, InductorConfig, SchedFlowConfig
from schedflow.example.hf.nanoflow import (
    NanoFlowScheduler,
    NanoFlowSchedulerConfig,
)
from schedflow.interface import SplitConfig
from schedflow.manager import SchedFlowManager

_scheduler = None
_manager = SchedFlowManager()


def schedflow_backend(
    gm: torch.fx.GraphModule, example_inputs: tuple[Any, ...]
) -> Callable:
    """Create a torch.compile backend that runs the model via SchedFlow.

    This backend assumes fullgraph=True from torch.compile. It constructs a
    minimal SchedFlow configuration and a NanoFlow scheduler, initializes the
    SchedFlow manager with the compiled FX graph and example inputs, and
    returns the callable that executes with programmable scheduling.
    """
    if not any(isinstance(i, torch.SymInt) for i in example_inputs):
        return gm

    global _scheduler
    assert _scheduler is None
    _scheduler = NanoFlowScheduler(
        NanoFlowSchedulerConfig(
            min_nano_split_tokens=64,
            max_num_nano_batches=2,
            cudagraph_capture_sizes=[64, 128, 256],
        )
    )
    inductor_cfg = InductorConfig(
        enabled=True,
        compile_sizes=set(),
    )
    cudagraph_cfg = CUDAGraphConfig(
        enabled=False,
        capture_sizes=[64, 128, 256],
    )
    schedflow_cfg = SchedFlowConfig(
        max_num_nano_batches=2,
        min_nano_split_tokens=1,
        inductor_config=inductor_cfg,
        cudagraph_config=cudagraph_cfg,
    )

    global _manager
    _manager.initialize(
        graph_module=gm,
        config=schedflow_cfg,
        scheduler=_scheduler,
        example_inputs=list(example_inputs),
    )
    return _manager.get_callable()

def _sync_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Trace an FX graph for a Hugging Face CausalLM model and split it "
            "into subgraphs using SchedFlow utilities."
        )
    )
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hf", "schedflow"],
        default="schedflow",
        help="Run plain HuggingFace model ('hf') or with SchedFlow backend ('schedflow').",
    )

    args = parser.parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and torch.cuda.is_available():
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        tp_plan="auto",
    ).eval()

    seq_len = max(args.seq_len, 2)
    token_id = tokenizer.eos_token_id or 1
    input_ids = torch.full(
        (args.batch_size, seq_len), token_id, dtype=torch.long
    )
    inputs = {"input_ids": input_ids.to(device)}
    example_inputs = {"input_ids": input_ids[0].unsqueeze(0).to(device)}


    print(f"Running in {args.mode} mode")
    
    if args.mode == "schedflow":
        compiled_model = torch.compile(
            model, backend=schedflow_backend, fullgraph=True
        )
    else:
        compiled_model = torch.compile(
            model, fullgraph=True
        )

    split_config = SplitConfig(
        num_nano_batches=2,
        batch_sizes=[args.batch_size // 2, args.batch_size // 2],
        batch_indices=[0, args.batch_size // 2, args.batch_size],
        num_tokens=[args.batch_size // 2, args.batch_size // 2],
        num_tokens_padded=[args.batch_size // 2, args.batch_size // 2],
        split_indices=[0, args.batch_size // 2, args.batch_size],
        is_dryrun=False,
        use_cudagraph=False,
    )
    _manager.override_split_config(split_config)

    
    warmup_steps = 10
    with torch.inference_mode():
        compiled_model(**example_inputs)
        for _ in range(warmup_steps):
            start_time = time.perf_counter()
            outputs = compiled_model(**inputs)
            _manager.override_split_config(split_config)
            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time) * 1000.0
            print(f"[Warmup] latency: {elapsed_ms:.2f} ms")
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            print(f"[Warmup] logits shape: {tuple(logits.shape)}")
        print("Warmup completed")
    
    latency_list = []
    with torch.inference_mode():
        for t in range(args.trials):
            start = time.perf_counter()
            outputs = compiled_model(**inputs)
            _manager.override_split_config(split_config)
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000.0
            latency_list.append(elapsed_ms)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs
            print(f"[Trial {t+1}/{args.trials}] latency: {elapsed_ms:.2f} ms")
            print(f"[Trial {t+1}/{args.trials}] logits shape: {tuple(logits.shape)}")

    avg = torch.mean(torch.tensor(latency_list))
    std = torch.std(torch.tensor(latency_list), unbiased=False)  # population std (divide by N)
    print(f"[Summary] trials={args.trials}, avg={avg:.2f} ms, std={std:.2f} ms")

    print("Execution completed")
    

    # with torch.inference_mode():
    #     compiled_model(**example_inputs)
    #     start_time = time.perf_counter()
    #     outputs = compiled_model(**inputs)
    #     end_time = time.perf_counter()
    #     elapsed_ms = (end_time - start_time) * 1000.0
    #     print(f"[Run] latency: {elapsed_ms:.2f} ms")
    #     logits = outputs.logits if hasattr(outputs, "logits") else outputs
    #     print(f"[Run] logits shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()

import argparse
import os
from collections.abc import Callable
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from schedflow.config import CUDAGraphConfig, InductorConfig, SchedFlowConfig
from schedflow.example.nanoflow import (
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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Trace an FX graph for a Hugging Face CausalLM model and split it "
            "into subgraphs using SchedFlow utilities."
        )
    )
    parser.add_argument("--model", type=str, default="/data/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=128)

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

    compiled_model = torch.compile(
        model, backend=schedflow_backend, fullgraph=True
    )


    _manager.override_split_config(SplitConfig(
        num_nano_batches=2,
        batch_sizes=[args.batch_size // 2, args.batch_size // 2],
        batch_indices=[0, args.batch_size // 2, args.batch_size],
        num_tokens=[args.batch_size // 2, args.batch_size // 2],
        num_tokens_padded=[args.batch_size // 2, args.batch_size // 2],
        split_indices=[0, args.batch_size // 2, args.batch_size],
        is_dryrun=False,
        use_cudagraph=False,
    ))

    with torch.inference_mode():
        compiled_model(**example_inputs)
        outputs = compiled_model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        print(f"[Run] logits shape: {tuple(logits.shape)}")

    print(
        "[OK] Executed with SchedFlow backend under torch.compile (fullgraph)"
    )


if __name__ == "__main__":
    main()

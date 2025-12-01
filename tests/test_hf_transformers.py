import argparse
import os
from collections.abc import Callable
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

from dynaflow.config import CUDAGraphConfig, DynaFlowConfig, InductorConfig
from dynaflow.example.hf.nanoflow import (
    NanoFlowScheduler,
    NanoFlowSchedulerConfig,
)
from dynaflow.interface import SplitConfig
from dynaflow.manager import DynaFlowManager

_scheduler = None
_manager = DynaFlowManager()


def dynaflow_backend(
    gm: torch.fx.GraphModule, example_inputs: tuple[Any, ...]
) -> Callable:
    """Create a torch.compile backend that runs the model via DynaFlow.

    This backend assumes fullgraph=True from torch.compile. It constructs a
    minimal DynaFlow configuration and a NanoFlow scheduler, initializes the
    DynaFlow manager with the compiled FX graph and example inputs, and
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
    dynaflow_cfg = DynaFlowConfig(
        max_num_nano_batches=2,
        min_nano_split_tokens=1,
        inductor_config=inductor_cfg,
        cudagraph_config=cudagraph_cfg,
    )

    global _manager
    _manager.initialize(
        graph_module=gm,
        config=dynaflow_cfg,
        scheduler=_scheduler,
        example_inputs=list(example_inputs),
    )
    return _manager.get_callable()


def run_inference(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    split_config: SplitConfig | None = None,
    warmup_steps: int = 10,
    trials: int = 10,
) -> tuple[float, float]:
    if split_config is not None:
        torch._dynamo.mark_dynamic(inputs["input_ids"], 0)
        compiled_model = torch.compile(model, backend=dynaflow_backend, fullgraph=True)
        _manager.override_split_config(split_config)
    else:
        compiled_model = torch.compile(model, fullgraph=True)
    latency_list: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup_steps):
            if split_config is not None:
                _manager.override_split_config(split_config)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            compiled_model(**inputs)
            end_event.record()
            end_event.synchronize()
            latency = start_event.elapsed_time(end_event)
            print(f"Warmup latency: {latency:.2f} ms")

        for _ in range(trials):
            if split_config is not None:
                _manager.override_split_config(split_config)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            outputs = compiled_model(**inputs)
            end_event.record()
            end_event.synchronize()
            latency = start_event.elapsed_time(end_event)
            print(
                f"Latency: {latency:.2f} ms; outputs shape: {outputs.logits.shape if hasattr(outputs, 'logits') else outputs.shape}"
            )
            latency_list.append(latency)

    avg = float(torch.mean(torch.tensor(latency_list)).item())
    std = float(torch.std(torch.tensor(latency_list), unbiased=False).item())
    return avg, std


def run_training(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    split_config: SplitConfig | None = None,
    warmup_steps: int = 10,
    trials: int = 10,
) -> tuple[float, float]:
    def backward_func(loss: torch.Tensor):
        return loss.backward()
    if split_config is not None:
        torch._dynamo.mark_dynamic(inputs["input_ids"], 0)
        # torch._dynamo.config.compiled_autograd = True
        torch._dynamo.config.recompile_limit = 1000
        compiled_model = torch.compile(model, backend=dynaflow_backend, fullgraph=True)
        # compiled_backward = torch.compile(backward_func, backend=dynaflow_backend)
        _manager.override_split_config(split_config)
    else:
        compiled_model = torch.compile(model, fullgraph=True)
        # compiled_backward = torch.compile(backward_func, fullgraph=True)
    compiled_backward = backward_func
    latency_list: list[float] = []
    for _ in range(warmup_steps):
        if split_config is not None:
            _manager.override_split_config(split_config)
        outputs = compiled_model(**inputs)
        loss = outputs.logits.mean()
        compiled_backward(loss)
        model.zero_grad()
    for _ in range(trials):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        if split_config is not None:
            _manager.override_split_config(split_config)
        outputs = compiled_model(**inputs)
        loss = outputs.logits.mean()
        compiled_backward(loss)
        model.zero_grad()
        end_event.record()
        end_event.synchronize()
        latency = start_event.elapsed_time(end_event)
        print(f"Latency: {latency:.2f} ms")
        latency_list.append(latency)
    avg = float(torch.mean(torch.tensor(latency_list)).item())
    std = float(torch.std(torch.tensor(latency_list), unbiased=False).item())
    return avg, std


def run_fwd_bwd_test(model: torch.nn.Module, inputs: dict[str, torch.Tensor]):
    torch._dynamo.config.compiled_autograd = True
    torch._dynamo.config.recompile_limit = 1000
    def forward_backward_func(model: torch.nn.Module, inputs: dict[str, torch.Tensor]):
        outputs = model(**inputs)
        loss = outputs.logits.mean()
        print(loss)
        loss.backward()
        return outputs
    compiled_forward_backward = torch.compile(forward_backward_func, backend=dynaflow_backend)
    compiled_forward_backward(model, inputs)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Trace an FX graph for a Hugging Face CausalLM model and split it "
            "into subgraphs using DynaFlow utilities."
        )
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hf", "dynaflow"],
        default="dynaflow",
        help="Run plain HuggingFace model ('hf') or with DynaFlow backend ('dynaflow').",
    )

    args = parser.parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and torch.cuda.is_available():
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        tp_plan="auto",
    ).eval()
    # model.to(device)

    seq_len = max(args.seq_len, 2)
    token_id = tokenizer.eos_token_id or 1
    input_ids = torch.full((args.batch_size, seq_len), token_id, dtype=torch.long)
    inputs = {"input_ids": input_ids.to(device)}

    print(f"Running in {args.mode} mode")

    if args.mode == "dynaflow":
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
    else:
        split_config = None

    # avg, std = run_inference(
    #   model, inputs, split_config, warmup_steps=10, trials=10
    # )
    avg, std = run_training(
        model, inputs, split_config, warmup_steps=10, trials=10
    )
    print(f"Average latency: {avg:.2f} ms, Standard deviation: {std:.2f} ms")
    # run_fwd_bwd_test(model, inputs)

if __name__ == "__main__":
    main()

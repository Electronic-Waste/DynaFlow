import argparse
import importlib.util
import os
from typing import Tuple

import torch
import torch.library

from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer import TransformerConfig
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.optimizer.optimizer import MegatronOptimizer

from dynaflow.config import CUDAGraphConfig, DynaFlowConfig, InductorConfig
from dynaflow.interface import SplitConfig
from dynaflow.manager import DynaFlowManager

os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"


@torch.library.custom_op("megatron::bwd_allreduce_marker", mutates_args=())
def _bwd_allreduce_marker(input_: torch.Tensor) -> torch.Tensor:
    """Identity in forward; all-reduce grad_input across TP ranks in backward."""
    return input_.clone()


@_bwd_allreduce_marker.register_fake
def _(input_: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(input_)


def _bwd_allreduce_marker_setup_context(ctx, inputs, output):
    pass  # group is looked up from global parallel_state at backward time


def _bwd_allreduce_marker_backward(ctx, grad_output: torch.Tensor):
    from torch.distributed._functional_collectives import all_reduce as _func_all_reduce
    from megatron.core.parallel_state import get_tensor_model_parallel_group
    group = get_tensor_model_parallel_group()
    if torch.distributed.get_world_size(group) <= 1:
        return grad_output
    return _func_all_reduce(grad_output.contiguous(), reduceOp="sum", group=group)


_bwd_allreduce_marker.register_autograd(
    _bwd_allreduce_marker_backward,
    setup_context=_bwd_allreduce_marker_setup_context,
)


def make_dynaflow_backend(manager: DynaFlowManager, require_sym_int: bool = True):
    """Return a torch.compile backend that routes graphs to the given manager."""
    sched_file = os.path.join(
        os.path.dirname(__file__), '..', 'scheduler', 'megatron', 'nanoflow.py'
    )

    def backend(gm: torch.fx.GraphModule, example_inputs) -> object:
        if require_sym_int and not any(isinstance(i, torch.SymInt) for i in example_inputs):
            return gm
        dynaflow_cfg = DynaFlowConfig(
            scheduler_path=sched_file + ':NanoFlowMegatronScheduler',
            max_num_splits=8,
            inductor_config=InductorConfig(enabled=False, compile_sizes=set()),
            cudagraph_config=CUDAGraphConfig(enabled=False, capture_sizes=[]),
            additional_config={"min_nano_split_tokens": 8, "num_nano_batches": num_nano_batches},
        )
        file_path, cls_name = dynaflow_cfg.scheduler_path.split(':', 1)
        spec = importlib.util.spec_from_file_location('_megatron_scheduler', file_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        scheduler = getattr(module, cls_name)(dynaflow_cfg)
        manager.initialize(
            graph_module=gm,
            config=dynaflow_cfg,
            scheduler=scheduler,
            example_inputs=list(example_inputs),
        )
        return manager.get_callable()

    return backend


_fwd_manager = DynaFlowManager()
_bwd_manager = DynaFlowManager()
num_nano_batches = 2  # overridden by --nano-batches CLI arg
dynaflow_backend = make_dynaflow_backend(_fwd_manager)


def setup_distributed() -> None:
    """Initialize distributed training."""
    torch.distributed.init_process_group(
        backend='nccl',
        rank=int(os.environ.get('RANK', 0)),
        world_size=int(os.environ.get('WORLD_SIZE', 1)),
    )
    torch.cuda.set_device(torch.distributed.get_rank() % torch.cuda.device_count())

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=int(os.environ.get('WORLD_SIZE', 1)),
        pipeline_model_parallel_size=1,
    )

    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    rng_tracker = get_cuda_rng_tracker()
    rng_tracker.add('model-parallel-rng', 1234)


def _patch_megatron_for_inference() -> None:
    """Patch forward communications to use functional collective API.

    Replaces autograd-function-wrapped collectives with direct functional API
    calls so TorchDynamo traces them as flat _c10d_functional nodes, making
    them visible to DynaFlow's Op matcher for graph splitting.

    Must be called after parallel_state is initialized.
    """
    from torch.distributed._functional_collectives import (
        all_reduce as _func_all_reduce,
        all_gather_tensor as _func_all_gather,
    )
    import megatron.core.tensor_parallel.mappings as _mappings
    import megatron.core.tensor_parallel.layers as _layers
    from megatron.core.parallel_state import (
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_world_size,
    )

    def _inference_reduce(input_: torch.Tensor) -> torch.Tensor:
        group = get_tensor_model_parallel_group()
        if torch.distributed.get_world_size(group) == 1:
            return input_
        return _func_all_reduce(input_, reduceOp="sum", group=group)

    def _inference_gather(input_: torch.Tensor) -> torch.Tensor:
        world_size = get_tensor_model_parallel_world_size()
        if world_size == 1:
            return input_
        group = get_tensor_model_parallel_group()
        # all_gather_tensor with gather_dim=-1 replicates _gather_along_last_dim:
        # internally calls all_gather_into_tensor (flat node), then chunk+cat on last dim.
        return _func_all_gather(input_.contiguous(), gather_dim=-1, group=group)

    _mappings.reduce_from_tensor_model_parallel_region = _inference_reduce
    _layers.reduce_from_tensor_model_parallel_region = _inference_reduce
    _mappings.gather_from_tensor_model_parallel_region = _inference_gather
    _layers.gather_from_tensor_model_parallel_region = _inference_gather


def _patch_megatron_for_training() -> None:
    """Patch forward + backward communications to use functional collective API.

    Extends _patch_megatron_for_inference() by also patching
    linear_with_grad_accumulation_and_async_allreduce to enable backward overlap.

    The trick: wrap the linear input with torch.ops.megatron.bwd_allreduce_marker
    (identity forward, functional all-reduce backward) and disable the internal
    allreduce_dgrad. DynaFlow splits the FX graph at this node and runs it on
    comm_stream. PyTorch's autograd then runs the backward all-reduce on comm_stream
    too (CUDA stream inheritance), enabling backward compute/comm overlap without
    compiled_autograd.
    """
    _patch_megatron_for_inference()

    import megatron.core.tensor_parallel.layers as _layers

    _orig_linear_fn = _layers.linear_with_grad_accumulation_and_async_allreduce

    def _patched_linear(
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer=None,
        wgrad_deferral_limit=0,
        **kwargs,
    ):
        if allreduce_dgrad:
            # Replace the internal dgrad all-reduce with a custom op that is
            # identity in the forward but issues the all-reduce in the backward.
            # DynaFlow splits the FX graph at this node and runs it on comm_stream,
            # so the backward all-reduce inherits comm_stream via CUDA stream
            # inheritance — enabling backward compute/comm overlap automatically.
            input = torch.ops.megatron.bwd_allreduce_marker(input)
            allreduce_dgrad = False
        return _orig_linear_fn(
            input,
            weight,
            bias,
            gradient_accumulation_fusion,
            allreduce_dgrad,
            sequence_parallel,
            grad_output_buffer,
            wgrad_deferral_limit,
        )

    _patched_linear.warned = _orig_linear_fn.warned  # type: ignore[attr-defined]
    _layers.linear_with_grad_accumulation_and_async_allreduce = _patched_linear


MODEL_CONFIGS = {
    "gpt-small": {
        "num_layers": 4,
        "hidden_size": 4096,
        "ffn_hidden_size": 11008,
        "num_attention_heads": 32,
        "num_query_groups": 32,
        "vocab_size": 32768,
        "max_sequence_length": 4096,
    },
    "llama-3-8b": {
        "num_layers": 32,
        "hidden_size": 4096,
        "ffn_hidden_size": 14336,
        "num_attention_heads": 32,
        "num_query_groups": 8,
        "vocab_size": 128256,
        "max_sequence_length": 8192,
    },
}


def create_model(model_name: str = "gpt-small", num_layers_override: int | None = None) -> GPTModel:
    """Create GPT model using megatron-core."""
    mcfg = MODEL_CONFIGS[model_name]
    num_layers = num_layers_override if num_layers_override is not None else mcfg["num_layers"]
    config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=mcfg["hidden_size"],
        ffn_hidden_size=mcfg["ffn_hidden_size"],
        num_attention_heads=mcfg["num_attention_heads"],
        num_query_groups=mcfg["num_query_groups"],
        layernorm_epsilon=1e-5,
        add_bias_linear=False,
        add_qkv_bias=False,
    )

    layer_spec = get_gpt_layer_local_spec()

    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=mcfg["vocab_size"],
        max_sequence_length=mcfg["max_sequence_length"],
        pre_process=True,
        post_process=True,
        parallel_output=False,
    ).cuda()

    from megatron.core.distributed import DistributedDataParallelConfig
    model.ddp_config = DistributedDataParallelConfig()

    return model


def create_optimizer(model: GPTModel) -> MegatronOptimizer:
    """Create optimizer using megatron-core."""
    optimizer_config = OptimizerConfig(
        optimizer='adam',
        lr=1e-3,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
    )

    optimizer = get_megatron_optimizer(
        config=optimizer_config,
        model_chunks=[model],
    )

    return optimizer


def create_dummy_batch(batch_size: int = 64, seq_length: int = 512, vocab_size: int = 32768) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a dummy batch of data for training."""

    tokens = torch.randint(0, vocab_size, (batch_size, seq_length), dtype=torch.long, device='cuda')

    position_ids = torch.arange(seq_length, dtype=torch.long, device='cuda').unsqueeze(0).expand(batch_size, -1)

    attention_mask = torch.tril(torch.ones((seq_length, seq_length), device='cuda', dtype=torch.bool))
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

    labels = torch.cat([tokens[:, 1:], torch.zeros(batch_size, 1, dtype=torch.long, device='cuda')], dim=1)

    torch._dynamo.mark_dynamic(tokens, 0)
    torch._dynamo.mark_dynamic(position_ids, 0)
    torch._dynamo.mark_dynamic(attention_mask, 0)
    torch._dynamo.mark_dynamic(labels, 0)

    return tokens, position_ids, attention_mask, labels


bwd_dynaflow_backend = make_dynaflow_backend(_bwd_manager, require_sym_int=True)


def run_inference(
    model: GPTModel,
    tokens: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    split_config: SplitConfig | None = None,
    warmup_steps: int = 5,
    trials: int = 10,
) -> tuple[float, float]:
    if split_config is not None:
        compiled_model = torch.compile(model, backend=dynaflow_backend, fullgraph=True)
        _fwd_manager.override_split_config(split_config)
    else:
        compiled_model = torch.compile(model, backend="eager", fullgraph=True)

    latency_list: list[float] = []
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for _ in range(warmup_steps):
            if split_config is not None:
                _fwd_manager.override_split_config(split_config)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            compiled_model(tokens, position_ids, attention_mask)
            e.record()
            e.synchronize()
            print(f"Warmup: {s.elapsed_time(e):.2f} ms")

        for _ in range(trials):
            if split_config is not None:
                _fwd_manager.override_split_config(split_config)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            compiled_model(tokens, position_ids, attention_mask)
            e.record()
            e.synchronize()
            lat = s.elapsed_time(e)
            latency_list.append(lat)
            print(f"Latency: {lat:.2f} ms")

    avg = float(torch.mean(torch.tensor(latency_list)).item())
    std = float(torch.std(torch.tensor(latency_list), unbiased=False).item())
    return avg, std


def run_training(
    model: GPTModel,
    optimizer: MegatronOptimizer,
    tokens: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    split_config: SplitConfig | None = None,
    warmup_steps: int = 5,
    trials: int = 10,
) -> tuple[float, float]:
    if split_config is not None:
        compiled_model = torch.compile(model, backend=dynaflow_backend, fullgraph=True)
        _fwd_manager.override_split_config(split_config)
    else:
        compiled_model = torch.compile(model, backend="eager", fullgraph=True)

    def step() -> float:
        if split_config is not None:
            _fwd_manager.override_split_config(split_config)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = compiled_model(tokens, position_ids, attention_mask)
            logits = output[0] if isinstance(output, tuple) else output
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=0,
            )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    latency_list: list[float] = []
    for _ in range(warmup_steps):
        step()
        print("Warmup done")

    for _ in range(trials):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        step()
        e.record()
        e.synchronize()
        lat = s.elapsed_time(e)
        latency_list.append(lat)
        print(f"Latency: {lat:.2f} ms")

    avg = float(torch.mean(torch.tensor(latency_list)).item())
    std = float(torch.std(torch.tensor(latency_list), unbiased=False).item())
    return avg, std


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Megatron GPT with optional DynaFlow overlap.")
    parser.add_argument("--mode", choices=["megatron", "dynaflow"], default="dynaflow",
                        help="'megatron': plain torch.compile; 'dynaflow': DynaFlow nano-batch overlap.")
    parser.add_argument("--task", choices=["inference", "training"], default="inference")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--nano-batches", type=int, default=2,
                        help="Number of nano-batches for DynaFlow splitting.")
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default="gpt-small",
                        help="Model configuration to use.")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Override number of transformer layers.")
    args = parser.parse_args()

    setup_distributed()

    if args.mode == "dynaflow":
        if args.task == "training":
            _patch_megatron_for_training()
        else:
            _patch_megatron_for_inference()

    model = create_model(args.model, num_layers_override=args.num_layers)
    optimizer = create_optimizer(model)
    mcfg = MODEL_CONFIGS[args.model]
    batch = create_dummy_batch(batch_size=args.batch_size, seq_length=args.seq_len, vocab_size=mcfg["vocab_size"])
    tokens, position_ids, attention_mask, labels = batch

    if args.mode == "dynaflow":
        global num_nano_batches
        num_nano_batches = args.nano_batches
        batch_size = tokens.shape[0]
        n = num_nano_batches
        chunk = batch_size // n
        sizes = [chunk] * n
        sizes[-1] = batch_size - chunk * (n - 1)
        indices = [chunk * i for i in range(n)] + [batch_size]
        split_config = SplitConfig(
            num_nano_batches=n,
            batch_sizes=sizes,
            batch_indices=indices,
            num_tokens=sizes,
            num_tokens_padded=sizes,
            split_indices=indices,
            is_dryrun=False,
            use_cudagraph=False,
        )
    else:
        split_config = None

    if args.task == "training":
        avg, std = run_training(
            model, optimizer, tokens, position_ids, attention_mask, labels,
            split_config=split_config,
            warmup_steps=args.warmup_steps,
            trials=args.trials,
        )
    else:
        avg, std = run_inference(
            model, tokens, position_ids, attention_mask,
            split_config=split_config,
            warmup_steps=args.warmup_steps,
            trials=args.trials,
        )

    rank = torch.distributed.get_rank()
    if rank == 0:
        print(f"Average latency: {avg:.2f} ms ± {std:.2f} ms")
        print(f"RESULT|{args.model}|{args.mode}|{args.task}|{args.batch_size}|{args.nano_batches}|{avg:.2f}|{std:.2f}")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

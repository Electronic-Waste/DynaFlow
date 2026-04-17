"""Benchmark Megatron GPT with MoE: eager baseline vs Comet via DynaFlow.

Both modes wrap MoELayer.forward in an opaque custom op (``moe::moe_forward``)
and use ``torch.compile``.  The only difference is the backend:

- **megatron**: ``torch.compile(backend="eager")`` — the custom op runs the
  real Megatron MoE forward (router + all-to-all + sequential expert GEMMs).
- **dynaflow**: ``torch.compile(backend=dynaflow_backend)`` — DynaFlow splits
  the FX graph around the custom op and the scheduler substitutes Comet's fused
  AGScatter + GatherRS kernels at schedule time.

Usage examples:
  # Baseline Megatron (eager, vanilla all-to-all + sequential expert GEMMs):
  torchrun --nproc_per_node=8 bench_megatron_comet.py --mode megatron --task inference

  # Comet via DynaFlow (fused AGScatter + GatherRS):
  torchrun --nproc_per_node=8 bench_megatron_comet.py --mode dynaflow --task inference

  # With TP+EP:
  torchrun --nproc_per_node=8 bench_megatron_comet.py --mode dynaflow --tp-size 2 --ep-size 4
"""

import argparse
import importlib.util
import os

import torch
import torch.library
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.transformer import TransformerConfig

from dynaflow.config import CUDAGraphConfig, DynaFlowConfig, InductorConfig
from dynaflow.interface import SplitConfig
from dynaflow.manager import DynaFlowManager

os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

# Dynamo settings for Megatron MoE traceability
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.capture_scalar_outputs = True


# ── MoE custom op wrapper ─────────────────────────────────────────────
# Wraps the real Megatron MoE forward as an opaque node for torch.compile.
# In eager backend mode the custom op body runs the real MoE.
# In DynaFlow mode the scheduler intercepts and substitutes Comet.

_moe_layer_ref = None  # Set by _patch_moe_as_custom_op(model) before compilation.
_orig_moe_forward = None


@torch.library.custom_op("moe::moe_forward", mutates_args=())
def _moe_forward(hidden_states: torch.Tensor) -> torch.Tensor:
    """Opaque wrapper around the real Megatron MoE forward."""
    output, _ = _orig_moe_forward(_moe_layer_ref, hidden_states)
    return output


@_moe_forward.register_fake
def _(hidden_states: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(hidden_states)


def _moe_forward_setup_context(ctx, inputs, output):
    hs = inputs[0]
    ctx.hs_shape = hs.shape
    ctx.hs_dtype = hs.dtype
    ctx.hs_device = hs.device


def _moe_forward_backward(ctx, grad_output):
    # Benchmark-only backward: return zeros matching input shape.
    return (torch.zeros(ctx.hs_shape, dtype=ctx.hs_dtype, device=ctx.hs_device),)


_moe_forward.register_autograd(
    _moe_forward_backward,
    setup_context=_moe_forward_setup_context,
)


def _patch_moe_as_custom_op(model: torch.nn.Module) -> None:
    """Replace MoELayer.forward with a call through the ``moe::moe_forward`` custom op.

    Must be called after model creation so the MoE layer reference can be stored.
    The compiled FX graph calls the custom op directly (bypassing the patched forward),
    so the layer reference must be set upfront — not via a global side-effect during tracing.
    """
    from megatron.core.transformer.moe.moe_layer import MoELayer

    global _orig_moe_forward, _moe_layer_ref
    _orig_moe_forward = MoELayer.forward

    # Store the MoE layer reference for the custom op body.
    for module in model.modules():
        if isinstance(module, MoELayer):
            _moe_layer_ref = module
            break
    assert _moe_layer_ref is not None, "No MoELayer found in model"

    def _patched_forward(self, hidden_states):
        return torch.ops.moe.moe_forward(hidden_states), None

    MoELayer.forward = _patched_forward


# ── DynaFlow backend ───────────────────────────────────────────────────

_fwd_manager = DynaFlowManager()


def make_dynaflow_backend(
    manager: DynaFlowManager,
    additional_config: dict,
    require_sym_int: bool = True,
):
    sched_file = os.path.join(
        os.path.dirname(__file__), '..', 'scheduler', 'megatron', 'comet.py',
    )

    def backend(gm: torch.fx.GraphModule, example_inputs) -> object:
        if require_sym_int and not any(isinstance(i, torch.SymInt) for i in example_inputs):
            return gm
        dynaflow_cfg = DynaFlowConfig(
            scheduler_path=sched_file + ':MegatronCometScheduler',
            max_num_splits=1,
            inductor_config=InductorConfig(enabled=False, compile_sizes=set()),
            cudagraph_config=CUDAGraphConfig(enabled=False, capture_sizes=[]),
            additional_config=additional_config,
        )
        file_path, cls_name = dynaflow_cfg.scheduler_path.split(':', 1)
        spec = importlib.util.spec_from_file_location('_megatron_comet_scheduler', file_path)
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


# ── FX graph saving backend ────────────────────────────────────────────

def make_save_graph_backend(save_dir: str):
    graph_count = [0]

    def backend(gm: torch.fx.GraphModule, example_inputs):
        rank = torch.distributed.get_rank()
        path = os.path.join(save_dir, f"comet_graph_rank{rank}_{graph_count[0]}.py")
        os.makedirs(save_dir, exist_ok=True)
        with open(path, "w") as f:
            f.write(gm.print_readable(print_output=False))
        print(f"[rank {rank}] Saved FX graph {graph_count[0]} → {path}")
        graph_count[0] += 1
        return gm

    return backend


# ── Model / data setup ─────────────────────────────────────────────────

def setup_distributed(tp_size: int, ep_size: int) -> None:
    torch.distributed.init_process_group(
        backend='nccl',
        rank=int(os.environ.get('RANK', 0)),
        world_size=int(os.environ.get('WORLD_SIZE', 1)),
    )
    torch.cuda.set_device(torch.distributed.get_rank() % torch.cuda.device_count())

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=ep_size,
    )

    from megatron.core.tensor_parallel.random import (
        get_cuda_rng_tracker,
        get_expert_parallel_rng_tracker_name,
    )
    rng_tracker = get_cuda_rng_tracker()
    rng_tracker.add('model-parallel-rng', 1234)
    rng_tracker.add(get_expert_parallel_rng_tracker_name(), 5678)


MODEL_CONFIGS: dict[str, dict] = {
    # Qwen2-MoE-2.7B with num_layers truncated to 6 for memory fit.
    "qwen2-6layers": {
        "num_layers": 6,
        "hidden_size": 2048,
        "ffn_hidden_size": 1408,
        "num_attention_heads": 16,
        "num_query_groups": 16,  # MHA (no GQA in Qwen2-MoE-2.7B)
        "add_qkv_bias": True,
        "num_moe_experts": 64,
        "moe_router_topk": 4,
        "vocab_size": 152064,
        "max_seq_length": 8192,
        "moe_shared_expert_intermediate_size": 1408,
    },
    # Mixtral 8x7B with num_layers truncated to 8.
    "mixtral-8layers": {
        "num_layers": 8,
        "hidden_size": 4096,
        "ffn_hidden_size": 14336,
        "num_attention_heads": 32,
        "num_query_groups": 8,  # GQA
        "add_qkv_bias": False,
        "num_moe_experts": 8,
        "moe_router_topk": 2,
        "vocab_size": 32000,
        "max_seq_length": 32768,
        "moe_shared_expert_intermediate_size": None,  # Mixtral has no shared experts
    },
}


def create_model(model_cfg: dict, tp_size: int, ep_size: int) -> GPTModel:
    config = TransformerConfig(
        num_layers=model_cfg["num_layers"],
        hidden_size=model_cfg["hidden_size"],
        ffn_hidden_size=model_cfg["ffn_hidden_size"],
        num_attention_heads=model_cfg["num_attention_heads"],
        num_query_groups=model_cfg["num_query_groups"],
        layernorm_epsilon=1e-6,
        add_bias_linear=False,
        add_qkv_bias=model_cfg["add_qkv_bias"],
        # MoE / EP config
        num_moe_experts=model_cfg["num_moe_experts"],
        expert_model_parallel_size=ep_size,
        moe_token_dispatcher_type="alltoall",
        moe_router_topk=model_cfg["moe_router_topk"],
        moe_router_load_balancing_type="none",
        moe_grouped_gemm=False,
        moe_layer_freq=1,
        moe_shared_expert_intermediate_size=model_cfg["moe_shared_expert_intermediate_size"],
    )

    layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=False)

    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=model_cfg["vocab_size"],
        max_sequence_length=model_cfg["max_seq_length"],
        pre_process=True,
        post_process=True,
        parallel_output=False,
    ).cuda()

    from megatron.core.distributed import DistributedDataParallelConfig
    model.ddp_config = DistributedDataParallelConfig()

    return model


def create_optimizer(model: GPTModel) -> MegatronOptimizer:
    optimizer_config = OptimizerConfig(
        optimizer='adam',
        lr=1e-3,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
    )
    return get_megatron_optimizer(config=optimizer_config, model_chunks=[model])


def create_dummy_batch(
    vocab_size: int,
    batch_size: int = 32,
    seq_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = torch.randint(0, vocab_size, (batch_size, seq_length), dtype=torch.long, device='cuda')
    position_ids = (
        torch.arange(seq_length, dtype=torch.long, device='cuda')
        .unsqueeze(0).expand(batch_size, -1)
    )
    attention_mask = (
        torch.tril(torch.ones((seq_length, seq_length), device='cuda', dtype=torch.bool))
        .unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    )
    labels = torch.cat(
        [tokens[:, 1:], torch.zeros(batch_size, 1, dtype=torch.long, device='cuda')],
        dim=1,
    )
    torch._dynamo.mark_dynamic(tokens, 0)
    torch._dynamo.mark_dynamic(position_ids, 0)
    torch._dynamo.mark_dynamic(attention_mask, 0)
    torch._dynamo.mark_dynamic(labels, 0)
    return tokens, position_ids, attention_mask, labels


# ── Benchmark loops ─────────────────────────────────────────────────────

def run_inference(
    compiled_model,
    tokens: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    split_config: SplitConfig | None = None,
    warmup_steps: int = 5,
    trials: int = 10,
) -> tuple[float, float]:
    latency_list: list[float] = []
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
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
    compiled_model,
    optimizer: MegatronOptimizer,
    tokens: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    split_config: SplitConfig | None = None,
    warmup_steps: int = 5,
    trials: int = 10,
) -> tuple[float, float]:
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


# ── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Megatron GPT MoE: eager baseline vs Comet via DynaFlow."
    )
    parser.add_argument(
        "--mode", choices=["megatron", "dynaflow"], default="dynaflow",
        help="'megatron': eager baseline; 'dynaflow': Comet via DynaFlow.",
    )
    parser.add_argument(
        "--config", choices=list(MODEL_CONFIGS.keys()), default="qwen2-6layers",
        help="Model configuration preset.",
    )
    parser.add_argument("--task", choices=["inference", "training"], default="inference")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument(
        "--tp-size", type=int, default=1,
        help="Tensor model parallel size (default: 1).",
    )
    parser.add_argument(
        "--ep-size", type=int, default=0,
        help="Expert model parallel size (default: world_size).",
    )
    parser.add_argument(
        "--save-graph", action="store_true",
        help="Dump compiled FX graphs to --graph-save-dir.",
    )
    parser.add_argument(
        "--graph-save-dir", type=str, default="/tmp",
        help="Directory to save FX graph files.",
    )
    args = parser.parse_args()

    # Default EP size = world_size (pure EP)
    if args.ep_size == 0:
        args.ep_size = int(os.environ.get('WORLD_SIZE', 1))

    setup_distributed(tp_size=args.tp_size, ep_size=args.ep_size)

    model_cfg = MODEL_CONFIGS[args.config]
    model = create_model(model_cfg, tp_size=args.tp_size, ep_size=args.ep_size)

    # Wrap MoELayer.forward as an opaque custom op (both modes).
    # Must be called after model creation so the MoE layer reference is available.
    _patch_moe_as_custom_op(model)
    tokens, position_ids, attention_mask, labels = create_dummy_batch(
        vocab_size=model_cfg["vocab_size"],
        batch_size=args.batch_size,
        seq_length=args.seq_len,
    )

    # Compile the model — backend differs by mode.
    if args.save_graph:
        compiled_model = torch.compile(
            model, backend=make_save_graph_backend(args.graph_save_dir), fullgraph=True,
        )
        split_config = None
    elif args.mode == "dynaflow":
        comet_config = {
            "num_moe_experts": model_cfg["num_moe_experts"],
            "moe_router_topk": model_cfg["moe_router_topk"],
            "hidden_size": model_cfg["hidden_size"],
            "ffn_hidden_size": model_cfg["ffn_hidden_size"],
            "seq_length": args.seq_len,
            "batch_size": args.batch_size,
        }
        dynaflow_backend = make_dynaflow_backend(
            _fwd_manager, additional_config=comet_config,
        )
        compiled_model = torch.compile(
            model, backend=dynaflow_backend, fullgraph=True,
        )
        split_config = SplitConfig(
            num_nano_batches=1,
            batch_sizes=[args.batch_size],
            batch_indices=[0, args.batch_size],
            num_tokens=[args.batch_size],
            num_tokens_padded=[args.batch_size],
            split_indices=[0, args.batch_size],
            is_dryrun=False,
            use_cudagraph=False,
            allow_fallback=False,
        )
    else:
        compiled_model = torch.compile(model, backend="eager", fullgraph=True)
        split_config = None

    if args.task == "training":
        optimizer = create_optimizer(model)
        avg, std = run_training(
            compiled_model, optimizer, tokens, position_ids, attention_mask, labels,
            split_config=split_config,
            warmup_steps=args.warmup_steps,
            trials=args.trials,
        )
    else:
        avg, std = run_inference(
            compiled_model, tokens, position_ids, attention_mask,
            split_config=split_config,
            warmup_steps=args.warmup_steps,
            trials=args.trials,
        )

    print(f"Average latency: {avg:.2f} ms ± {std:.2f} ms")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

import itertools
from dataclasses import dataclass
from typing import Any

import torch
import triton
from triton_dist.kernels.nvidia import (
    create_gemm_ar_context,
    gemm_allreduce_op,
)
from triton_dist.utils import (
    init_nvshmem_by_torch_process_group,
    nvshmem_barrier_all_on_stream,
)
from typing_extensions import override
from vllm.distributed.parallel_state import get_tp_group

# NOTE(yi): the following line is for vLLM only.
# Change this in other systems
from vllm.forward_context import get_forward_context as get_vllm_forward_context

from schedflow.interface import (
    ExecutionContext,
    InputInfo,
    OperatorHandle,
    OpSchedulerBase,
    OpSchedulerConfigBase,
    SplitConfig,
)
from schedflow.matching import MatchingRule, Op
from schedflow.utils import pack_tokens


@dataclass
class FluxSchedulerConfig(OpSchedulerConfigBase):
    """Configuration options for the Flux example scheduler."""

    cudagraph_capture_sizes: list[int]

    @classmethod
    def get_scheduler_cls(cls) -> type[OpSchedulerBase]:
        return FluxScheduler


class FluxScheduler(OpSchedulerBase):
    def __init__(self, config: FluxSchedulerConfig) -> None:
        super().__init__(config, policy_name="flux")
        self.config = config
        self.cudagraph_capture_sizes = config.cudagraph_capture_sizes
        self.comm_stream = torch.cuda.Stream()
        self.comp_stream = torch.cuda.Stream()
        self.triton_distributed_ctx: dict[int, Any] = {}

    def lazy_initialize_triton_distributed_ctx(
        self, hidden_dim: int, output_dim: int, dtype: torch.dtype
    ) -> None:
        tp_group = get_tp_group()
        init_nvshmem_by_torch_process_group(tp_group.device_group)
        world_size = tp_group.world_size
        world_rank = tp_group.rank_in_group
        MAX_M = 16384
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        NUM_COMM_SMS = 36 if hidden_dim == 2048 else 44
        NUM_GEMM_SMS = NUM_SMS - NUM_COMM_SMS
        self.triton_distributed_ctx[output_dim] = create_gemm_ar_context(
            self.comm_stream,
            world_rank,
            world_size,
            world_size,
            MAX_M,
            output_dim,
            dtype,
            NUM_COMM_SMS=NUM_COMM_SMS,
        )
        BM, BN, BK = 128, 256, 64
        num_stages = 4
        num_warps = 8
        self.gemm_config = triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": 1,
                "NUM_GEMM_SMS": NUM_GEMM_SMS,
            },
            num_stages=num_stages,
            num_warps=num_warps,
        )
        nvshmem_barrier_all_on_stream()

    def fused_gemm_allreduce(
        self, x: torch.Tensor, size: int, weight: torch.Tensor
    ) -> torch.Tensor:
        if weight.shape[0] not in self.triton_distributed_ctx:
            self.lazy_initialize_triton_distributed_ctx(
                x.shape[-1], weight.shape[0], x.dtype
            )
        return gemm_allreduce_op(
            self.triton_distributed_ctx[weight.shape[0]],
            x,
            weight,
            self.gemm_config,
            copy_to_local=True,
            USE_MULTIMEM_ST=True,
        )

    @override
    def get_split_rules(self) -> list[MatchingRule]:
        return [
            MatchingRule(condition=(Op(pattern=r"linear"), Op(pattern=r"all_reduce"))),
            # NOTE(yi): attention operators should be split
            # when using cudagraph
            # MatchingRule(condition=Op(pattern=r"unified_attention.*")),
        ]

    @override
    def get_tag_rules(self) -> dict[MatchingRule, set[str]]:
        return {
            MatchingRule(
                condition=(Op(pattern=r"linear"), Op(pattern=r"all_reduce"))
            ): {"linear_all_reduce"},
            MatchingRule(condition=Op(pattern=r"unified_attention.*")): {
                "attention",
                "no-cudagraph",
            },
        }

    @override
    def get_split_config(
        self,
        input_info: InputInfo,
        use_cudagraph: bool,
    ) -> SplitConfig:
        prefix_sum = [0] + list(itertools.accumulate(input_info.num_tokens))
        num_tokens_padded = prefix_sum[-1]
        if use_cudagraph:
            num_tokens_padded = pack_tokens(
                prefix_sum[-1], self.cudagraph_capture_sizes
            )
        return SplitConfig(
            num_nano_batches=1,
            batch_sizes=[input_info.batch_size],
            batch_indices=[0, input_info.batch_size],
            num_tokens=[prefix_sum[-1]],
            num_tokens_padded=[num_tokens_padded],
            split_indices=[0, prefix_sum[-1]],
            is_dryrun=False,
            use_cudagraph=use_cudagraph,
        )

    @override
    async def schedule(self, context: ExecutionContext) -> None:
        num_batches = context.split_config.num_nano_batches
        batch_indices = list(range(num_batches))
        ctx = get_vllm_forward_context()
        attn_metadata_list = ctx.attn_metadata
        assert isinstance(attn_metadata_list, list)

        while batch_indices:
            ops: list[tuple[int, OperatorHandle]] = []
            for batch_idx in batch_indices:
                op = await context.pop(batch_idx)
                if op is None:
                    batch_indices.remove(batch_idx)
                    continue
                ops.append((batch_idx, op))

            for batch_idx, op in ops:
                ctx.attn_metadata = attn_metadata_list[batch_idx]
                if "linear_all_reduce" in op.tag:
                    func = self.fused_gemm_allreduce
                    with torch.cuda.stream(self.comp_stream):
                        await context.execute((op,), func)
                else:
                    with torch.cuda.stream(self.comp_stream):
                        await context.execute((op,))

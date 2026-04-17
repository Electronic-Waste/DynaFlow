import itertools
from typing import Any

import flux
import torch
from typing_extensions import override
from vllm.distributed.parallel_state import get_tp_group

from dynaflow.config import DynaFlowConfig

# NOTE(yi): the following line is for vLLM only.
# Change this in other systems
from vllm.forward_context import get_forward_context as get_vllm_forward_context

from dynaflow.interface import (
    ExecutionContext,
    InputInfo,
    OperatorHandle,
    OpSchedulerBase,
    SplitConfig,
)
from dynaflow.matching import MatchingRule, Op
from dynaflow.utils import pack_tokens


class FluxScheduler(OpSchedulerBase):
    def __init__(self, config: DynaFlowConfig) -> None:
        super().__init__(policy_name="flux")
        self.cudagraph_capture_sizes = config.cudagraph_config.capture_sizes
        self.comm_stream = torch.cuda.Stream()
        self.comp_stream = torch.cuda.Stream()
        self._gemm_rs_ops: dict[tuple[int, int], flux.GemmRS] = {}
        self._tp_group: Any = None
        self._world_size: int = 0
        self._nnodes: int = 1

    def _get_gemm_rs(
        self, K: int, N: int, dtype: torch.dtype
    ) -> flux.GemmRS:
        key = (N, K)
        if key in self._gemm_rs_ops:
            return self._gemm_rs_ops[key]
        if self._tp_group is None:
            tp = get_tp_group()
            self._tp_group = tp.device_group
            self._world_size = tp.world_size
            flux.init_flux_shm(self._tp_group)
        MAX_M = 16384
        op = flux.GemmRS(
            self._tp_group,
            self._nnodes,
            MAX_M,
            N,
            input_dtype=dtype,
            output_dtype=dtype,
            transpose_weight=False,
        )
        self._gemm_rs_ops[key] = op
        return op

    def fused_gemm_allreduce(
        self, x: torch.Tensor, size: int, weight: torch.Tensor
    ) -> torch.Tensor:
        K, N = x.shape[-1], weight.shape[0]
        gemm_rs = self._get_gemm_rs(K, N, x.dtype)

        # Fused GEMM + ReduceScatter → [M/TP, N]
        rs_output = gemm_rs.forward(x, weight)

        # AllGather to reconstruct [M, N]
        full_output = torch.empty(
            [x.shape[0], N], dtype=rs_output.dtype, device=rs_output.device
        )
        torch.distributed.all_gather_into_tensor(
            full_output, rs_output, group=self._tp_group
        )
        return full_output

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
            num_tokens_padded, use_cudagraph = pack_tokens(
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
            allow_fallback=False,
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

import itertools
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from flashinfer import green_ctx
from typing_extensions import override

# NOTE(yi): the following line is for vLLM only.
# Change this in other systems
from vllm.forward_context import get_forward_context as get_vllm_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm

from schedflow.interface import (
    ExecutionContext,
    InputInfo,
    OperatorHandle,
    OpSchedulerBase,
    SplitConfig,
)
from schedflow.matching import MatchingRule, Mod, Op
from schedflow.utils import pack_tokens


@dataclass
class TokenWeaveSchedulerConfig:
    """Configuration options for the TokenWeave example scheduler."""

    min_nano_split_tokens: int
    max_num_nano_batches: int
    cudagraph_capture_sizes: list[int]


class TokenWeaveScheduler(OpSchedulerBase):
    def __init__(self, config: TokenWeaveSchedulerConfig) -> None:
        super().__init__("tokenweave")
        self.config = config
        self.cudagraph_capture_sizes = config.cudagraph_capture_sizes
        self.comm_stream: torch.cuda.Stream = (
            green_ctx.split_device_green_ctx_by_sm_count(
                dev=torch.device(f"cuda:{torch.cuda.current_device()}"), sm_counts=[48]
            )[0][0]
        )  # type: ignore
        self.comp_stream = torch.cuda.Stream()
        self.symm_mem_hdl: Any | None = None

    def fused_ar_add_rms_norm(
        self,
        output_residual: bool,
        x: torch.Tensor,
        size: int,
        *args,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if len(args) == 1:
            residual = torch.zeros_like(x)
            weight = args[0]
        else:
            residual, weight = args
        if self.symm_mem_hdl is None:
            self.lazy_initialize_tokenweave_custom_op(x.shape[-1], x.dtype, x.device)
        assert self.symm_mem_hdl is not None
        torch.ops._tokenweave_C.fused_rs_ln_ag_cta(
            x,
            residual,
            weight.data,
            self.symm_mem_hdl.multicast_ptr,
            self.symm_mem_hdl.signal_pad_ptrs_dev,
            self.tp_rank,
            self.tp_size,
            self.MAX_CTAS,
            1e-5,
        )
        return (x, residual) if output_residual else x

    def lazy_initialize_tokenweave_custom_op(
        self, hidden_dim: int, dtype: torch.dtype, device: torch.device
    ) -> None:
        if self.symm_mem_hdl is not None:
            return
        import sys

        # NOTE(yi): temporary hardcode
        sys.path.append("/root/vllm/custom_ops/build")
        import _tokenweave_C  # noqa: F401
        import torch.distributed._symmetric_memory as symm_mem
        from vllm.distributed.parallel_state import get_tp_group

        tp_group = get_tp_group()
        self.tp_rank, self.tp_size = tp_group.rank_in_group, tp_group.world_size
        self.MAX_CTAS = 48

        self.staging_buffer = symm_mem.empty(
            (16384, hidden_dim), dtype=dtype, device=device
        )
        tp_group = get_tp_group()
        self.symm_mem_hdl = symm_mem.rendezvous(
            self.staging_buffer, tp_group.device_group
        )

    @override
    def get_split_rules(self) -> list[MatchingRule]:
        return [
            MatchingRule(
                condition=(Op(pattern=r"all_reduce"), Mod(target_cls=RMSNorm))
            ),
            # NOTE(yi): attention operators should be split
            # when using cudagraph
            # MatchingRule(condition=Op(pattern=r"unified_attention.*")),
        ]

    @override
    def get_tag_rules(self) -> dict[MatchingRule, set[str]]:
        return {
            MatchingRule(
                condition=(Op(pattern=r"all_reduce"), Mod(target_cls=RMSNorm))
            ): {"ar_add_rms_norm"},
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
        # print(f"total num tokens: {sum(input_info.num_tokens)}")
        mid = min(
            range(len(prefix_sum)),
            key=lambda i: abs(prefix_sum[i] - (prefix_sum[-1] - prefix_sum[i])),
        )

        if (
            prefix_sum[mid] < self.config.min_nano_split_tokens
            or (prefix_sum[-1] - prefix_sum[mid]) < self.config.min_nano_split_tokens
        ):
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
        else:
            num_tokens_padded = [
                prefix_sum[mid],
                prefix_sum[-1] - prefix_sum[mid],
            ]
            if use_cudagraph:
                num_tokens_padded = [
                    pack_tokens(num_tokens, self.cudagraph_capture_sizes)
                    for num_tokens in num_tokens_padded
                ]
            return SplitConfig(
                num_nano_batches=2,
                batch_sizes=[mid, input_info.batch_size - mid],
                batch_indices=[0, mid, input_info.batch_size],
                num_tokens=[prefix_sum[mid], prefix_sum[-1] - prefix_sum[mid]],
                num_tokens_padded=num_tokens_padded,
                split_indices=[0, prefix_sum[mid], prefix_sum[-1]],
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
                if "ar_add_rms_norm" in op.tag:
                    # NOTE(yi): temporary hardcode
                    func = partial(
                        self.fused_ar_add_rms_norm, op.module_name != "submod_129"
                    )
                    with torch.cuda.stream(self.comm_stream):
                        await context.execute((op,), func)
                else:
                    with torch.cuda.stream(self.comp_stream):
                        await context.execute((op,))

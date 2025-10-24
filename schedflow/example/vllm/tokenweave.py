import itertools
from dataclasses import dataclass

import torch
from typing_extensions import override

# NOTE(yi): the following line is for vLLM only.
# Change this in other systems
from vllm.forward_context import get_forward_context as get_vllm_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm

from schedflow.interface import (
    ExecutionContext,
    InputInfo,
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


def fused_ar_add_rms_norm(
    world_size: int,
    world_rank: int,
    x: torch.Tensor,
    size: int,
    residual: torch.Tensor,
    weight: torch.nn.Parameter,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(x)
    return torch.ops.vllm.flashinfer_trtllm_fused_allreduce_norm(
        x,
        residual,
        weight,
        1e-5,
        world_size,
        world_rank,
        False,
        False,
        True,
        size,
        1,
        False,
        output,
    )


class TokenWeaveScheduler(OpSchedulerBase):
    def __init__(self, config: TokenWeaveSchedulerConfig) -> None:
        super().__init__("tokenweave")
        self.config = config
        self.cudagraph_capture_sizes = config.cudagraph_capture_sizes
        self.comm_stream = torch.cuda.Stream()
        self.comp_stream = torch.cuda.Stream()

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
        assert self.config and self.cudagraph_capture_sizes
        prefix_sum = [0] + list(itertools.accumulate(input_info.num_tokens))
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
        from vllm.distributed.parallel_state import get_tp_group

        world_size = get_tp_group().world_size
        world_rank = get_tp_group().rank_in_group
        attn_metadata_list = ctx.attn_metadata
        assert isinstance(attn_metadata_list, list)

        def fused_ar_add_rms_norm_op(x, size, *args):
            if len(args) == 1:
                residual = torch.zeros_like(x)
                return fused_ar_add_rms_norm(
                    world_size, world_rank, x, size, residual, args[0]
                )
            else:
                return fused_ar_add_rms_norm(
                    world_size, world_rank, x, size, args[0], args[1]
                )

        while batch_indices:
            ops = []
            for batch_idx in batch_indices:
                op = await context.pop(batch_idx)
                if op is None:
                    batch_indices.remove(batch_idx)
                    continue
                ops.append((batch_idx, op))

            for batch_idx, op in ops:
                ctx.attn_metadata = attn_metadata_list[batch_idx]
                if "ar_add_rms_norm" in op.tag:
                    await context.execute((op,), fused_ar_add_rms_norm_op)
                else:
                    await context.execute((op,))

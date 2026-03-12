import itertools

import torch
from typing_extensions import override

# NOTE(yi): the following line is for SGLang only.
# Change this in other systems
from sglang.srt.compilation.piecewise_context_manager import get_forward_context as get_sglang_forward_context, replace_forward_context as replace_sglang_forward_context

from dynaflow.config import DynaFlowConfig
from dynaflow.interface import (
    ExecutionContext,
    InputInfo,
    OpSchedulerBase,
    SplitConfig,
)
from dynaflow.matching import MatchingRule, Op
from dynaflow.utils import pack_tokens


class NanoFlowScheduler(OpSchedulerBase):
    """Simple scheduler that overlaps network and compute when possible."""

    def __init__(self, config: DynaFlowConfig) -> None:
        super().__init__("nanoflow")
        additional = config.additional_config
        self.min_nano_split_tokens = additional["min_nano_split_tokens"]
        self.max_num_nano_batches = config.max_num_splits
        self.cudagraph_capture_sizes = config.cudagraph_config.capture_sizes
        self.comm_stream = torch.cuda.Stream()
        self.comp_stream = torch.cuda.Stream()

    @override
    def get_split_rules(self) -> list[MatchingRule]:
        return [
            MatchingRule(condition=Op(pattern=r"inplace_all_reduce|outplace_all_reduce")),
            # NOTE(yi): attention operators should be split
            # when using cudagraph
            # MatchingRule(condition=Op(pattern=r"unified_attention.*")),
        ]

    @override
    def get_tag_rules(self) -> dict[MatchingRule, set[str]]:
        return {
            MatchingRule(condition=Op(pattern=r"inplace_all_reduce|outplace_all_reduce")): {"network"},
            # MatchingRule(condition=Op(pattern=r"unified_attention.*")): {"attention", "no-cudagraph"},
        }

    @override
    def get_split_config(
        self,
        input_info: InputInfo,
        use_cudagraph: bool,
    ) -> SplitConfig:
        prefix_sum = [0] + list(itertools.accumulate(input_info.num_tokens))
        mid = min(
            range(len(prefix_sum)),
            key=lambda i: abs(prefix_sum[i] - (prefix_sum[-1] - prefix_sum[i])),
        )

        if (
            prefix_sum[mid] < self.min_nano_split_tokens
            or (prefix_sum[-1] - prefix_sum[mid]) < self.min_nano_split_tokens
        ):
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
            )
        else:
            num_tokens_padded = [
                prefix_sum[mid],
                prefix_sum[-1] - prefix_sum[mid],
            ]
            if use_cudagraph:
                cudagraph_pack_results = [
                    pack_tokens(num_tokens, self.cudagraph_capture_sizes)
                    for num_tokens in num_tokens_padded
                ]
                if all(use for _, use in cudagraph_pack_results):
                    num_tokens_padded = [padded for padded, _ in cudagraph_pack_results]
                else:
                    use_cudagraph = False
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
        fwd_ctx = get_sglang_forward_context()
        assert fwd_ctx is not None and fwd_ctx.ubatch_contexts is not None

        while batch_indices:
            ops = []
            for batch_idx in batch_indices:
                op = await context.pop(batch_idx)
                if op is None:
                    batch_indices.remove(batch_idx)
                    continue
                ops.append((batch_idx, op))

            for batch_idx, op in ops:
                if "network" in op.tag:
                    stream = self.comm_stream
                else:
                    stream = self.comp_stream

                replace_sglang_forward_context(fwd_ctx.ubatch_contexts[batch_idx])

                with torch.cuda.stream(stream):
                    await context.execute((op,))

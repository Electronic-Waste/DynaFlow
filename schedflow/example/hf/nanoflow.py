import itertools
from dataclasses import dataclass

import torch
from typing_extensions import override

from schedflow.interface import (
    ExecutionContext,
    InputInfo,
    OpSchedulerBase,
    OpSchedulerConfigBase,
    SplitConfig,
)
from schedflow.matching import MatchingRule, Op
from schedflow.utils import pack_tokens


@dataclass
class NanoFlowSchedulerConfig(OpSchedulerConfigBase):
    """Configuration options for the NanoFlow example scheduler."""

    min_nano_split_tokens: int
    max_num_nano_batches: int
    cudagraph_capture_sizes: list[int]

    @classmethod
    def get_scheduler_cls(cls) -> type[OpSchedulerBase]:
        return NanoFlowScheduler


class NanoFlowScheduler(OpSchedulerBase):
    """Simple scheduler that overlaps network and compute when possible."""

    def __init__(self, config: NanoFlowSchedulerConfig) -> None:
        super().__init__(config, policy_name="nanoflow")
        self.config = config
        self.cudagraph_capture_sizes = config.cudagraph_capture_sizes
        self.comm_stream = torch.cuda.Stream()
        self.comp_stream = torch.cuda.Stream()

        def _is_all_reduce(node_list: list[torch.fx.Node], start_idx: int) -> bool:
            suppress_tokens = {"gate_proj", "up_proj", "q_proj", "k_proj", "v_proj"}
            idx = start_idx - 1
            while idx >= 0:
                node = node_list[idx]
                if (
                    node.op != "call_function"
                    or not callable(node.target)
                    or node.target.__name__ != "linear"
                ):
                    idx -= 1
                    continue
                for a in node.args:
                    if a is not None and any(tok in str(a) for tok in suppress_tokens):
                        return False
                break
            return True

        self.is_all_reduce = _is_all_reduce

    @override
    def get_split_rules(self) -> list[MatchingRule]:
        return [
            MatchingRule(
                condition=Op(pattern=r"prim_redistribute"), hook=self.is_all_reduce
            ),
        ]

    @override
    def get_tag_rules(self) -> dict[MatchingRule, set[str]]:
        return {
            MatchingRule(
                condition=Op(pattern=r"prim_redistribute"), hook=self.is_all_reduce
            ): {"network"},
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
        # ctx = get_vllm_forward_context()
        # attn_metadata_list = ctx.attn_metadata
        # assert isinstance(attn_metadata_list, list)

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
                # ctx.attn_metadata = attn_metadata_list[batch_idx]
                with torch.cuda.stream(stream):
                    await context.execute((op,))

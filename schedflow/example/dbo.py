import itertools
from dataclasses import dataclass

import torch
from typing_extensions import override

# NOTE(yi): the following line is for vLLM only.
# Change this in other systems
from vllm.forward_context import DPMetadata
from vllm.forward_context import (
    get_forward_context as get_vllm_forward_context,
)

from schedflow.interface import (
    ExecutionContext,
    InputInfo,
    OperatorHandle,
    OpSchedulerBase,
    SplitConfig,
)
from schedflow.utils import pack_tokens


@dataclass
class DBOSchedulerConfig:
    """Configuration options for the DBO example scheduler."""

    min_nano_split_tokens: int
    max_num_nano_batches: int
    cudagraph_capture_sizes: list[int]


class DBOScheduler(OpSchedulerBase):
    """Dual-batch overlap example scheduler.

    Demonstrates a policy that splits a batch into at most two nano-batches
    and orchestrates a handcrafted execution sequence to overlap attention,
    dispatch/communication, and expert compute.
    """

    def __init__(self, config: DBOSchedulerConfig) -> None:
        super().__init__("dbo")
        self.config = config
        self.cudagraph_capture_sizes = config.cudagraph_capture_sizes
        self.comm_stream = torch.cuda.Stream()
        self.comp_stream = torch.cuda.Stream()
        self.dp_metadata: list[DPMetadata] | None = None

    @override
    def get_splitting_ops(self) -> list[str]:
        return [
            # NOTE(yi): attention operators should be split
            # when using cudagraph
            "vllm.unified_attention",
            "vllm.unified_attention_with_output",
            "vllm.moe_forward_dispatch",
            "vllm.moe_forward_expert",
            "vllm.moe_forward_combine",
            "vllm.moe_forward_combine_with_shared",
        ]

    @override
    def get_op_tags(self) -> dict[str, set[str]]:
        return {
            "vllm.unified_attention": {"attention", "no-cudagraph"},
            "vllm.unified_attention_with_output": {"attention", "no-cudagraph"},
            # NOTE(yi): We disable TorchInductor for MoE operators
            # as its input size cannot be determined
            "vllm.moe_forward_dispatch": {
                "dispatch",
                "network",
                "no-inductor",
            },
            "vllm.moe_forward_expert": {"expert", "no-inductor"},
            "vllm.moe_forward_combine": {
                "combine",
                "network",
                "no-inductor",
            },
            "vllm.moe_forward_combine_with_shared": {
                "combine",
                "network",
                "no-inductor",
            },
        }

    @override
    def get_split_config(
        self,
        input_info: InputInfo,
        use_cudagraph: bool,
    ) -> SplitConfig:
        """Compute a two-way split when beneficial, else fall back to 1-way."""
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

    def set_dp_metadata(self, metadata: list[DPMetadata] | None) -> None:
        self.dp_metadata = metadata

    @override
    async def schedule(self, context: ExecutionContext) -> None:
        """Run the handcrafted overlap sequence across nano-batches."""
        num_batches = context.split_config.num_nano_batches
        batch_indices = list(range(num_batches))
        ctx = get_vllm_forward_context()
        attn_metadata_list = ctx.attn_metadata
        assert isinstance(attn_metadata_list, list)

        if num_batches == 1:
            ctx.attn_metadata = attn_metadata_list[0]
            while (op := await context.pop(0)) is not None:
                await context.execute((op,))
            return

        warm_up_sequence = [
            ("attention", 0),
            ("attention", 1),
            ("attention", 0),
        ]
        sequence = [
            ("attention", 1),
            ("dispatch", 0),
            ("expert", 0),
            ("dispatch", 1),
            ("expert", 1),
            ("combine", 0),
            ("attention", 0),
            ("combine", 1),
        ]
        sequence_keys = set(key for key, _ in sequence)
        current_batch_idx = batch_indices[0]
        current_seq = warm_up_sequence
        current_seq_idx = -len(warm_up_sequence)
        buffered_op: list[OperatorHandle | None] = [None] * num_batches

        while batch_indices:
            assert current_batch_idx in batch_indices
            if buffered_op[current_batch_idx] is None:
                op = await context.pop(current_batch_idx)
                if op is None:
                    batch_indices.remove(current_batch_idx)
                    current_batch_idx ^= 1
                    current_seq_idx += 1
                    continue
                if not sequence_keys.intersection(op.tag):
                    with torch.cuda.stream(self.comp_stream):
                        await context.execute((op,))
                    continue
                if current_seq[current_seq_idx][1] != current_batch_idx:
                    buffered_op[current_batch_idx] = op
                    current_batch_idx ^= 1
                    continue
            else:
                op = buffered_op[current_batch_idx]
                assert op is not None
                buffered_op[current_batch_idx] = None

            assert current_seq[current_seq_idx][0] in op.tag
            current_seq_idx += 1
            if current_seq_idx > 0:
                current_seq_idx %= len(sequence)
            elif current_seq_idx == 0:
                current_seq = sequence
            stream = self.comm_stream if ("network" in op.tag) else self.comp_stream
            ctx.attn_metadata = attn_metadata_list[current_batch_idx]
            if self.dp_metadata is not None:
                ctx.dp_metadata = self.dp_metadata[current_batch_idx]
            with torch.cuda.stream(stream):
                await context.execute((op,))

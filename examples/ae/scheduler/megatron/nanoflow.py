import torch
from typing_extensions import override

from dynaflow.config import DynaFlowConfig
from dynaflow.interface import ExecutionContext, InputInfo, OpSchedulerBase, SplitConfig
from dynaflow.matching import MatchingRule, Op


class NanoFlowMegatronScheduler(OpSchedulerBase):
    """NanoFlow scheduler for Megatron-LM TP overlap (all-reduce + all-gather)."""

    def __init__(self, config: DynaFlowConfig) -> None:
        super().__init__(policy_name="nanoflow_megatron")
        additional = config.additional_config or {}
        self.min_nano_split_tokens: int = additional.get("min_nano_split_tokens", 64)
        self.num_nano_batches: int = additional.get("num_nano_batches", 2)
        self.comm_stream = torch.cuda.Stream()
        self.comp_stream = torch.cuda.Stream()

    @override
    def get_split_rules(self) -> list[MatchingRule]:
        # Match flat collective nodes emitted after monkey-patching:
        #   - all_reduce(_?): from reduce_from_tensor_model_parallel_region patch
        #   - all_gather_into_tensor: from gather_from_tensor_model_parallel_region patch
        #   - bwd_allreduce_marker: identity forward / all-reduce backward custom op;
        #     placing it on comm_stream causes its backward all-reduce to inherit
        #     comm_stream, enabling backward compute/comm overlap automatically.
        return [
            MatchingRule(condition=Op(pattern=r"all_reduce_?")),
            MatchingRule(condition=Op(pattern=r"all_gather_into_tensor")),
            MatchingRule(condition=Op(pattern=r"bwd_allreduce_marker")),
        ]

    @override
    def get_tag_rules(self) -> dict[MatchingRule, set[str]]:
        return {
            MatchingRule(condition=Op(pattern=r"all_reduce_?")): {"network"},
            MatchingRule(condition=Op(pattern=r"all_gather_into_tensor")): {"network"},
            MatchingRule(condition=Op(pattern=r"bwd_allreduce_marker")): {"network", "bwd_marker"},
        }

    @override
    def get_split_config(
        self, input_info: InputInfo, use_cudagraph: bool = False
    ) -> SplitConfig:
        batch_size = input_info.batch_size
        n = self.num_nano_batches
        chunk = batch_size // n
        if chunk < self.min_nano_split_tokens:
            # Fall back to fewer splits
            n = max(1, batch_size // self.min_nano_split_tokens)
            chunk = batch_size // n if n > 1 else batch_size
        if n <= 1:
            return SplitConfig(
                num_nano_batches=1,
                batch_sizes=[batch_size],
                batch_indices=[0, batch_size],
                num_tokens=[batch_size],
                num_tokens_padded=[batch_size],
                split_indices=[0, batch_size],
                is_dryrun=False,
                use_cudagraph=use_cudagraph,
            )
        sizes = [chunk] * n
        sizes[-1] = batch_size - chunk * (n - 1)  # last gets remainder
        indices = [chunk * i for i in range(n)] + [batch_size]
        return SplitConfig(
            num_nano_batches=n,
            batch_sizes=sizes,
            batch_indices=indices,
            num_tokens=sizes,
            num_tokens_padded=sizes,
            split_indices=indices,
            is_dryrun=False,
            use_cudagraph=use_cudagraph,
        )

    @override
    async def schedule(self, context: ExecutionContext) -> None:
        num_batches = context.split_config.num_nano_batches
        batch_indices = list(range(num_batches))

        if num_batches >= 2:
            # Warm-up: advance nb0 by one subgraph so that nb0 and nb1
            # alternate between comp_stream and comm_stream each iteration.
            op = await context.pop(0)
            if op is None:
                batch_indices.remove(0)
            else:
                stream = self.comm_stream if "network" in op.tag else self.comp_stream
                with torch.cuda.stream(stream):
                    await context.execute((op,))

        while batch_indices:
            for batch_idx in list(batch_indices):
                op = await context.pop(batch_idx)
                if op is None:
                    batch_indices.remove(batch_idx)
                    continue
                stream = self.comm_stream if "network" in op.tag else self.comp_stream
                with torch.cuda.stream(stream):
                    await context.execute((op,))

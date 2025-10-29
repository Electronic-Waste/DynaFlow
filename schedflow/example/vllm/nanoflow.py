import itertools
from dataclasses import dataclass
from functools import partial

import flashinfer.comm as flashinfer_comm
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
    OpSchedulerConfigBase,
    SplitConfig,
)
from schedflow.matching import MatchingRule, Mod, Op
from schedflow.utils import pack_tokens


@dataclass
class NanoFlowSchedulerConfig(OpSchedulerConfigBase):
    """Configuration options for the NanoFlow example scheduler."""

    min_nano_split_tokens: int
    max_num_nano_batches: int
    use_ar_norm_fusion: bool
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
        self.use_ar_norm_fusion = config.use_ar_norm_fusion
        self.comm_stream: torch.cuda.Stream = (
            green_ctx.split_device_green_ctx_by_sm_count(
                dev=torch.device(f"cuda:{torch.cuda.current_device()}"), sm_counts=[48]
            )[0][0]
        )  # type: ignore
        self.comp_stream = torch.cuda.Stream()
        self.ipc_handles: list[list[int]] | None = None
        self.workspace_tensor: torch.Tensor | None = None

    @override
    def get_split_rules(self) -> list[MatchingRule]:
        return [
            MatchingRule(condition=Op(pattern=r"all_reduce"))
            if not self.use_ar_norm_fusion
            else MatchingRule(
                condition=(Op(pattern=r"all_reduce"), Mod(target_cls=RMSNorm))
            ),
            # NOTE(yi): attention operators should be split
            # when using cudagraph
            # MatchingRule(condition=Op(pattern=r"unified_attention.*")),
        ]

    @override
    def get_tag_rules(self) -> dict[MatchingRule, set[str]]:
        return {
            MatchingRule(condition=Op(pattern=r"all_reduce"))
            if not self.use_ar_norm_fusion
            else MatchingRule(
                condition=(Op(pattern=r"all_reduce"), Mod(target_cls=RMSNorm))
            ): {"network"},
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

    def lazy_initialize_ipc_workspace(self, hidden_dim: int) -> None:
        if self.ipc_handles is not None:
            return
        from vllm.distributed.parallel_state import get_tp_group

        tp_group = get_tp_group()
        self.tp_rank, self.tp_size = tp_group.rank_in_group, tp_group.world_size
        self.ipc_handles, self.workspace_tensor = (
            flashinfer_comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                max_token_num=16384,
                hidden_dim=hidden_dim,
                group=tp_group.device_group,
                use_fp32_lamport=True,
            )
        )

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
        residual_out = torch.empty_like(residual)
        output = torch.empty_like(x)
        if self.ipc_handles is None:
            self.lazy_initialize_ipc_workspace(x.shape[-1])
        assert self.workspace_tensor is not None
        flashinfer_comm.trtllm_allreduce_fusion(
            allreduce_in=x,
            token_num=size,
            residual_in=residual,
            residual_out=residual_out,
            norm_out=output,
            rms_gamma=weight,
            rms_eps=1e-5,
            world_rank=self.tp_rank,
            world_size=self.tp_size,
            hidden_dim=x.shape[-1],
            workspace_ptrs=self.workspace_tensor,
            launch_with_pdl=False,
            use_oneshot=False,
            trigger_completion_at_end=False,
            fp32_acc=True,
            pattern_code=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,  # pyright: ignore[reportArgumentType]
            allreduce_out=None,
            quant_out=None,
            scale_out=None,
            layout_code=flashinfer_comm.QuantizationSFLayout.SWIZZLED_128x4,  # pyright: ignore[reportArgumentType]
            scale_factor=None,
        )
        return (output, x) if output_residual else output

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
                if "network" in op.tag:
                    # NOTE(yi): temporary hardcode
                    func = partial(
                        self.fused_ar_add_rms_norm, op.module_name != "submod_321"
                    )
                    with torch.cuda.stream(self.comm_stream):
                        await context.execute(
                            (op,), func if self.use_ar_norm_fusion else None
                        )
                else:
                    with torch.cuda.stream(self.comp_stream):
                        await context.execute((op,))

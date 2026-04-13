import itertools

import torch
import torch.nn.functional as F
from typing_extensions import override

from dynaflow.config import DynaFlowConfig
from dynaflow.interface import (
    ExecutionContext,
    InputInfo,
    OpSchedulerBase,
    SplitConfig,
)
from dynaflow.matching import MatchingRule, Op


def generate_scatter_index(
    splits: torch.Tensor,
    num_tokens: int,
    topk: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic uniform token-to-expert assignment.

    Reproduces the logic from the Flux Megatron patch
    (megatron_flux.patch:4457-4479).
    """
    choosed_experts = torch.zeros((num_tokens, topk), dtype=torch.int64)
    bin_counter = splits.clone()
    offsets = torch.cumsum(splits, dim=0) - splits

    for tid in range(num_tokens):
        _, bins = torch.topk(bin_counter, topk)
        choosed_experts[tid] = bins
        bin_counter[bins] -= 1

    scatter_index = torch.zeros((num_tokens, topk), dtype=torch.int64)
    write_pos = torch.zeros_like(splits)
    for i in range(num_tokens):
        for j in range(topk):
            eid = choosed_experts[i][j].item()
            scatter_index[i][j] = write_pos[eid] + offsets[eid]
            write_pos[eid] += 1

    return choosed_experts, scatter_index.to(device)


class MegatronCometScheduler(OpSchedulerBase):
    """Scheduler that substitutes Megatron MoE with Comet fused kernels.

    Splits the FX graph around the opaque ``moe_forward`` custom op.
    At schedule time, intercepts the MoE subgraph and replaces it with
    Comet AGScatter + GatherRS using pre-computed uniform routing tensors.

    Non-MoE subgraphs execute normally (via the default path).
    """

    def __init__(self, config: DynaFlowConfig) -> None:
        super().__init__(policy_name="megatron_comet")
        self._cfg = config.additional_config

        # Comet state — lazily initialized on first _comet_moe_call()
        self._flux_ag_op = None
        self._flux_rs_op = None
        self._splits_gpu: torch.Tensor | None = None
        self._splits_cpu: torch.Tensor | None = None
        self._scatter_index: torch.Tensor | None = None
        self._intermediate_buf: torch.Tensor | None = None
        self._fake_fc1: torch.Tensor | None = None
        self._fake_fc2: torch.Tensor | None = None
        self._initialized = False

    def _init_comet_ops(self) -> None:
        """Initialize Comet ops and pre-compute static routing tensors.

        Called lazily on first ``_comet_moe_call()`` — ``parallel_state``
        must already be initialized.
        """
        if self._initialized:
            return

        import flux

        from megatron.core import parallel_state

        cfg = self._cfg
        num_moe_experts = cfg["num_moe_experts"]
        moe_router_topk = cfg["moe_router_topk"]
        hidden_size = cfg["hidden_size"]
        ffn_hidden_size = cfg["ffn_hidden_size"]
        seq_length = cfg["seq_length"]
        batch_size = cfg["batch_size"]

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
        ep_world_size = parallel_state.get_expert_model_parallel_world_size()
        dp_size = world_size // tp_world_size
        device = torch.cuda.current_device()

        tokens_per_rank = batch_size * seq_length
        global_tokens = tokens_per_rank * world_size

        # --- NVSHMEM & Comet environment ---
        TP_GROUP = torch.distributed.group.WORLD
        ep_group = parallel_state.get_expert_model_parallel_group()
        flux.init_flux_shm(TP_GROUP)
        tp_env = flux.DistEnvTPWithEP(
            tp_group=TP_GROUP, nnodes=1, ep_group=ep_group
        )

        flux_m_max = global_tokens * moe_router_topk
        nexperts_ep = num_moe_experts // ep_world_size
        ffn_shard = ffn_hidden_size // tp_world_size

        moe_args = flux.MoeArguments(
            max_ntokens=flux_m_max // moe_router_topk,
            hidden=hidden_size,
            ffn_hidden=ffn_hidden_size,
            nexperts=num_moe_experts,
            topk=moe_router_topk,
            input_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
        )

        if flux.util.get_arch() >= 90:
            self._flux_ag_op = flux.GemmGroupedV3AGScatter(tp_env, moe_args)
            self._flux_rs_op = flux.GemmGroupedV3GatherRS(
                num_moe_experts,
                flux_m_max,
                hidden_size,
                moe_router_topk,
                rank,
                world_size,
                tp_world_size,
                ep_world_size,
            )
        else:
            self._flux_ag_op = flux.GemmGroupedV2AGScatterOp(tp_env, moe_args)
            self._flux_rs_op = flux.GemmGroupedV2GatherRSOp(
                TP_GROUP,
                num_moe_experts,
                flux_m_max,
                hidden_size,
                moe_router_topk,
                torch.bfloat16,
                tp_world_size,
                ep_world_size,
                1,  # max_input_groups
            )

        # --- Pre-compute uniform routing tensors ---
        tokens_per_expert = global_tokens * moe_router_topk // num_moe_experts
        self._splits_gpu = torch.full(
            (num_moe_experts,), tokens_per_expert, dtype=torch.int32, device=device
        )
        self._splits_cpu = self._splits_gpu.cpu()
        _, self._scatter_index = generate_scatter_index(
            self._splits_cpu, global_tokens, moe_router_topk, device
        )
        self._scatter_index = self._scatter_index.to(torch.int32)

        # --- Pre-allocate buffers ---
        nrows_ep = tokens_per_expert * nexperts_ep
        self._intermediate_buf = torch.zeros(
            (nrows_ep, ffn_shard), dtype=torch.bfloat16, device=device
        )

        # Fake expert weights (benchmark-only)
        self._fake_fc1 = torch.rand(
            (nexperts_ep, ffn_shard, hidden_size),
            dtype=torch.bfloat16,
            device=device,
        )
        self._fake_fc2 = torch.rand(
            (nexperts_ep, hidden_size, ffn_shard),
            dtype=torch.bfloat16,
            device=device,
        )
        print(
            f"[rank {rank}] Comet scheduler init: "
            f"tp={tp_world_size} ep={ep_world_size} dp={dp_size} "
            f"nexperts_ep={nexperts_ep} ffn_shard={ffn_shard} "
            f"tokens_per_rank={tokens_per_rank} global_tokens={global_tokens} "
            f"flux_m_max={flux_m_max} "
            f"scatter_index={list(self._scatter_index.shape)}"
        )
        self._initialized = True

    def _comet_moe_call(self, hidden_states: torch.Tensor, *args) -> torch.Tensor:
        """Comet MoE forward — called by DynaFlow engine as func replacement.

        Extra positional args (e.g. symbolic batch size) are ignored.
        """
        self._init_comet_ops()

        cfg = self._cfg
        orig_shape = hidden_states.shape
        inputs_shard = hidden_states.reshape(-1, cfg["hidden_size"]).to(
            torch.bfloat16
        )

        # Layer 0: AllGather + Scatter + GroupedGEMM
        self._flux_ag_op.clear_buffers()
        self._flux_ag_op.forward(
            inputs_shard=inputs_shard,
            weights=self._fake_fc1,
            splits_gpu=self._splits_gpu,
            scatter_index=self._scatter_index,
            outputs_buf=self._intermediate_buf,
        )

        # Activation
        intermediate = F.gelu(self._intermediate_buf)

        # Layer 1: GroupedGEMM + Gather + ReduceScatter
        output_shard = self._flux_rs_op.forward_gather_rs(
            intermediate,
            self._fake_fc2,
            self._splits_cpu,
            self._scatter_index.view(-1),
        )

        return output_shard.reshape(orig_shape).to(hidden_states.dtype)

    @override
    def get_split_rules(self) -> list[MatchingRule]:
        return [
            MatchingRule(condition=Op(pattern=r"moe_forward.*")),
        ]

    @override
    def get_tag_rules(self) -> dict[MatchingRule, set[str]]:
        return {
            MatchingRule(condition=Op(pattern=r"moe_forward.*")): {
                "no-inductor",
                "no-cudagraph",
                "comet",
            },
        }

    @override
    def get_split_config(
        self,
        input_info: InputInfo,
        use_cudagraph: bool,
    ) -> SplitConfig:
        prefix_sum = [0] + list(itertools.accumulate(input_info.num_tokens))
        return SplitConfig(
            num_nano_batches=1,
            batch_sizes=[input_info.batch_size],
            batch_indices=[0, input_info.batch_size],
            num_tokens=[prefix_sum[-1]],
            num_tokens_padded=[prefix_sum[-1]],
            split_indices=[0, prefix_sum[-1]],
            is_dryrun=False,
            use_cudagraph=False,
        )

    @override
    async def schedule(self, context: ExecutionContext) -> None:
        while (op := await context.pop(0)) is not None:
            if "comet" in op.tag:
                await context.execute((op,), func=self._comet_moe_call)
            else:
                await context.execute((op,))

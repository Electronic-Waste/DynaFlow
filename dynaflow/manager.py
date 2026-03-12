import asyncio
import os
import tempfile
from collections.abc import Callable
from typing import Any

import torch

from dynaflow.config import DynaFlowConfig
from dynaflow.executor.compiler import SubgraphCompiler
from dynaflow.interface import (
    ExecutionContext,
    InputInfo,
    OpSchedulerBase,
    SplitConfig,
)
from dynaflow.matching import split_graph, tag_graph
from dynaflow.runtime.engine import DynaFlowEngine


class DynaFlowManager:
    """
    DynaFlow integration manager.

    Extracts modules from FX graph and executes them with user-defined
    (programmable) scheduling.
    """

    def __init__(self) -> None:
        self.initialized = False
        self.config: DynaFlowConfig | None = None
        self.graph_module: torch.fx.GraphModule | None = None
        self.cached_config: SplitConfig | None = None
        self.scheduler: OpSchedulerBase | None = None
        self.engine: DynaFlowEngine | None = None

    def initialize(
        self,
        graph_module: torch.fx.GraphModule,
        config: DynaFlowConfig,
        scheduler: OpSchedulerBase,
        example_inputs: list[Any],
    ) -> None:
        """Initialize DynaFlow with a traced FX graph and a scheduler.

        This performs the following steps:
        - Partition the input FX graph into schedulable subgraphs using the
          scheduler-provided splitting rules and tags.
        - Prepare the execution engine which can run subgraphs under
          programmable scheduling and low-level optimizations.

        Note:
            The batch dimension of inputs in `example_inputs` should be marked
            with `torch.SymInt` (symbolic) so dynamic batch size can be
            propagated into subgraphs and recognized by backends (e.g., for
            CUDA Graph capture sets and Inductor compilation choices).

        Args:
            graph_module: A traced full-graph `torch.fx.GraphModule` to run.
            config: Global DynaFlow configuration toggles.
            scheduler: A user-provided policy implementing `OpSchedulerBase`.
            example_inputs: Example inputs used to compile/capture subgraphs.
        """
        self.initialized = True
        self.config = config
        # Note: batch-related dimensions (e.g., input_ids length, positions)
        # must be marked with torch.SymInt in example_inputs so that subgraphs
        # can discover a symbolic shape argument during tracing/interpretation.
        self.graph_module = split_graph(graph_module, scheduler.get_split_rules())
        tag_graph(self.graph_module, scheduler.get_tag_rules())
        inductor_compile_targets = (
            [
                name
                for name, module in self.graph_module.named_modules()
                if isinstance(tag := getattr(module, "tag", None), set)
                and "no-inductor" not in tag
            ]
            if config.inductor_config.enabled
            else []
        )
        cudagraph_targets = (
            [
                name
                for name, module in self.graph_module.named_modules()
                if isinstance(tag := getattr(module, "tag", None), set)
                and "no-cudagraph" not in tag
            ]
            if config.cudagraph_config.enabled
            else []
        )

        # Persist a human-readable copy of the transformed FX for debugging.
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, "gm_with_subgraphs.py")
        with open(tmp_path, "w") as f:
            f.write("# Graph\n")
            f.write(f"# Inductor Compile Targets: {inductor_compile_targets}\n")
            f.write(f"# CUDAGraph Targets: {cudagraph_targets}\n")
            f.write(self.graph_module.print_readable(print_output=False))
        print(f"Graph saved to {tmp_path}")

        SubgraphCompiler(
            self.graph_module,
            config,
            inductor_compile_targets=inductor_compile_targets,
            cudagraph_targets=cudagraph_targets,
        ).run(*example_inputs)
        self.engine = DynaFlowEngine(
            self.graph_module,
            self.config,
        )
        self.scheduler = scheduler

    def prepare(
        self,
        batch_size: int,
        num_tokens: list[int],
        is_dryrun: bool = False,
        use_cudagraph: bool = False,
    ) -> SplitConfig:
        """Prepare a `SplitConfig` for the next forward and prime scheduler.

        If `initialized` is False or `is_dryrun` is True, creates a trivial
        single-batch split. Otherwise, delegates to the active scheduler to
        compute a policy-driven split.

        Args:
            batch_size: Total batch size for the forward pass.
            num_tokens: Per-request token counts used for splitting decisions.
            is_dryrun: If True, generate config for capture/dry-run.
            use_cudagraph: Enable CUDA Graph capture targets if supported.
                Note: the batch dimension in `example_inputs` must be a
                `torch.SymInt` so the backend can identify and track the
                symbolic batch size for capture.

        Returns:
            A `SplitConfig` describing nano-batch partitioning and padding.
        """
        if not self.initialized or is_dryrun:
            self.cached_config = SplitConfig(
                num_nano_batches=1,
                batch_sizes=[batch_size],
                batch_indices=[0, batch_size],
                num_tokens=[sum(num_tokens)],
                num_tokens_padded=[sum(num_tokens)],
                split_indices=[0, sum(num_tokens)],
                is_dryrun=is_dryrun,
                use_cudagraph=use_cudagraph,
            )
        else:
            assert self.scheduler is not None
            self.cached_config = self.scheduler.get_split_config(
                InputInfo(batch_size, num_tokens, sum(num_tokens)),
                use_cudagraph=use_cudagraph,
            )
        return self.cached_config

    def override_split_config(self, split_config: SplitConfig) -> None:
        self.cached_config = split_config

    def get_callable(self) -> Callable:
        """Return a synchronous callable that executes the FX graph.

        The returned Python callable will internally orchestrate the
        asynchronous engine and scheduler, then join results to present a
        regular, synchronous forward API.
        """
        assert self.initialized

        def _forward(*args, **kwargs) -> Any:
            assert (
                self.cached_config is not None
                and self.graph_module is not None
                and self.config is not None
            )
            print(f"Executing with SplitConfig: {self.cached_config}")
            if (
                not self.cached_config.is_dryrun
                and self.cached_config.num_nano_batches == 1
            ):
                assert self.engine is not None
                result = self.engine.execute_single_batch(self.cached_config, args, kwargs)
                self.cached_config = None
                return result
            result = asyncio.run(self._forward_async(args, kwargs))
            self.cached_config = None
            return result

        return _forward

    async def _forward_async(self, args: tuple, kwargs: dict):
        """Async forward that runs the engine and scheduler concurrently.

        Creates per-nano-batch queues, spawns the engine task, and drives the
        scheduler to issue execute requests. Concatenates results across
        nano-batches while preserving tensor structure.

        Args:
            args: Positional inputs to the FX graph.
            kwargs: Keyword inputs to the FX graph.

        Returns:
            The final result (tensor or tuple of tensors) matching the
            original forward's semantics.
        """
        assert (
            self.initialized
            and self.engine is not None
            and self.scheduler is not None
            and self.cached_config is not None
        )
        num_nano_batches = self.cached_config.num_nano_batches
        # Queues for producer (engine) -> consumer (scheduler) handoff
        op_queue = {i: asyncio.Queue(maxsize=10) for i in range(num_nano_batches)}
        execute_queue = asyncio.Queue()
        context = ExecutionContext(self.cached_config, op_queue, execute_queue)

        if self.cached_config.is_dryrun:
            results_dict, events = self.engine.dryrun(
                args,
                kwargs,
                self.cached_config,
            )
        else:
            (results_dict, events), _ = await asyncio.gather(
                self.engine.execute(
                    args,
                    kwargs,
                    op_queue,
                    execute_queue,
                    self.cached_config,
                ),
                self.scheduler.schedule(context),
            )
        for event in events:
            event.wait()
        assert all(
            isinstance(e, type(results_dict[0])) for e in results_dict.values()
        ), f"Results have different types: {results_dict}"
        if isinstance(results_dict[0], torch.Tensor):
            # Concatenate per-nano-batch outputs along batch dimension
            return torch.cat(
                [results_dict[idx] for idx in range(num_nano_batches)], dim=0
            )
        elif isinstance(results_dict[0], tuple):
            num_elements = len(results_dict[0])
            assert all(
                len(r) == num_elements for r in results_dict.values()
            ), f"Results have different number of elements: {results_dict}"
            concatenated = []
            for i in range(num_elements):
                elements = [results_dict[idx][i] for idx in range(num_nano_batches)]
                assert all(
                    isinstance(e, torch.Tensor) for e in elements
                ), f"Elements are not tensors: {elements}"
                # Concatenate each tuple field across nano-batches
                concatenated.append(torch.cat(elements, dim=0))
            return tuple(concatenated)
        else:
            return results_dict[0]

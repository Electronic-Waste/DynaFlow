import asyncio
from collections import deque
from collections.abc import Callable
from typing import Any

import torch

from dynaflow.config import DynaFlowConfig
from dynaflow.executor import SubgraphBackend
from dynaflow.interface import OperatorHandle, SplitConfig
from dynaflow.runtime.env import ExecutionEnvironment, set_op_handle


class DynaFlowEngine:
    """Engine that executes FX graph with programmable operator scheduling.

    High-level model:
    - Producer: walks each nano-batch's node list; executes non-module nodes
      immediately and enqueues ready `call_module` operators to `op_queue`.
    - Consumer (scheduler-driven): pulls ready operators and issues execute
      requests via `execute_queue`, optionally grouping multiple operators.
    - Engine executes requested operators, publishes results into the
      per-nano-batch `ExecutionEnvironment`, and signals completion.

    Notes:
    - `dryrun()` primes Inductor/CUDA Graphs without involving the scheduler.
    - `execute()` runs the full producer/consumer loop across nano-batches.
    - Non-module ops (call_function/method/get_attr/output) are handled by the
      engine and never exposed to the scheduler.
    """

    def __init__(self, graph_module: torch.fx.GraphModule, config: DynaFlowConfig):
        """Construct the engine.

        Builds helper indices for placeholders, attributes, and module nodes
        to speed up execution.
        """
        self.graph_module = graph_module
        self.config = config
        self.input_buffers = [{}] * config.max_num_splits

        # Map submodule names -> FX nodes (used to re-materialize arguments)
        self.module_name_to_node: dict[str, torch.fx.Node] = {}
        for node in self.graph_module.graph.nodes:
            if node.op == "call_module":
                assert isinstance(node.target, str)
                self.module_name_to_node[node.target] = node

        # Ordered placeholders and index map for quick access
        self.placeholder_nodes: list[torch.fx.Node] = [
            n for n in self.graph_module.graph.nodes if n.op == "placeholder"
        ]
        self.placeholder_node_to_idx: dict[torch.fx.Node, int] = {
            node: idx for idx, node in enumerate(self.placeholder_nodes)
        }
        # Cache static attributes for fast get_attr execution
        self.get_attr_cache: dict[str, Any] = {}
        for node in self.graph_module.graph.nodes:
            if node.op == "get_attr":
                assert isinstance(node.target, str)
                self.get_attr_cache[node.target] = getattr(
                    self.graph_module, node.target
                )

    def allocate_input_buffer(
        self,
        batch_idx: int,
        placeholder_idx: int,
        num_tokens: int,
        example_value: torch.Tensor,
    ) -> None:
        assert self.input_buffers is not None
        if self.input_buffers[batch_idx].get(placeholder_idx) is None:
            # allocate a large static buffer for max token length
            self.input_buffers[batch_idx][placeholder_idx] = torch.zeros(
                (num_tokens,) + example_value.shape[1:],
                dtype=example_value.dtype,
                device=example_value.device,
            )

    def dryrun(
        self,
        args: tuple,
        kwargs: dict,
        split_config: SplitConfig,
    ) -> tuple[dict[int, Any], list[torch.cuda.Event]]:
        # Single-nano-batch priming path (optionally simulate >1 when capturing)
        assert split_config.num_nano_batches == 1 and split_config.is_dryrun
        use_fake_nano_batches = split_config.use_cudagraph
        num_nano_batches = (
            self.config.max_num_splits if use_fake_nano_batches else 1
        )

        results: dict[int, Any] = {}
        env = ExecutionEnvironment(num_nano_batches)
        num_tokens = split_config.num_tokens_padded[0]
        last_events = [torch.cuda.Event() for _ in range(num_nano_batches)]
        # Each fake nano-batch iterates the same ordered node list
        node_queue: dict[int, deque[torch.fx.Node]] = {
            i: deque(self.graph_module.graph.nodes) for i in range(num_nano_batches)
        }

        for batch_idx in range(num_nano_batches):
            while node_queue[batch_idx]:
                node = node_queue[batch_idx][0]
                node_queue[batch_idx].popleft()
                if node.op == "placeholder":
                    placeholder_idx = self.placeholder_node_to_idx[node]
                    example_value = node.meta.get("example_value", None)
                    if isinstance(example_value, torch.SymInt):
                        assert args[placeholder_idx] == num_tokens
                        env.put(batch_idx, node, num_tokens)
                        continue
                    assert isinstance(example_value, torch.Tensor)
                    if not isinstance(example_value.shape[0], torch.SymInt):
                        # forward the original arg for parameters
                        env.put(batch_idx, node, args[placeholder_idx])
                        continue
                    if not split_config.use_cudagraph:
                        # normal execution
                        env.put(batch_idx, node, args[placeholder_idx])
                        continue
                    # CUDA graph capture
                    self.allocate_input_buffer(
                        batch_idx, placeholder_idx, 16384, example_value
                    )
                    self.input_buffers[batch_idx][placeholder_idx][:num_tokens].copy_(
                        args[placeholder_idx][:num_tokens]
                    )
                    env.put(
                        batch_idx,
                        node,
                        self.input_buffers[batch_idx][placeholder_idx][:num_tokens],
                    )
                elif node.op == "call_module":
                    assert isinstance(node.target, str)
                    module = getattr(self.graph_module, node.target)
                    assert isinstance(module, SubgraphBackend)
                    assert isinstance(module.tag, set)
                    node_args = [
                        env.get(batch_idx, arg)
                        if isinstance(arg, torch.fx.Node)
                        else arg
                        for arg in node.args
                    ]
                    node_kwargs = {
                        k: env.get(batch_idx, v) if isinstance(v, torch.fx.Node) else v
                        for k, v in node.kwargs.items()
                    }
                    with set_op_handle(
                        OperatorHandle(
                            module_name=node.target,
                            batch_idx=batch_idx,
                            batch_size=num_tokens,
                            tag=module.tag,
                            _is_dryrun=split_config.is_dryrun,
                            _use_cudagraph=split_config.use_cudagraph,
                        )
                    ):
                        env.put(batch_idx, node, module(*node_args, **node_kwargs))
                elif node.op == "output":
                    if isinstance(node.args[0], torch.fx.Node):
                        results[batch_idx] = env.get(batch_idx, node.args[0])
                    elif isinstance(node.args[0], tuple | list):
                        results[batch_idx] = tuple(
                            env.get(batch_idx, arg)
                            if isinstance(arg, torch.fx.Node)
                            else arg
                            for arg in node.args[0]
                        )
                    else:
                        results[batch_idx] = node.args[0]
                else:
                    self._execute_non_module(node, batch_idx, args, env, split_config)
            last_events[batch_idx].record()

        return {0: results[0]}, last_events

    def execute_single_batch(
        self,
        split_config: SplitConfig,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """Synchronous single-nano-batch execution without scheduler interaction.

        Copies dynamic input tensors into pre-allocated static buffers (the same
        tensor objects used during CUDA graph capture), then calls graph_module
        directly. This keeps CUDA graph replay correct with minimal CPU overhead.
        """
        batch_idx = 0
        num_tokens_padded = split_config.num_tokens_padded[batch_idx]

        new_args = list(args)
        for node in self.placeholder_nodes:
            placeholder_idx = self.placeholder_node_to_idx[node]
            example_value = node.meta.get("example_value", None)
            if isinstance(example_value, torch.SymInt):
                new_args[placeholder_idx] = num_tokens_padded
            elif (
                isinstance(example_value, torch.Tensor)
                and isinstance(example_value.shape[0], torch.SymInt)
                and split_config.use_cudagraph
            ):
                buf = self.input_buffers[batch_idx][placeholder_idx]
                buf[:num_tokens_padded].copy_(args[placeholder_idx])
                new_args[placeholder_idx] = buf[:num_tokens_padded]

        with set_op_handle(
            OperatorHandle(
                module_name=self.graph_module.__class__.__name__,
                batch_idx=batch_idx,
                batch_size=num_tokens_padded,
                tag=set(),
                _is_dryrun=False,
                _use_cudagraph=split_config.use_cudagraph,
            )
        ):
            return self.graph_module(*new_args, **kwargs)

    async def execute(
        self,
        args: tuple,
        kwargs: dict,
        op_queue: dict[int, asyncio.Queue[OperatorHandle | None]],
        execute_queue: asyncio.Queue[
            tuple[
                tuple[OperatorHandle, ...],
                Callable | None,
                asyncio.Event,
            ]
        ],
        split_config: SplitConfig,
    ) -> tuple[dict[int, Any], list[torch.cuda.Event]]:
        """Execute the model with simplified single-loop pattern.

        Args:
            args: Input arguments
            kwargs: Input keyword arguments
            op_queue: Queues to push ready operators (one per nano-batch)
            execute_queue: Queue to receive execution requests from scheduler
            split_config: Configuration for nano-batch splitting
            hook: Optional hook for operator execution

        Returns:
            Dictionary mapping nano-batch index to results
        """
        num_nano_batches = split_config.num_nano_batches

        env = ExecutionEnvironment(num_nano_batches)

        results: dict[int, Any] = {}
        last_events = [torch.cuda.Event() for _ in range(num_nano_batches)]
        node_queue: dict[int, deque[torch.fx.Node]] = {
            i: deque(self.graph_module.graph.nodes) for i in range(num_nano_batches)
        }

        for batch_idx in range(num_nano_batches):
            while node_queue[batch_idx]:
                node = node_queue[batch_idx][0]
                if node.op == "call_module" or node.op == "output":
                    break
                self._execute_non_module(node, batch_idx, args, env, split_config)
                node_queue[batch_idx].popleft()
            last_events[batch_idx].record()

        pushed_operators: dict[int, list[OperatorHandle]] = {
            i: [] for i in range(num_nano_batches)
        }

        while any(node_queue.values()):
            for batch_idx in range(num_nano_batches):
                while node_queue[batch_idx]:
                    if op_queue[batch_idx].full():
                        break
                    node = node_queue[batch_idx][0]
                    if node.op != "call_module":
                        break
                    assert isinstance(node.target, str)
                    module = getattr(self.graph_module, node.target)
                    assert isinstance(module.tag, set)
                    op_handle = OperatorHandle(
                        module_name=node.target,
                        batch_idx=batch_idx,
                        batch_size=split_config.num_tokens_padded[batch_idx],
                        tag=module.tag,
                        _is_dryrun=split_config.is_dryrun,
                        _use_cudagraph=split_config.use_cudagraph,
                    )
                    await op_queue[batch_idx].put(op_handle)
                    pushed_operators[batch_idx].append(op_handle)
                    node_queue[batch_idx].popleft()

            item = await execute_queue.get()
            operators, func, done_event = item
            node_args = []
            node_kwargs = []
            for op in operators:
                batch_idx = op.batch_idx
                node = self.module_name_to_node[op.module_name]
                last_events[batch_idx].wait()
                last_events[batch_idx] = torch.cuda.Event()
                node_args.append(
                    [
                        env.get(batch_idx, arg)
                        for arg in node.args
                        if isinstance(arg, torch.fx.Node)
                    ]
                )
                node_kwargs.append(
                    {
                        k: env.get(batch_idx, v) if isinstance(v, torch.fx.Node) else v
                        for k, v in node.kwargs.items()
                    }
                )
            node_args = tuple(node_args)
            node_kwargs = tuple(node_kwargs)
            exec_results = []
            if func is not None:
                if len(operators) != 1:
                    raise NotImplementedError("Operator batching is not implemented")
                op = operators[0]
                with set_op_handle(op):
                    exec_results.append(func(*node_args[0], **node_kwargs[0]))
            elif len(operators) != 1 and all(
                op.module_name == operators[0].module_name for op in operators
            ):
                raise NotImplementedError("Operator batching is not implemented")
                assert (
                    not split_config.use_cudagraph
                ), "CUDA graph is not supported for operator batching"
                # module_name = operators[0].module_name
                # batch_indices = tuple(op.batch_idx for op in operators)
                with (
                    torch.cuda.nvtx.range(
                        f"op_{operators[0].module_name}_"
                        f"({','.join(str(op.batch_idx) for op in operators)})"
                    ),
                    set_forward_context(
                        DynaFlowContext(
                            batch_idx=tuple(op.batch_idx for op in operators),
                            num_tokens_padded=tuple(
                                split_config.num_tokens_padded[op.batch_idx]
                                for op in operators
                            ),
                            is_dryrun=split_config.is_dryrun,
                            use_cudagraph=False,
                        )
                    ),
                ):
                    module = getattr(self.graph_module, operators[0].module_name)
                    args: list[torch.Tensor | int] = []
                    for args_combined in zip(*node_args):
                        if isinstance(args_combined[0], int):
                            args.append(sum(args_combined))
                        elif all(args_combined[0] == arg for arg in args_combined):
                            args.append(args_combined[0])
                        else:
                            args.append(torch.cat(args_combined, dim=0))
                    kwargs: dict[str, torch.Tensor | int] = {}
                    for key, values_combined in zip(*node_kwargs):
                        if isinstance(values_combined[0], int):
                            kwargs[key] = sum(values_combined)
                        elif all(
                            values_combined[0] == value for value in values_combined
                        ):
                            kwargs[key] = values_combined[0]
                        else:
                            kwargs[key] = torch.cat(values_combined, dim=0)
                    exec_results.append(module(tuple(args), kwargs))
            else:
                for idx, op in enumerate(operators):
                    with (
                        torch.cuda.nvtx.range(
                            f"op_{op.module_name}_{op.batch_idx}"
                        ),
                        set_op_handle(op),
                    ):
                        module = getattr(self.graph_module, op.module_name)
                        exec_results.append(module(*node_args[idx], **node_kwargs[idx]))

            for op, result in zip(operators, exec_results):
                batch_idx = op.batch_idx
                node = self.module_name_to_node[op.module_name]
                last_events[batch_idx].record()
                env.put(batch_idx, node, result)
                assert op == pushed_operators[batch_idx][0]
                pushed_operators[batch_idx].pop(0)

            for batch_idx in range(num_nano_batches):
                if pushed_operators[batch_idx]:
                    continue
                while node_queue[batch_idx]:
                    node = node_queue[batch_idx][0]
                    if node.op == "call_module":
                        break
                    elif node.op == "output":
                        if isinstance(node.args[0], torch.fx.Node):
                            results[batch_idx] = env.get(batch_idx, node.args[0])
                        elif isinstance(node.args[0], tuple | list):
                            results[batch_idx] = tuple(
                                env.get(batch_idx, arg)
                                if isinstance(arg, torch.fx.Node)
                                else arg
                                for arg in node.args[0]
                            )
                        else:
                            results[batch_idx] = node.args[0]
                        node_queue[batch_idx].popleft()
                        await op_queue[batch_idx].put(None)
                        break
                    else:
                        self._execute_non_module(
                            node,
                            batch_idx,
                            args,
                            env,
                            split_config,
                        )
                        node_queue[batch_idx].popleft()
            done_event.set()

        return results, last_events

    def _execute_non_module(
        self,
        node: torch.fx.Node,
        batch_idx: int,
        args: tuple,
        env: ExecutionEnvironment,
        split_config: SplitConfig,
    ) -> None:
        """Execute non-module operations (placeholder, get_attr, call_function,
        call_method)."""
        if node.op == "placeholder":
            placeholder_idx = self.placeholder_node_to_idx[node]

            example_value = node.meta.get("example_value", None)

            token_start = split_config.split_indices[batch_idx]
            token_end = split_config.split_indices[batch_idx + 1]
            num_tokens = split_config.num_tokens[batch_idx]
            num_tokens_padded = split_config.num_tokens_padded[batch_idx]
            total_num_tokens_padded = sum(split_config.num_tokens_padded)

            if isinstance(example_value, torch.Tensor):
                if isinstance(example_value.shape[0], torch.SymInt):
                    if split_config.use_cudagraph:
                        print(f"Copying input tensor of shape {args[placeholder_idx].shape} into static buffer for batch {batch_idx}, placeholder {placeholder_idx}, num_tokens {num_tokens}")
                        import sys
                        sys.stdout.flush()
                        assert self.input_buffers is not None
                        assert (
                            self.input_buffers[batch_idx].get(placeholder_idx)
                            is not None
                        )
                        self.input_buffers[batch_idx][placeholder_idx][
                            :num_tokens
                        ].copy_(args[placeholder_idx][token_start:token_end])
                        env.put(
                            batch_idx,
                            node,
                            self.input_buffers[batch_idx][placeholder_idx][
                                :num_tokens_padded
                            ],
                        )
                    else:
                        env.put(
                            batch_idx,
                            node,
                            args[placeholder_idx][token_start:token_end],
                        )
                else:
                    env.put(batch_idx, node, args[placeholder_idx])
            elif isinstance(example_value, torch.SymInt):
                assert args[placeholder_idx] == total_num_tokens_padded
                env.put(batch_idx, node, num_tokens_padded)
            else:
                raise ValueError(f"Invalid example value type: {type(example_value)}")
        elif node.op == "call_function":
            target = node.target
            assert callable(target)
            node_args = [
                env.get(batch_idx, arg) if isinstance(arg, torch.fx.Node) else arg
                for arg in node.args
            ]
            node_kwargs = {
                k: env.get(batch_idx, v) if isinstance(v, torch.fx.Node) else v
                for k, v in node.kwargs.items()
            }
            env.put(batch_idx, node, target(*node_args, **node_kwargs))
        elif node.op == "call_method":
            assert isinstance(node.target, str)
            self_obj = env.get(batch_idx, node.args[0])  # type: ignore
            target = getattr(self_obj, node.target)
            assert callable(target)
            method_args = [
                env.get(batch_idx, arg) if isinstance(arg, torch.fx.Node) else arg
                for arg in node.args[1:]
            ]
            node_kwargs = {
                k: env.get(batch_idx, v) if isinstance(v, torch.fx.Node) else v
                for k, v in node.kwargs.items()
            }
            env.put(batch_idx, node, target(*method_args, **node_kwargs))
        elif node.op == "get_attr":
            assert isinstance(node.target, str)
            env.put(batch_idx, node, self.get_attr_cache[node.target])
        else:
            raise ValueError(f"Invalid node operation: {node.op}")

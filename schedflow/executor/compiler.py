from collections.abc import Callable
from typing import Any

import torch
import torch.fx as fx
from typing_extensions import override

from schedflow.config import (
    CUDAGraphConfig,
    InductorConfig,
    SchedFlowConfig,
)
from schedflow.context import get_forward_context
from schedflow.executor.cudagraph import CUDAGraphWrapper
from schedflow.executor.inductor import inductor_compile_fx_adaptor_style


class SubgraphBackend:
    """Backend wrapper for a single FX subgraph module.

    Optionally compiles with TorchInductor and/or wraps with CUDA Graph
    capture. Dispatches calls based on dynamic shape and policy.
    """

    def __init__(
        self,
        subgraph_gm: fx.GraphModule,
        shape_arg_index: int,
        inductor_config: InductorConfig,
        cudagraph_config: CUDAGraphConfig,
        *,
        use_inductor: bool,
        use_cudagraph: bool,
    ) -> None:
        self.gm = subgraph_gm
        self.tag = getattr(subgraph_gm, "tag", set())
        assert isinstance(self.tag, set)
        self.inductor_config = inductor_config
        self.cudagraph_config = cudagraph_config
        self.shape_arg_index = shape_arg_index
        assert not use_inductor or inductor_config.enabled
        assert not use_cudagraph or cudagraph_config.enabled
        self.use_inductor = use_inductor
        self.use_cudagraph = use_cudagraph
        self.callable_dynamic: Callable[..., Any] | None = None
        self.callables_per_size: dict[int, Callable[..., Any]] = {}

    def compile(
        self, example_inputs: list[Any], runtime_shape: int | None = None
    ) -> Callable[..., Any]:
        """Compile or wrap the subgraph according to backend configuration."""
        shape = example_inputs[self.shape_arg_index]
        assert (isinstance(shape, torch.SymInt) and runtime_shape is None) or (
            int(shape) == runtime_shape
        )
        fn = (
            inductor_compile_fx_adaptor_style(
                self.gm,
                example_inputs,
                inductor_cfg=self.inductor_config,
                runtime_shape=runtime_shape,
            )
            if self.use_inductor
            else self.gm
        )
        return CUDAGraphWrapper(fn, self.cudagraph_config) if self.use_cudagraph else fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the appropriate compiled or dynamic callable for inputs."""
        assert self.shape_arg_index < len(args)
        size = int(args[self.shape_arg_index])
        if (
            size is not None
            and self.inductor_config.compile_sizes is not None
            and size in self.inductor_config.compile_sizes
        ):
            if size not in self.callables_per_size:
                assert get_forward_context().is_dryrun
                self.callables_per_size[size] = self.compile(list(args), size)
            return self.callables_per_size[size](*args, **kwargs)
        assert self.callable_dynamic is not None
        return self.callable_dynamic(*args, **kwargs)


class SubgraphCompiler(fx.Interpreter):
    """Interpreter that replaces submodules with backend-wrapped callables."""

    def __init__(
        self,
        module: fx.GraphModule,
        config: SchedFlowConfig,
        *,
        inductor_compile_targets: list[str],
        cudagraph_targets: list[str],
    ) -> None:
        super().__init__(module)
        self.config = config
        self.inductor_compile_targets = set(inductor_compile_targets)
        self.cudagraph_targets = set(cudagraph_targets)
        from torch._guards import detect_fake_mode

        self.fake_mode = detect_fake_mode()

    @override
    def run(
        self,
        *args: Any,
        initial_env: dict[fx.Node, Any] | None = None,
        enable_io_processing: bool = True,
    ) -> Any:
        """Execute in fake mode to discover dynamic shape and patch modules."""
        assert self.fake_mode is not None
        fake_args = [
            self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in args
        ]
        from torch._dispatch.python import enable_python_dispatcher

        with self.fake_mode, enable_python_dispatcher():
            return super().run(*fake_args)

    @override
    def call_module(self, target, args: tuple, kwargs: dict) -> Any:
        """On module call, patch the module with a `SubgraphBackend`."""
        assert isinstance(target, str)
        out = super().call_module(target, args, kwargs)
        submod = self.fetch_attr(target)
        sym_shape_indices = [
            i for i, x in enumerate(args) if isinstance(x, torch.SymInt)
        ]
        # At least one argument in example inputs (e.g., batch dimension of
        # input_ids/positions) must be `torch.SymInt` so backends can treat it
        # as the dynamic batch shape for compilation/capture.
        assert len(sym_shape_indices) > 0
        primary_shape_index: int = sym_shape_indices[0]
        assert isinstance(submod, fx.GraphModule)
        backend = SubgraphBackend(
            submod,
            primary_shape_index,
            self.config.inductor_config,
            self.config.cudagraph_config,
            use_inductor=target in self.inductor_compile_targets,
            use_cudagraph=target in self.cudagraph_targets,
        )
        backend.callable_dynamic = backend.compile(list(args))
        self.module.__dict__[target] = backend
        return out

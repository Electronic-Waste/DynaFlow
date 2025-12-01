from collections.abc import Callable
from typing import Any

import torch

from dynaflow.config import CUDAGraphConfig
from dynaflow.context import get_forward_context


class CUDAGraphPool:
    def __init__(self):
        self.pools: dict[int, tuple[int, int]] = {}

    def get_pool(self, key: int) -> tuple[int, int]:
        # NOTE(yi): the following lines are for vLLM only.
        # Change this in other systems to set the graph pool for NCCL
        from vllm.distributed.device_communicators.pynccl_allocator import (
            set_graph_pool_id,
        )

        if key not in self.pools:
            # Create a new CUDA graph pool for this nano-batch key
            self.pools[key] = torch.cuda.graph_pool_handle()
        pool = self.pools[key]
        # Ensure NCCL uses the same pool during capture/replay
        set_graph_pool_id(pool)
        return pool


_global_pool = CUDAGraphPool()


def _weak_ref_tensor(tensor: Any) -> Any:
    """Create a weak reference to a tensor via torch custom op if available."""
    if isinstance(tensor, torch.Tensor):
        return torch.ops._C.weak_ref_tensor(tensor)  # type: ignore[attr-defined]
    return tensor


def _weak_ref_tensors(
    tensors: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor] | Any,
) -> torch.Tensor | list[Any] | tuple[Any] | Any:
    """Apply weak refs to tensors, lists, tuples, and IntermediateTensors."""
    if isinstance(tensors, torch.Tensor):
        return _weak_ref_tensor(tensors)
    if isinstance(tensors, list):
        return [_weak_ref_tensor(t) for t in tensors]
    if isinstance(tensors, tuple):
        return tuple(_weak_ref_tensor(t) for t in tensors)
    if (
        hasattr(tensors, "tensors")
        and tensors.__class__.__name__ == "IntermediateTensors"
    ):
        inner = tensors.tensors
        if isinstance(inner, dict):
            new_inner = {k: _weak_ref_tensor(v) for k, v in inner.items()}
            try:
                return tensors.__class__(new_inner)
            except Exception:
                pass
    return tensors


class CUDAGraphWrapper:
    """Wrap a callable to optionally capture and replay with CUDA Graphs.

    Capture occurs during a dry-run under `set_forward_context(...)`; replay
    is keyed by nano-batch and padded token size to ensure correctness.
    """

    def __init__(
        self,
        runnable: Callable[..., Any],
        config: CUDAGraphConfig,
    ):
        self.runnable = runnable
        self.config = config
        self._entries: dict[tuple[Any, ...], dict[str, Any]] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute or replay the captured CUDA Graph for matching inputs."""
        forward_context = get_forward_context()
        size = forward_context.num_tokens_padded[0]
        if (
            not forward_context.use_cudagraph
            or len(forward_context.nano_batch_idx) != 1
            or size not in self.config.capture_sizes
        ):
            # Not capturing or size not selected: run eagerly
            return self.runnable(*args, **kwargs)

        key = (
            size,
            forward_context.nano_batch_idx[0],
            forward_context.num_tokens_padded,
        )
        entry = self._entries.get(key)
        if entry is None:
            assert forward_context.is_dryrun
            pool = _global_pool.get_pool(forward_context.nano_batch_idx[0])
            cudagraph = torch.cuda.CUDAGraph()
            input_addresses = [
                a.data_ptr() for a in args if isinstance(a, torch.Tensor)
            ]
            with torch.cuda.graph(cudagraph, pool=pool):
                out = self.runnable(*args, **kwargs)
                result = out
            self._entries[key] = {
                "graph": cudagraph,
                "out": _weak_ref_tensors(out),
                "inputs": input_addresses,
            }
            return result

        if self.config.check_ptr_consistency:
            new_addrs = [a.data_ptr() for a in args if isinstance(a, torch.Tensor)]
            if new_addrs != entry.get("inputs", new_addrs):
                raise RuntimeError(
                    "CUDAGraph input addresses changed between capture and " "replay"
                )
        entry["graph"].replay()
        # Replay returns buffered outputs (weak-ref'ed) captured earlier
        return entry["out"]

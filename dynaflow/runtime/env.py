from typing import Any

import torch
from collections.abc import Generator
from contextlib import contextmanager

from dynaflow.interface import OperatorHandle


class ActivationBuffer:
    """Reference-counted holder for node outputs within a nano-batch."""

    def __init__(self, data: Any, ref_count: int):
        self.data = data
        self.ref_count = ref_count

    def get_ref(self) -> Any:
        self.ref_count -= 1
        data = self.data
        if self.ref_count == 0:
            self.data = None
        return data


class ExecutionEnvironment:
    """Per-nano-batch storage for intermediate values and arguments."""

    def __init__(self, num_nano_batches: int):
        self.env: list[dict[torch.fx.Node, ActivationBuffer]] = [
            {} for _ in range(num_nano_batches)
        ]

    def put(
        self,
        batch_idx: int,
        node: torch.fx.Node,
        data: Any,
    ) -> None:
        """Store a node's result alongside a reference count of its users."""
        self.env[batch_idx][node] = ActivationBuffer(data, len(node.users))

    def get(self, batch_idx: int, node: torch.fx.Node) -> Any:
        """Retrieve and decrement the reference count; clear when last use."""
        buffer = self.env[batch_idx][node].get_ref()
        assert buffer is not None
        return buffer

_current_op_handle: OperatorHandle | tuple[OperatorHandle] | None = None

def get_op_handle() -> OperatorHandle | tuple[OperatorHandle]:
    """Return the current operator handle.

    Must be used inside regions established by `set_operator_handle`.
    """
    assert _current_op_handle is not None
    return _current_op_handle


@contextmanager
def set_op_handle(
    handle: OperatorHandle | tuple[OperatorHandle]
) -> Generator[None, None, None]:
    """Set the current operator handle for the duration of the context."""
    global _current_op_handle
    prev_handle = _current_op_handle
    _current_op_handle = handle
    try:
        yield
    finally:
        _current_op_handle = prev_handle

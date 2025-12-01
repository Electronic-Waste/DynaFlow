from typing import Any

import torch


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
        nano_batch_idx: int,
        node: torch.fx.Node,
        data: Any,
    ) -> None:
        """Store a node's result alongside a reference count of its users."""
        self.env[nano_batch_idx][node] = ActivationBuffer(data, len(node.users))

    def get(self, nano_batch_idx: int, node: torch.fx.Node) -> Any:
        """Retrieve and decrement the reference count; clear when last use."""
        buffer = self.env[nano_batch_idx][node].get_ref()
        assert buffer is not None
        return buffer

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class SchedFlowContext:
    """Per-call execution context visible to backends and schedulers.

    For CUDA Graph capture and replay, and for policy decisions that depend on
    nano-batch indices and padded token counts.
    """

    nano_batch_idx: tuple[int, ...]
    """The index of the nano-batch."""
    num_tokens_padded: tuple[int, ...]
    """The number of tokens in the padded nano-batch."""
    is_dryrun: bool
    """Whether this is a dry run."""
    use_cudagraph: bool
    """Whether to use CUDA graph."""


_forward_context: SchedFlowContext | None = None


def get_forward_context() -> SchedFlowContext:
    """Return the current forward execution context.

    Must be used inside regions established by `set_forward_context`.
    """
    assert _forward_context is not None
    return _forward_context


@contextmanager
def set_forward_context(
    context: SchedFlowContext,
) -> Generator[None, None, None]:
    """Install a forward context for the dynamic extent of the with-block."""
    global _forward_context
    prev_context = _forward_context
    _forward_context = context
    try:
        yield
    finally:
        _forward_context = prev_context

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CUDAGraphConfig:
    enabled: bool
    capture_sizes: list[int]
    weak_ref_output: bool = True
    check_ptr_consistency: bool = False


@dataclass
class InductorConfig:
    enabled: bool
    compile_sizes: set[int] | None = None
    options: dict | None = None
    disable_remote_cache: bool = True
    disable_autograd_cache: bool = True


@dataclass
class DynaFlowConfig:
    max_num_nano_batches: int
    min_nano_split_tokens: int
    inductor_config: InductorConfig
    cudagraph_config: CUDAGraphConfig

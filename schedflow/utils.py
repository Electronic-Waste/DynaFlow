import os
import tempfile
from typing import Any

import torch

from schedflow.backend import SubgraphCompileInterpreter
from schedflow.config import SchedFlowConfig


def split_graph(
    graph: torch.fx.GraphModule, splitting_ops: list[str]
) -> torch.fx.GraphModule:
    """Partition a traced full-graph into schedulable subgraphs.

    Creates a new stitched `GraphModule` where top-level submodules
    (`submod_0`, `submod_1`, ...) correspond to segments delimited by
    `splitting_ops` boundaries.
    """
    node_to_subgraph_id: dict[torch.fx.Node, int] = {}
    subgraph_id = 0
    split_op_graphs: list[int] = []

    for node in graph.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue
        if node.op == "call_function" and str(node.target) in splitting_ops:
            # Insert a split boundary before and after this op
            subgraph_id += 1
            node_to_subgraph_id[node] = subgraph_id
            split_op_graphs.append(subgraph_id)
            subgraph_id += 1
        else:
            node_to_subgraph_id[node] = subgraph_id

    split_gm = torch.fx.passes.split_module.split_module(  # type: ignore[attr-defined]
        graph,
        None,
        lambda n: node_to_subgraph_id.get(n, 0),
        keep_original_order=True,
    )

    return split_gm


def tag_graph(gm: torch.fx.GraphModule, op_tags: dict[str, set[str]]) -> None:
    """Annotate submodules with tag sets derived from contained ops.

    Tags inform backends (e.g., CUDA Graphs, Inductor) about constraints or
    capabilities within each subgraph.
    """
    submodules = [
        (name, module)
        for (name, module) in gm.named_modules()
        if hasattr(module, "graph")
    ]
    for name, module in submodules:
        if "." in name or name == "":
            continue
        module.tag = set()  # type: ignore[assignment]  # tags guide backend decisions
        for node in module.graph.nodes:
            if (
                node.op == "call_function"
                and (tag := op_tags.get(str(node.target))) is not None
            ):
                module.tag.update(tag)


def compile_subgraphs(
    fullgraph: torch.fx.GraphModule,
    config: SchedFlowConfig,
    example_inputs: list[Any],
    splitting_ops: list[str],
    op_tags: dict[str, set[str]],
) -> torch.fx.GraphModule:
    """Split, tag, and compile subgraphs with selected backends.

    Subgraphs are optionally compiled by TorchInductor and/or wrapped with
    CUDA Graph capture depending on tags and global config.
    """
    # Note: users must mark batch-related dimensions (e.g., input_ids length,
    # positions) with torch.SymInt in example_inputs so that subgraphs can
    # discover a symbolic shape argument during tracing/interpretation.
    gm_with_subgraphs = split_graph(fullgraph, splitting_ops)
    tag_graph(gm_with_subgraphs, op_tags)
    # MoE operators cannot be compiled with Inductor because their shape cannot
    # be inferred statically
    inductor_compile_targets = (
        [
            name
            for name, module in gm_with_subgraphs.named_modules()
            if isinstance(tag := getattr(module, "tag", None), set)
            and "no-inductor" not in tag
        ]
        if config.inductor_config.enabled
        else []
    )
    # Attention operators cannot be captured by CUDAGraph
    cudagraph_targets = (
        [
            name
            for name, module in gm_with_subgraphs.named_modules()
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
        f.write(gm_with_subgraphs.print_readable(print_output=False))
    print(f"Graph saved to {tmp_path}")

    SubgraphCompileInterpreter(
        gm_with_subgraphs,
        config,
        inductor_compile_targets=inductor_compile_targets,
        cudagraph_targets=cudagraph_targets,
    ).run(*example_inputs)
    return gm_with_subgraphs


def pack_tokens(num_tokens: int, cudagraph_capture_sizes: list[int]) -> int:
    """Pad token count up to the next capture size when using CUDA Graphs."""
    if num_tokens <= max(cudagraph_capture_sizes):
        return min(size for size in cudagraph_capture_sizes if size >= num_tokens)
    return num_tokens

import re
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Op:
    pattern: str

    def matches(self, node: torch.fx.Node) -> bool:
        if node.op != "call_function" or not callable(node.target):
            return False
        return bool(re.fullmatch(self.pattern, node.target.__name__))


@dataclass(frozen=True)
class Mod:
    target_cls: type[torch.nn.Module]

    @classmethod
    def get_module_info(
        cls, node: torch.fx.Node
    ) -> tuple[str, type[torch.nn.Module]] | None:
        if "nn_module_stack" not in node.meta:
            return None
        module_list = list(node.meta["nn_module_stack"].values())
        module_list.sort(key=lambda x: x[0])
        module_name, module_cls = module_list[-1]
        return module_name, module_cls

    def matches(self, node_list: list[torch.fx.Node]) -> int:
        last_module = None
        current_idx = 0
        while current_idx < len(node_list):
            node = node_list[current_idx]
            module_info = self.get_module_info(node)
            if module_info is None:
                return current_idx
            module_name, module_cls = module_info
            if module_cls is not self.target_cls:
                return current_idx
            if last_module is None:
                last_module = module_name
            elif last_module != module_name:
                return current_idx
            current_idx += 1
        return current_idx


@dataclass(frozen=True)
class MatchingRule:
    condition: Op | Mod | tuple[Op | Mod, ...]

    def matches(self, node_list: list[torch.fx.Node]) -> int:
        condition_list = (
            [self.condition]
            if isinstance(self.condition, (Op | Mod))
            else self.condition
        )
        node_idx = 0
        condition_idx = 0
        while node_idx < len(node_list) and condition_idx < len(condition_list):
            cond = condition_list[condition_idx]
            if isinstance(cond, Op):
                if not cond.matches(node_list[node_idx]):
                    return 0
                node_idx += 1
            else:
                matched_idx = cond.matches(node_list[node_idx:])
                if matched_idx == 0:
                    return 0
                node_idx += matched_idx
            condition_idx += 1

        return node_idx


def split_graph(
    graph: torch.fx.GraphModule, split_rules: list[MatchingRule]
) -> torch.fx.GraphModule:
    """Partition a full-graph into schedulable subgraphs with cut-around rules."""
    nodes = list(graph.graph.nodes)

    node_to_subgraph_id: dict[torch.fx.Node, int] = {}
    subgraph_id = 0
    node_idx = 0

    while node_idx < len(nodes):
        matched = False
        for rule in split_rules:
            matched_idx = rule.matches(nodes[node_idx:])
            if matched_idx > 0:
                subgraph_id += 1
                for i in range(node_idx, node_idx + matched_idx):
                    node_to_subgraph_id[nodes[i]] = subgraph_id
                node_idx += matched_idx
                subgraph_id += 1
                matched = True
                break
        if not matched:
            node_to_subgraph_id[nodes[node_idx]] = subgraph_id
            node_idx += 1

    split_gm = torch.fx.passes.split_module.split_module(  # type: ignore[attr-defined]
        graph,
        None,
        lambda n: node_to_subgraph_id.get(n, 0),
        keep_original_order=True,
    )

    return split_gm


def tag_graph(
    gm: torch.fx.GraphModule, tag_rules: dict[MatchingRule, set[str]]
) -> None:
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
        node_list = list(module.graph.nodes)
        node_idx = 0
        while node_idx < len(node_list):
            matched = False
            for rule, tags in tag_rules.items():
                matched_idx = rule.matches(node_list[node_idx:])
                if matched_idx > 0:
                    module.tag.update(tags)
                    node_idx += matched_idx
                    matched = True
                    break
            if not matched:
                node_idx += 1

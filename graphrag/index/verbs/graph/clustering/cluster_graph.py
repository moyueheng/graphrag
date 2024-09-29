# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing cluster_graph, apply_clustering and run_layout methods definition."""

import logging
from enum import Enum
from random import Random
from typing import Any, cast

import networkx as nx
import pandas as pd
from datashaper import TableContainer, VerbCallbacks, VerbInput, progress_iterable, verb

from graphrag.index.utils import gen_uuid, load_graph

from .typing import Communities

log = logging.getLogger(__name__)


@verb(name="cluster_graph")
def cluster_graph(
    input: VerbInput,
    callbacks: VerbCallbacks,
    strategy: dict[str, Any],
    column: str,
    to: str,
    level_to: str | None = None,
    **_kwargs,
) -> TableContainer:
    """
    对图应用层次聚类算法。预期图以graphml格式提供。该动词输出一个包含聚类图的新列，以及一个包含图层级的新列。

    ## 用法
    ```yaml
    verb: cluster_graph
    args:
        column: entity_graph # 包含图的列名，应为graphml格式的图
        to: clustered_graph # 输出聚类图的列名
        level_to: level # 输出层级的列名
        strategy: <strategy config> # 见下面的策略部分
    ```

    ## 策略
    cluster_graph动词使用策略来聚类图。策略是一个定义使用方法的json对象。可用的策略如下：

    ### leiden
    此策略使用leiden算法来聚类图。策略配置如下：
    ```yaml
    strategy:
        type: leiden
        max_cluster_size: 10 # 可选，使用的最大聚类大小，默认：10
        use_lcc: true # 可选，是否使用最大连通分量与leiden算法，默认：true
        seed: 0xDEADBEEF # 可选，用于leiden算法的种子，默认：0xDEADBEEF
        levels: [0, 1] # 可选，输出的层级，默认：检测到的所有层级

    ```
    """
    # 获取输入数据框
    output_df = cast(pd.DataFrame, input.get_input())
    # 对每个图应用布局算法
    results = output_df[column].apply(lambda graph: run_layout(strategy, graph))

    community_map_to = "communities"
    output_df[community_map_to] = results

    # 设置层级列名
    level_to = level_to or f"{to}_level"
    # 提取每行的所有唯一层级
    output_df[level_to] = output_df.apply(
        lambda x: list({level for level, _, _ in x[community_map_to]}), axis=1
    )
    # 初始化聚类图列
    output_df[to] = [None] * len(output_df)

    num_total = len(output_df)

    # 为此次运行创建种子（如果未提供）
    seed = strategy.get("seed", Random().randint(0, 0xFFFFFFFF))  # noqa S311

    # 遍历每一行
    graph_level_pairs_column: list[list[tuple[int, str]]] = []
    for _, row in progress_iterable(
        output_df.iterrows(), callbacks.progress, num_total
    ):
        levels = row[level_to]
        graph_level_pairs: list[tuple[int, str]] = []

        # 对每个层级，获取图并添加到列表中
        for level in levels:
            graph = "\n".join(
                nx.generate_graphml(
                    apply_clustering(
                        cast(str, row[column]),
                        cast(Communities, row[community_map_to]),
                        level,
                        seed=seed,
                    )
                )
            )
            graph_level_pairs.append((level, graph))
        graph_level_pairs_column.append(graph_level_pairs)
    output_df[to] = graph_level_pairs_column

    # 将(level, graph)对列表展开为单独的行
    output_df = output_df.explode(to, ignore_index=True)

    # 将(level, graph)对分割为单独的列
    # TODO: 可能有更好的方法来做这个, FIX: 报错了, ValueError: Columns must be same length as key
    output_df[[level_to, to]] = pd.DataFrame(
        output_df[to].tolist(), index=output_df.index
    )

    # 清理社区映射
    output_df.drop(columns=[community_map_to], inplace=True)

    return TableContainer(table=output_df)


# TODO: This should support str | nx.Graph as a graphml param
def apply_clustering(
    graphml: str, communities: Communities, level: int = 0, seed: int | None = None
) -> nx.Graph:
    """Apply clustering to a graphml string."""
    random = Random(seed)  # noqa S311
    graph = nx.parse_graphml(graphml)
    for community_level, community_id, nodes in communities:
        if level == community_level:
            for node in nodes:
                graph.nodes[node]["cluster"] = community_id
                graph.nodes[node]["level"] = level

    # add node degree
    for node_degree in graph.degree:
        graph.nodes[str(node_degree[0])]["degree"] = int(node_degree[1])

    # add node uuid and incremental record id (a human readable id used as reference in the final report)
    for index, node in enumerate(graph.nodes()):
        graph.nodes[node]["human_readable_id"] = index
        graph.nodes[node]["id"] = str(gen_uuid(random))

    # add ids to edges
    for index, edge in enumerate(graph.edges()):
        graph.edges[edge]["id"] = str(gen_uuid(random))
        graph.edges[edge]["human_readable_id"] = index
        graph.edges[edge]["level"] = level
    return graph


class GraphCommunityStrategyType(str, Enum):
    """GraphCommunityStrategyType class definition."""

    leiden = "leiden"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'


def run_layout(
    strategy: dict[str, Any], graphml_or_graph: str | nx.Graph
) -> Communities:
    """Run layout method definition."""
    graph = load_graph(graphml_or_graph)
    if len(graph.nodes) == 0:
        log.warning("Graph has no nodes")
        return []

    clusters: dict[int, dict[str, list[str]]] = {}
    strategy_type = strategy.get("type", GraphCommunityStrategyType.leiden)
    match strategy_type:
        case GraphCommunityStrategyType.leiden:
            from .strategies.leiden import run as run_leiden

            clusters = run_leiden(graph, strategy)
        case _:
            msg = f"Unknown clustering strategy {strategy_type}"
            raise ValueError(msg)

    results: Communities = []
    for level in clusters:
        for cluster_id, nodes in clusters[level].items():
            results.append((level, cluster_id, nodes))
    return results

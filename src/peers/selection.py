"""Peer graph construction, community detection, and assignment hysteresis."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

import community as community_louvain
import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering

CommunityMethod = Literal["louvain", "greedy_modularity", "spectral_k"]


def build_peer_graph(
    cids: list[int] | tuple[int, ...],
    sim_matrix: np.ndarray,
    threshold: float,
    top_k: int,
) -> nx.Graph:
    """Build a weighted client similarity graph with threshold and/or top-k sparsification.

    All clients appear as nodes even if they have no incident edges (isolated).
    An edge ``(i, j)`` is kept when ``sim[i, j] >= threshold`` and/or ``j`` is among
    the ``top_k`` highest-similarity neighbors of ``i`` (union of both rules).
    """
    n = len(cids)
    if sim_matrix.shape != (n, n):
        raise ValueError(
            f"sim_matrix shape {sim_matrix.shape} does not match {n} clients"
        )

    graph = nx.Graph()
    graph.add_nodes_from(cids)

    if n <= 1:
        return graph

    edge_indices: set[tuple[int, int]] = set()

    for row in range(n):
        for col in range(row + 1, n):
            if sim_matrix[row, col] >= threshold:
                edge_indices.add((row, col))

    if top_k > 0:
        k = min(top_k, n - 1)
        for row in range(n):
            similarities = sim_matrix[row].copy()
            similarities[row] = -np.inf
            neighbor_cols = np.argpartition(-similarities, k - 1)[:k]
            for col in neighbor_cols:
                if not np.isfinite(similarities[col]):
                    continue
                low, high = (row, col) if row < col else (col, row)
                edge_indices.add((low, high))

    for row, col in edge_indices:
        weight = float(sim_matrix[row, col])
        if weight <= 0.0:
            continue
        graph.add_edge(cids[row], cids[col], weight=weight)

    return graph


def _affinity_from_graph(graph: nx.Graph, ordered_nodes: list[int]) -> np.ndarray:
    """Build a symmetric affinity matrix from edge weights (diagonal = 1)."""
    index = {node: idx for idx, node in enumerate(ordered_nodes)}
    size = len(ordered_nodes)
    affinity = np.eye(size, dtype=np.float64)
    for left, right, data in graph.edges(data=True):
        weight = float(data.get("weight", 1.0))
        i, j = index[left], index[right]
        affinity[i, j] = weight
        affinity[j, i] = weight
    return affinity


def select_peers(
    graph: nx.Graph,
    method: CommunityMethod,
    k: int | None = None,
) -> dict[int, int]:
    """Assign each client to an edge-aggregator group via community detection.

    Isolated nodes (no edges) receive their own singleton group. Connected
    components are clustered separately for methods that require connectivity.
    """
    if method not in {"louvain", "greedy_modularity", "spectral_k"}:
        raise ValueError(f"Unsupported community method: {method!r}")
    if method == "spectral_k" and (k is None or k <= 0):
        raise ValueError("spectral_k requires a positive integer k")

    nodes = sorted(graph.nodes())
    if not nodes:
        return {}

    if method == "spectral_k":
        if len(nodes) <= k:
            return {node: idx for idx, node in enumerate(nodes)}
        affinity = _affinity_from_graph(graph, nodes)
        labels = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=0,
        ).fit_predict(affinity)
        return {
            node: int(label) for node, label in zip(nodes, labels, strict=True)
        }

    assignment: dict[int, int] = {}
    next_group_id = 0

    for component in sorted(nx.connected_components(graph), key=lambda c: min(c)):
        component_nodes = sorted(component)
        if len(component_nodes) == 1:
            assignment[component_nodes[0]] = next_group_id
            next_group_id += 1
            continue

        subgraph = graph.subgraph(component_nodes)

        if method == "louvain":
            partition = community_louvain.best_partition(subgraph, weight="weight")
            local_labels = {node: partition[node] for node in component_nodes}
        else:
            communities = nx.community.greedy_modularity_communities(
                subgraph, weight="weight"
            )
            local_labels = {}
            for local_id, comm in enumerate(communities):
                for node in comm:
                    local_labels[node] = local_id

        label_to_group = {}
        for node in component_nodes:
            local_label = local_labels[node]
            if local_label not in label_to_group:
                label_to_group[local_label] = next_group_id
                next_group_id += 1
            assignment[node] = label_to_group[local_label]

    return assignment


def apply_hysteresis(
    prev_assignment: dict[int, int] | None,
    new_assignment: dict[int, int],
    hysteresis: float,
) -> dict[int, int]:
    """Stabilize group ids across rounds by matching new clusters to prior groups.

    Each detected community is mapped to the previous group with maximum overlap
    when overlap / |community| >= ``hysteresis``; otherwise it receives a fresh
    group id. This avoids relabelling the same community with a new integer
    every round.
    """
    if not prev_assignment:
        return dict(new_assignment)
    if not 0.0 <= hysteresis <= 1.0:
        raise ValueError(f"hysteresis must be in [0, 1], got {hysteresis}")

    clusters: dict[int, list[int]] = defaultdict(list)
    for client_id, cluster_label in new_assignment.items():
        clusters[cluster_label].append(client_id)

    prev_groups: dict[int, set[int]] = defaultdict(set)
    for client_id, group_id in prev_assignment.items():
        prev_groups[group_id].add(client_id)

    used_group_ids: set[int] = set()
    next_group_id = max(prev_assignment.values(), default=-1) + 1
    stable: dict[int, int] = {}

    for cluster_label in sorted(clusters.keys()):
        members = set(clusters[cluster_label])
        best_group: int | None = None
        best_overlap = 0
        for group_id, group_members in prev_groups.items():
            if group_id in used_group_ids:
                continue
            overlap = len(members & group_members)
            if overlap > best_overlap:
                best_overlap = overlap
                best_group = group_id

        overlap_fraction = best_overlap / len(members) if members else 0.0
        if best_group is not None and overlap_fraction >= hysteresis:
            stable[cluster_label] = best_group
            used_group_ids.add(best_group)
        else:
            while next_group_id in used_group_ids:
                next_group_id += 1
            stable[cluster_label] = next_group_id
            used_group_ids.add(next_group_id)
            next_group_id += 1

    return {cid: stable[label] for cid, label in new_assignment.items()}

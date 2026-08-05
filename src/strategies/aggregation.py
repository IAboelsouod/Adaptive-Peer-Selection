"""Pure hierarchical FedAvg aggregation (Abad et al. Algorithm 2)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from flwr.server.strategy.aggregate import aggregate

NDArrayList = list[np.ndarray]


def hierarchical_aggregate(
    results_by_cid: dict[int, tuple[NDArrayList, int]],
    assignment: dict[int, int],
    is_sync_round: bool,
) -> tuple[NDArrayList | None, dict[int, NDArrayList]]:
    """FedAvg within each cluster every round; global FedAvg across clusters on sync rounds.

    Parameters
    ----------
    results_by_cid:
        Client id -> (model weights, num_examples) from the current fit round.
    assignment:
        Client id -> edge-aggregator / cluster group id.
    is_sync_round:
        When ``True``, also average cluster models into a global model.

    Returns
    -------
    global_params:
        Updated global weights on sync rounds, otherwise ``None``.
    cluster_params:
        FedAvg cluster model per group id present in ``assignment``.
    """
    groups: dict[int, list[tuple[NDArrayList, int]]] = defaultdict(list)
    for client_id, (params, num_examples) in results_by_cid.items():
        group_id = assignment[client_id]
        groups[group_id].append((params, num_examples))

    cluster_params: dict[int, NDArrayList] = {}
    cluster_weights: list[tuple[NDArrayList, int]] = []

    for group_id in sorted(groups.keys()):
        member_results = groups[group_id]
        cluster_model = aggregate(member_results)
        cluster_params[group_id] = cluster_model
        total_examples = sum(num_examples for _, num_examples in member_results)
        cluster_weights.append((cluster_model, total_examples))

    global_params: NDArrayList | None = None
    if is_sync_round and cluster_weights:
        global_params = aggregate(cluster_weights)

    return global_params, cluster_params

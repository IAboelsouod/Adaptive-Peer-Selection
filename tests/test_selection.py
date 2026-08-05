"""Tests for peer similarity graph construction and community selection."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from src.models.cnn import GrayscaleCNN, classifier_param_indices, get_parameters
from src.peers.selection import (
    apply_hysteresis,
    build_peer_graph,
    select_peers,
)
from src.peers.similarity import compute_deltas, cosine_similarity_matrix


def _planted_block_deltas(
    *,
    n_clusters: int = 3,
    n_per_cluster: int = 6,
    dim: int = 32,
    noise: float = 0.06,
    seed: int = 0,
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """Synthetic deltas with clear block structure for community recovery tests."""
    rng = np.random.default_rng(seed)
    centroids = rng.normal(size=(n_clusters, dim))
    deltas: dict[int, np.ndarray] = {}
    planted: dict[int, int] = {}
    client_id = 0
    for cluster_id in range(n_clusters):
        for _ in range(n_per_cluster):
            deltas[client_id] = centroids[cluster_id] + rng.normal(
                scale=noise, size=dim
            )
            planted[client_id] = cluster_id
            client_id += 1
    return deltas, planted


def _params_from_deltas(
    deltas: dict[int, np.ndarray],
    *,
    global_params: list[np.ndarray] | None = None,
) -> tuple[dict[int, list[np.ndarray]], list[np.ndarray]]:
    """Wrap flat deltas as single-layer client/global parameter lists."""
    if global_params is None:
        sample_dim = next(iter(deltas.values())).shape[0]
        global_params = [np.zeros(sample_dim, dtype=np.float64)]
    client_params = {
        cid: [global_params[0] + delta] for cid, delta in deltas.items()
    }
    return client_params, global_params


def _assignment_ari(planted: dict[int, int], recovered: dict[int, int]) -> float:
    shared = sorted(planted.keys())
    return float(
        adjusted_rand_score(
            [planted[cid] for cid in shared],
            [recovered[cid] for cid in shared],
        )
    )


def _run_selection_pipeline(
    deltas: dict[int, np.ndarray],
    *,
    threshold: float = 0.5,
    top_k: int = 4,
    method: str = "louvain",
    k: int | None = None,
) -> dict[int, int]:
    ordered_cids, sim_matrix = cosine_similarity_matrix(deltas)
    graph = build_peer_graph(ordered_cids, sim_matrix, threshold, top_k)
    return select_peers(graph, method, k=k)  # type: ignore[arg-type]


def test_compute_deltas_last_layer_matches_classifier_slice():
    model = GrayscaleCNN()
    last_slice = classifier_param_indices(model)
    global_params = get_parameters(model)

    rng = np.random.default_rng(7)
    client_params: dict[int, list[np.ndarray]] = {}
    expected: dict[int, np.ndarray] = {}
    start, end = last_slice
    for client_id in range(4):
        weights = [array.copy() for array in global_params]
        for param_idx in range(start, end):
            weights[param_idx] = weights[param_idx] + rng.normal(
                scale=0.1, size=weights[param_idx].shape
            )
        client_params[client_id] = weights
        expected[client_id] = np.concatenate(
            [
                (weights[param_idx] - global_params[param_idx]).ravel()
                for param_idx in range(start, end)
            ]
        )

    deltas = compute_deltas(client_params, global_params, "last_layer", last_slice)
    for client_id, delta in deltas.items():
        assert np.allclose(delta, expected[client_id], atol=1e-6)


def test_compute_deltas_full_uses_all_layers():
    model = GrayscaleCNN()
    last_slice = classifier_param_indices(model)
    global_params = get_parameters(model)
    client_params = {0: [array + 0.01 for array in global_params]}
    full_delta = compute_deltas(client_params, global_params, "full", last_slice)[0]
    last_delta = compute_deltas(client_params, global_params, "last_layer", last_slice)[0]
    assert full_delta.shape[0] > last_delta.shape[0]


def test_cosine_similarity_matrix_returns_sorted_cids():
    deltas, _ = _planted_block_deltas(n_per_cluster=3)
    ordered_cids, matrix = cosine_similarity_matrix(deltas)
    assert ordered_cids == sorted(deltas.keys())
    assert matrix.shape == (len(ordered_cids), len(ordered_cids))
    assert np.allclose(np.diag(matrix), 1.0)


@pytest.mark.parametrize("method", ["louvain", "greedy_modularity", "spectral_k"])
def test_recovered_communities_match_planted_labels(method: str):
    deltas, planted = _planted_block_deltas()
    k = 3 if method == "spectral_k" else None
    recovered = _run_selection_pipeline(deltas, threshold=0.3, top_k=5, method=method, k=k)
    ari = _assignment_ari(planted, recovered)
    assert ari >= 0.95, f"{method} ARI={ari:.3f}"


def test_threshold_reduces_graph_density():
    deltas, _ = _planted_block_deltas()
    ordered_cids, sim_matrix = cosine_similarity_matrix(deltas)

    dense = build_peer_graph(ordered_cids, sim_matrix, threshold=0.0, top_k=0)
    sparse = build_peer_graph(ordered_cids, sim_matrix, threshold=0.85, top_k=0)

    assert dense.number_of_edges() > sparse.number_of_edges()


def test_top_k_changes_graph_density():
    deltas, _ = _planted_block_deltas()
    ordered_cids, sim_matrix = cosine_similarity_matrix(deltas)

    # No threshold edges; only top-k contributes.
    small_k = build_peer_graph(ordered_cids, sim_matrix, threshold=1.1, top_k=2)
    large_k = build_peer_graph(ordered_cids, sim_matrix, threshold=1.1, top_k=8)

    assert large_k.number_of_edges() > small_k.number_of_edges()
    assert small_k.number_of_nodes() == len(ordered_cids)


def test_isolated_node_receives_singleton_group():
    deltas, planted = _planted_block_deltas(n_per_cluster=4)
    outlier_id = max(planted.keys()) + 1
    deltas[outlier_id] = np.ones(32) * 50.0

    ordered_cids, sim_matrix = cosine_similarity_matrix(deltas)
    graph = build_peer_graph(ordered_cids, sim_matrix, threshold=0.9, top_k=0)
    assert graph.degree(outlier_id) == 0

    assignment = select_peers(graph, "louvain")
    singleton_groups = {
        group_id
        for group_id in set(assignment.values())
        if sum(1 for cid in assignment if assignment[cid] == group_id) == 1
    }
    assert assignment[outlier_id] in singleton_groups

    planted[outlier_id] = 99
    core_assignment = {cid: assignment[cid] for cid in assignment if cid != outlier_id}
    core_planted = {cid: planted[cid] for cid in planted if cid != outlier_id}
    assert _assignment_ari(core_planted, core_assignment) >= 0.95


def test_apply_hysteresis_preserves_stable_group_ids():
    deltas, _ = _planted_block_deltas(seed=1)
    first = _run_selection_pipeline(deltas, threshold=0.3, top_k=5)

    noisy_deltas = {
        cid: delta + np.random.default_rng(cid).normal(scale=0.02, size=delta.shape)
        for cid, delta in deltas.items()
    }
    second_raw = _run_selection_pipeline(noisy_deltas, threshold=0.3, top_k=5)
    second_stable = apply_hysteresis(first, second_raw, hysteresis=0.5)

    assert set(first.values()) == set(second_stable.values())
    stable_clients = sum(1 for cid in first if first[cid] == second_stable[cid])
    assert stable_clients / len(first) >= 0.8


def test_end_to_end_from_parameter_deltas():
    deltas, planted = _planted_block_deltas()
    client_params, global_params = _params_from_deltas(deltas)
    computed = compute_deltas(
        client_params,
        global_params,
        "last_layer",
        (0, 1),
    )
    ordered_cids, sim_matrix = cosine_similarity_matrix(computed)
    graph = build_peer_graph(ordered_cids, sim_matrix, threshold=0.4, top_k=4)
    recovered = select_peers(graph, "louvain")
    assert _assignment_ari(planted, recovered) >= 0.95

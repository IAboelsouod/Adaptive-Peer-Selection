"""System C: hierarchical FL with adaptive peer selection."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from flwr.common import Parameters
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    import networkx as nx

from src.config import Config, set_seed
from src.metrics.logging import RoundLogger
from src.models.cnn import classifier_param_indices, create_model
from src.peers.selection import apply_hysteresis, build_peer_graph, select_peers
from src.peers.similarity import compute_deltas, cosine_similarity_matrix
from src.strategies.hierarchical_base import HierarchicalStrategy

PeerGraphSnapshotCallback = Callable[["nx.Graph", dict[int, int], int], None]


def random_assignment(
    num_clients: int,
    num_groups: int,
    seed: int,
) -> dict[int, int]:
    """Random client-to-group assignment for warm-up rounds."""
    rng = np.random.default_rng(seed)
    return {
        client_id: int(rng.integers(0, num_groups))
        for client_id in range(num_clients)
    }


class AdaptivePeersStrategy(HierarchicalStrategy):
    """Hierarchical FL with adaptive cosine-delta peer selection at sync boundaries."""

    def __init__(
        self,
        *,
        cfg: Config,
        test_loader: DataLoader,
        round_logger: RoundLogger,
        ground_truth_labels: list[int],
        num_clients: int,
        initial_parameters: Parameters,
        peer_graph_snapshot_cb: PeerGraphSnapshotCallback | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(
            cfg=cfg,
            test_loader=test_loader,
            round_logger=round_logger,
            ground_truth_labels=ground_truth_labels,
            num_clients=num_clients,
            initial_parameters=initial_parameters,
            **kwargs,
        )
        model = create_model(cfg)
        self._last_layer_slice = classifier_param_indices(model)
        self._assignment = random_assignment(
            num_clients,
            cfg.hierarchy.num_groups,
            cfg.experiment.seed,
        )
        self._peer_graph_snapshot_cb = peer_graph_snapshot_cb

    def _should_recluster(self, server_round: int) -> bool:
        return (
            server_round > self._cfg.peers.warmup_rounds
            and self._is_sync_round(server_round)
            and server_round % self._cfg.peers.recluster_every == 0
        )

    def _recluster(
        self,
        server_round: int,
        results_by_cid: dict[int, tuple[list[np.ndarray], int]],
        reference_global: list[np.ndarray],
    ) -> None:
        """Recompute assignment from deltas against the pre-round global model."""
        client_params = {
            client_id: params for client_id, (params, _) in results_by_cid.items()
        }
        deltas = compute_deltas(
            client_params,
            reference_global,
            self._cfg.peers.similarity_source,
            self._last_layer_slice,
        )
        ordered_cids, sim_matrix = cosine_similarity_matrix(deltas)
        graph = build_peer_graph(
            ordered_cids,
            sim_matrix,
            self._cfg.peers.threshold,
            self._cfg.peers.top_k,
        )
        method = self._cfg.peers.community_method
        k = (
            self._cfg.data.num_latent_groups
            if method == "spectral_k"
            else None
        )
        new_assignment = select_peers(graph, method, k=k)

        for client_id in range(self._num_clients):
            if client_id not in new_assignment:
                new_assignment[client_id] = max(new_assignment.values(), default=-1) + 1

        self._assignment = apply_hysteresis(
            self._assignment,
            new_assignment,
            self._cfg.peers.hysteresis,
        )
        if self._peer_graph_snapshot_cb is not None:
            self._peer_graph_snapshot_cb(
                graph, dict(self._assignment), server_round
            )

    def _update_assignment(
        self,
        server_round: int,
        results_by_cid: dict[int, tuple[list[np.ndarray], int]],
        *,
        before_aggregate: bool,
        reference_global: list[np.ndarray] | None = None,
    ) -> None:
        if before_aggregate:
            if server_round <= self._cfg.peers.warmup_rounds:
                self._assignment = random_assignment(
                    self._num_clients,
                    self._cfg.hierarchy.num_groups,
                    self._cfg.experiment.seed + server_round,
                )
            return

        if (
            self._should_recluster(server_round)
            and reference_global is not None
        ):
            self._recluster(server_round, results_by_cid, reference_global)


def build_adaptive_peers_strategy(
    cfg: Config,
    test_loader: DataLoader,
    round_logger: RoundLogger,
    ground_truth_labels: list[int],
    initial_parameters: Parameters,
    peer_graph_snapshot_cb: PeerGraphSnapshotCallback | None = None,
) -> AdaptivePeersStrategy:
    """Construct the adaptive peer-selection strategy."""
    set_seed(cfg.experiment.seed)
    num_clients = cfg.federation.num_clients
    min_fit_clients = max(2, int(num_clients * cfg.federation.fraction_fit))

    def on_fit_config_fn(server_round: int) -> dict[str, float | int]:
        _ = server_round
        return {"local_epochs": cfg.federation.local_epochs}

    return AdaptivePeersStrategy(
        cfg=cfg,
        test_loader=test_loader,
        round_logger=round_logger,
        ground_truth_labels=ground_truth_labels,
        num_clients=num_clients,
        initial_parameters=initial_parameters,
        fraction_fit=cfg.federation.fraction_fit,
        fraction_evaluate=0.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=2,
        min_available_clients=num_clients,
        on_fit_config_fn=on_fit_config_fn,
        peer_graph_snapshot_cb=peer_graph_snapshot_cb,
    )

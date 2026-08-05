"""Shared hierarchical FL strategy logic (Abad et al. Algorithm 2)."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np
import torch
from flwr.common import (
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch.utils.data import DataLoader

from src.client import evaluate_model
from src.config import Config
from src.metrics.logging import RoundLogger
from src.models.cnn import create_model, set_parameters
from src.strategies.aggregation import hierarchical_aggregate


def partition_id_from_proxy(client: ClientProxy) -> int:
    """Return the simulation partition id for a ``ClientProxy``."""
    if hasattr(client, "partition_id"):
        return int(client.partition_id)
    raise TypeError(
        f"Cannot resolve client id from proxy type {type(client).__name__}; "
        "expected simulation RayActorClientProxy with partition_id."
    )


def results_by_client_id(
    results: Sequence[tuple[ClientProxy, FitRes]],
) -> dict[int, tuple[list[np.ndarray], int]]:
    """Map fit results to client ids using metrics from Phase 2 ``fit()``."""
    mapped: dict[int, tuple[list[np.ndarray], int]] = {}
    for _, fit_res in results:
        client_id = int(fit_res.metrics["client_id"])
        params = parameters_to_ndarrays(fit_res.parameters)
        mapped[client_id] = (params, fit_res.num_examples)
    return mapped


class HierarchicalStrategy(FedAvg):
    """Base strategy for hierarchical FL with H-periodic global sync."""

    def __init__(
        self,
        *,
        cfg: Config,
        test_loader: DataLoader,
        round_logger: RoundLogger,
        ground_truth_labels: Sequence[int],
        num_clients: int,
        initial_parameters: Parameters,
        **kwargs: object,
    ) -> None:
        kwargs.setdefault("initial_parameters", initial_parameters)
        super().__init__(**kwargs)
        self._cfg = cfg
        self._test_loader = test_loader
        self._round_logger = round_logger
        self._ground_truth = list(ground_truth_labels)
        self._num_clients = num_clients
        self._global_params: list[np.ndarray] = parameters_to_ndarrays(initial_parameters)
        self._last_global_params: list[np.ndarray] = [
            array.copy() for array in self._global_params
        ]
        self._cluster_params: dict[int, list[np.ndarray]] = {}
        self._assignment: dict[int, int] = {}
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._eval_model = create_model(cfg)
        self._eval_model.to(self._device)
        self._start_time = time.perf_counter()
        self.evaluate_fn = self._make_evaluate_fn()

    @property
    def global_sync_every(self) -> int:
        return self._cfg.hierarchy.global_sync_every

    def _is_sync_round(self, server_round: int) -> bool:
        return server_round % self.global_sync_every == 0

    def _send_global_model(self, server_round: int) -> bool:
        """Distribution rule: global after sync or when configured for pure HFL."""
        if self._cfg.experiment.model_distribution == "global":
            return True
        if not self._cluster_params:
            return True
        return (server_round - 1) % self.global_sync_every == 0

    def _cluster_metrics(self) -> tuple[float, float]:
        predicted = [self._assignment[cid] for cid in range(self._num_clients)]
        truth = self._ground_truth
        return (
            float(adjusted_rand_score(truth, predicted)),
            float(normalized_mutual_info_score(truth, predicted)),
        )

    def _make_evaluate_fn(
        self,
    ) -> Callable[
        [int, NDArrays, dict[str, Scalar]],
        tuple[float, dict[str, Scalar]] | None,
    ]:
        def evaluate_fn(
            server_round: int,
            parameters: NDArrays,
            config: dict[str, Scalar],
        ) -> tuple[float, dict[str, Scalar]] | None:
            _ = parameters, config
            is_sync = self._is_sync_round(server_round)
            ari, nmi = self._cluster_metrics()
            num_groups = len(set(self._assignment.values())) if self._assignment else None

            global_acc: float | None = None
            global_loss: float | None = None
            metrics: dict[str, Scalar] = {}

            if is_sync:
                set_parameters(self._eval_model, self._global_params)
                global_loss, global_acc = evaluate_model(
                    self._eval_model, self._test_loader, self._device
                )
                global_acc = float(global_acc)
                global_loss = float(global_loss)
                metrics["accuracy"] = global_acc

            self._round_logger.log_round(
                round=server_round,
                global_acc=global_acc,
                global_loss=global_loss,
                num_groups=num_groups,
                ari=ari,
                nmi=nmi,
                wall_time=time.perf_counter() - self._start_time,
            )

            if is_sync:
                return global_loss, metrics
            return None

        return evaluate_fn

    def _params_for_client(self, client_id: int, server_round: int) -> Parameters:
        if self._send_global_model(server_round):
            return ndarrays_to_parameters(self._global_params)
        group_id = self._assignment[client_id]
        return ndarrays_to_parameters(self._cluster_params[group_id])

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        _ = parameters
        config: dict[str, Scalar] = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        return [
            (client, FitIns(self._params_for_client(partition_id_from_proxy(client), server_round), config))
            for client in clients
        ]

    def _update_assignment(
        self,
        server_round: int,
        results_by_cid: dict[int, tuple[list[np.ndarray], int]],
        *,
        before_aggregate: bool,
        reference_global: list[np.ndarray] | None = None,
    ) -> None:
        """Hook for subclasses to refresh client->cluster assignment."""

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        results_by_cid = results_by_client_id(results)
        is_sync = self._is_sync_round(server_round)
        reference_global = [array.copy() for array in self._last_global_params]

        self._update_assignment(
            server_round, results_by_cid, before_aggregate=True
        )

        global_params, cluster_params = hierarchical_aggregate(
            results_by_cid, self._assignment, is_sync
        )
        self._cluster_params = cluster_params

        if global_params is not None:
            self._global_params = global_params
            self._last_global_params = [array.copy() for array in global_params]

        self._update_assignment(
            server_round,
            results_by_cid,
            before_aggregate=False,
            reference_global=reference_global,
        )

        return ndarrays_to_parameters(self._global_params), {}

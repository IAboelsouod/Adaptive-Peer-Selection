"""Baseline B: hierarchical FL with static client-to-cluster assignment."""

from __future__ import annotations

from flwr.common import Parameters
from torch.utils.data import DataLoader

from src.config import Config
from src.metrics.logging import RoundLogger
from src.strategies.hierarchical_base import HierarchicalStrategy


def round_robin_assignment(num_clients: int, num_groups: int) -> dict[int, int]:
    """Assign clients to fixed groups in round-robin order."""
    return {client_id: client_id % num_groups for client_id in range(num_clients)}


class HierStaticStrategy(HierarchicalStrategy):
    """Hierarchical FL with a fixed round-robin cluster assignment."""

    def __init__(
        self,
        *,
        cfg: Config,
        test_loader: DataLoader,
        round_logger: RoundLogger,
        ground_truth_labels: list[int],
        num_clients: int,
        initial_parameters: Parameters,
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
        self._assignment = round_robin_assignment(
            num_clients, cfg.hierarchy.num_groups
        )


def build_hier_static_strategy(
    cfg: Config,
    test_loader: DataLoader,
    round_logger: RoundLogger,
    ground_truth_labels: list[int],
    initial_parameters: Parameters,
) -> HierStaticStrategy:
    """Construct the static hierarchical baseline strategy."""
    num_clients = cfg.federation.num_clients
    min_fit_clients = max(2, int(num_clients * cfg.federation.fraction_fit))

    def on_fit_config_fn(server_round: int) -> dict[str, float | int]:
        _ = server_round
        return {"local_epochs": cfg.federation.local_epochs}

    return HierStaticStrategy(
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
    )

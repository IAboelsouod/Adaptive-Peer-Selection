"""Baseline A: flat FedAvg with centralized test-set evaluation."""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
from flwr.common import NDArrays, Parameters, Scalar
from flwr.server.strategy import FedAvg
from torch.utils.data import DataLoader

from src.client import evaluate_model
from src.config import Config
from src.metrics.logging import RoundLogger
from src.models.cnn import create_model, set_parameters


class FlatFedAvg(FedAvg):
    """FedAvg baseline that logs centralized global metrics each round."""

    def __init__(
        self,
        *,
        cfg: Config,
        test_loader: DataLoader,
        round_logger: RoundLogger,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._cfg = cfg
        self._test_loader = test_loader
        self._round_logger = round_logger
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._eval_model = create_model(cfg)
        self._eval_model.to(self._device)
        self._start_time = time.perf_counter()
        self.evaluate_fn = self._make_evaluate_fn()

    def _make_evaluate_fn(
        self,
    ) -> Callable[
        [int, NDArrays, dict[str, Scalar]],
        tuple[float, dict[str, Scalar]],
    ]:
        """Build centralized evaluation on the shared held-out test loader."""

        def evaluate_fn(
            server_round: int,
            parameters: NDArrays,
            config: dict[str, Scalar],
        ) -> tuple[float, dict[str, Scalar]]:
            _ = config
            set_parameters(self._eval_model, parameters)
            loss, accuracy = evaluate_model(
                self._eval_model, self._test_loader, self._device
            )
            wall_time = time.perf_counter() - self._start_time
            self._round_logger.log_round(
                round=server_round,
                global_acc=float(accuracy),
                global_loss=float(loss),
            )
            return float(loss), {"accuracy": float(accuracy)}

        return evaluate_fn


def build_flat_fedavg_strategy(
    cfg: Config,
    test_loader: DataLoader,
    round_logger: RoundLogger,
    initial_parameters: Parameters,
) -> FlatFedAvg:
    """Construct the flat FedAvg strategy for simulation."""
    num_clients = cfg.federation.num_clients
    min_fit_clients = max(
        2, int(num_clients * cfg.federation.fraction_fit)
    )

    def on_fit_config_fn(server_round: int) -> dict[str, Scalar]:
        _ = server_round
        return {"local_epochs": cfg.federation.local_epochs}

    return FlatFedAvg(
        cfg=cfg,
        test_loader=test_loader,
        round_logger=round_logger,
        fraction_fit=cfg.federation.fraction_fit,
        fraction_evaluate=0.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=2,
        min_available_clients=num_clients,
        on_fit_config_fn=on_fit_config_fn,
        initial_parameters=initial_parameters,
    )

"""Flower NumPyClient wrapper for local training and evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
from flwr.client import Client, NumPyClient
from flwr.common import Context
from torch.utils.data import DataLoader, Dataset

from src.config import Config, set_seed
from src.models.cnn import create_model, get_parameters, set_parameters


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    if cfg.training.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.training.lr, momentum=0.9)
    if cfg.training.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    raise ValueError(f"Unsupported optimizer: {cfg.training.optimizer}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    criterion: nn.Module | None = None,
    global_params: Sequence[np.ndarray] | None = None,
    fedprox_mu: float = 0.0,
) -> float:
    """Run one local training epoch and return the mean batch loss."""
    model.train()
    loss_fn = criterion or nn.CrossEntropyLoss()
    total_loss = 0.0
    num_batches = 0

    global_tensors: list[torch.Tensor] | None = None
    if global_params is not None and fedprox_mu > 0.0:
        global_tensors = [torch.tensor(array, device=device) for array in global_params]

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        if global_tensors is not None:
            prox = torch.tensor(0.0, device=device)
            for local_param, global_param in zip(
                model.parameters(), global_tensors, strict=True
            ):
                prox = prox + torch.sum((local_param - global_param) ** 2)
            loss = loss + (fedprox_mu / 2.0) * prox

        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[float, float]:
    """Return mean loss and accuracy on ``loader``."""
    model.eval()
    loss_fn = criterion or nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        batch_loss = loss_fn(outputs, targets)
        total_loss += float(batch_loss.item()) * targets.size(0)
        predictions = outputs.argmax(dim=1)
        correct += int((predictions == targets).sum().item())
        total += int(targets.size(0))

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


class FlowerClient(NumPyClient):
    """Flower client that trains a local model on one data partition."""

    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        model: nn.Module,
        cfg: Config,
        device: torch.device | None = None,
    ) -> None:
        self.client_id = client_id
        self.train_loader = train_loader
        self.model = model
        self.cfg = cfg
        self.device = device or _resolve_device()
        self.model.to(self.device)

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, float | int | str | bool],
    ) -> tuple[list[np.ndarray], int, dict[str, float | int]]:
        """Train on local data and return updated weights plus client id in metrics."""
        set_parameters(self.model, parameters)
        global_params = parameters
        optimizer = _build_optimizer(self.model, self.cfg)
        fedprox_mu = (
            self.cfg.training.fedprox_mu
            if self.cfg.training.local_optimizer == "fedprox"
            else 0.0
        )

        local_epochs = int(config.get("local_epochs", self.cfg.federation.local_epochs))
        for _ in range(local_epochs):
            train_one_epoch(
                self.model,
                self.train_loader,
                optimizer,
                self.device,
                global_params=global_params,
                fedprox_mu=fedprox_mu,
            )

        updated_params = get_parameters(self.model)
        num_examples = len(self.train_loader.dataset)
        metrics = {"client_id": self.client_id}
        return updated_params, num_examples, metrics

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, float | int | str | bool],
    ) -> tuple[float, int, dict[str, float]]:
        """Evaluate global parameters on the client's local dataset."""
        _ = config
        set_parameters(self.model, parameters)
        loss, accuracy = evaluate_model(self.model, self.train_loader, self.device)
        num_examples = len(self.train_loader.dataset)
        return loss, num_examples, {"accuracy": float(accuracy)}


def make_client_fn(
    cfg: Config,
    client_datasets: Sequence[Dataset],
) -> Callable[[Context], Client]:
    """Build a simulation ``client_fn`` that maps partition id to a Flower client."""
    set_seed(cfg.experiment.seed)

    def client_fn(context: Context) -> Client:
        partition_id = int(context.node_config["partition-id"])
        train_loader = DataLoader(
            client_datasets[partition_id],
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=0,
        )
        model = create_model(cfg)
        client = FlowerClient(
            client_id=partition_id,
            train_loader=train_loader,
            model=model,
            cfg=cfg,
        )
        return client.to_client()

    return client_fn

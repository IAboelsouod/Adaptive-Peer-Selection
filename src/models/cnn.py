"""Small CNN models and NumPy parameter (de)serialization helpers."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from src.config import Config, ModelName

_CLASSIFIER_PREFIX = "classifier."


class GrayscaleCNN(nn.Module):
    """Compact CNN for 28x28 grayscale inputs (FMNIST / MNIST)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(64 * 7 * 7, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        return self.classifier(x)


class CifarCNN(nn.Module):
    """Compact CNN for 32x32 RGB inputs (CIFAR-10)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        return self.classifier(x)


def create_model(cfg: Config, num_classes: int = 10) -> nn.Module:
    """Instantiate the model architecture named in the config."""
    name: ModelName = cfg.model.name
    if name in {"cnn_fmnist", "cnn_mnist"}:
        return GrayscaleCNN(num_classes=num_classes)
    if name == "cnn_cifar10":
        return CifarCNN(num_classes=num_classes)
    raise ValueError(f"Unsupported model name: {name}")


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    """Serialize model weights in ``state_dict`` order."""
    return [
        value.detach().cpu().numpy().copy() for value in model.state_dict().values()
    ]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """Load NumPy weight arrays into the model, preserving ``state_dict`` order."""
    state_dict = model.state_dict()
    if len(parameters) != len(state_dict):
        raise ValueError(
            f"Expected {len(state_dict)} parameter arrays, got {len(parameters)}"
        )
    ordered = OrderedDict(
        (key, torch.tensor(array))
        for key, array in zip(state_dict.keys(), parameters, strict=True)
    )
    model.load_state_dict(ordered, strict=True)


def classifier_param_indices(model: nn.Module) -> tuple[int, int]:
    """Return ``[start, end)`` indices of classifier parameters in ``get_parameters``."""
    keys = list(model.state_dict().keys())
    classifier_indices = [
        idx for idx, key in enumerate(keys) if key.startswith(_CLASSIFIER_PREFIX)
    ]
    if not classifier_indices:
        raise ValueError("Model has no parameters under the classifier module prefix")
    return min(classifier_indices), max(classifier_indices) + 1


def parameters_equal(
    left: list[np.ndarray], right: list[np.ndarray], atol: float = 1e-8
) -> bool:
    """Return whether two parameter lists are numerically identical."""
    if len(left) != len(right):
        return False
    return all(np.allclose(a, b, atol=atol) for a, b in zip(left, right, strict=True))

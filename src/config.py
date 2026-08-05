"""Typed configuration loader and reproducibility utilities."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml

DatasetName = Literal["fmnist", "mnist", "cifar10"]
PartitionScheme = Literal["dirichlet_latent", "dirichlet_only"]
ModelName = Literal["cnn_fmnist", "cnn_mnist", "cnn_cifar10"]
SimilaritySource = Literal["last_layer", "full"]
CommunityMethod = Literal["louvain", "greedy_modularity", "spectral_k"]
LocalOptimizer = Literal["fedavg", "fedprox"]
StrategyName = Literal["flat", "hier_static", "adaptive"]
ModelDistribution = Literal["global", "cluster"]
OptimizerName = Literal["sgd", "adam"]


@dataclass(frozen=True)
class DatasetConfig:
    name: DatasetName


@dataclass(frozen=True)
class FederationConfig:
    num_clients: int
    num_rounds: int
    local_epochs: int
    fraction_fit: float


@dataclass(frozen=True)
class HierarchyConfig:
    global_sync_every: int
    num_groups: int


@dataclass(frozen=True)
class DataConfig:
    alpha: float
    num_latent_groups: int
    partition_scheme: PartitionScheme


@dataclass(frozen=True)
class ModelConfig:
    name: ModelName


@dataclass(frozen=True)
class PeersConfig:
    similarity_source: SimilaritySource
    threshold: float
    top_k: int
    community_method: CommunityMethod
    recluster_every: int
    warmup_rounds: int
    hysteresis: float


@dataclass(frozen=True)
class TrainingConfig:
    lr: float
    batch_size: int
    optimizer: OptimizerName
    local_optimizer: LocalOptimizer
    fedprox_mu: float


@dataclass(frozen=True)
class ExperimentConfig:
    strategy: StrategyName
    model_distribution: ModelDistribution
    seed: int
    run_name: str


@dataclass(frozen=True)
class Config:
    """Root experiment configuration loaded from config.yaml."""

    dataset: DatasetConfig
    federation: FederationConfig
    hierarchy: HierarchyConfig
    data: DataConfig
    model: ModelConfig
    peers: PeersConfig
    training: TrainingConfig
    experiment: ExperimentConfig


_VALID_DATASETS = {"fmnist", "mnist", "cifar10"}
_VALID_PARTITION_SCHEMES = {"dirichlet_latent", "dirichlet_only"}
_VALID_MODELS = {"cnn_fmnist", "cnn_mnist", "cnn_cifar10"}
_VALID_SIMILARITY_SOURCES = {"last_layer", "full"}
_VALID_COMMUNITY_METHODS = {"louvain", "greedy_modularity", "spectral_k"}
_VALID_LOCAL_OPTIMIZERS = {"fedavg", "fedprox"}
_VALID_STRATEGIES = {"flat", "hier_static", "adaptive"}
_VALID_DISTRIBUTIONS = {"global", "cluster"}
_VALID_OPTIMIZERS = {"sgd", "adam"}


def _section(raw: dict[str, Any], section: str) -> dict[str, Any]:
    if section not in raw:
        raise ValueError(f"Missing required config section: {section}")
    value = raw[section]
    if not isinstance(value, dict):
        raise ValueError(f"Config section {section} must be a mapping")
    return value


def _require(section_raw: dict[str, Any], key: str, field: str) -> Any:
    if key not in section_raw:
        raise ValueError(f"Missing required config key: {field}")
    return section_raw[key]


def _validate_positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer, got {value!r}")
    return value


def _validate_fraction(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be a number in (0, 1], got {value!r}")
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{field} must be in (0, 1], got {value}")
    return value


def _validate_choice(value: Any, field: str, choices: set[str]) -> str:
    if value not in choices:
        raise ValueError(f"{field} must be one of {sorted(choices)}, got {value!r}")
    return value


def _parse_dataset(raw: dict[str, Any]) -> DatasetConfig:
    section = _section(raw, "dataset")
    name = _validate_choice(
        _require(section, "name", "dataset.name"), "dataset.name", _VALID_DATASETS
    )
    return DatasetConfig(name=name)  # type: ignore[arg-type]


def _parse_federation(raw: dict[str, Any]) -> FederationConfig:
    section = _section(raw, "federation")
    return FederationConfig(
        num_clients=_validate_positive_int(
            _require(section, "num_clients", "federation.num_clients"),
            "federation.num_clients",
        ),
        num_rounds=_validate_positive_int(
            _require(section, "num_rounds", "federation.num_rounds"),
            "federation.num_rounds",
        ),
        local_epochs=_validate_positive_int(
            _require(section, "local_epochs", "federation.local_epochs"),
            "federation.local_epochs",
        ),
        fraction_fit=_validate_fraction(
            _require(section, "fraction_fit", "federation.fraction_fit"),
            "federation.fraction_fit",
        ),
    )


def _parse_hierarchy(raw: dict[str, Any]) -> HierarchyConfig:
    section = _section(raw, "hierarchy")
    return HierarchyConfig(
        global_sync_every=_validate_positive_int(
            _require(section, "global_sync_every", "hierarchy.global_sync_every"),
            "hierarchy.global_sync_every",
        ),
        num_groups=_validate_positive_int(
            _require(section, "num_groups", "hierarchy.num_groups"),
            "hierarchy.num_groups",
        ),
    )


def _parse_data(raw: dict[str, Any]) -> DataConfig:
    section = _section(raw, "data")
    alpha = float(_require(section, "alpha", "data.alpha"))
    if alpha <= 0:
        raise ValueError(f"data.alpha must be positive, got {alpha}")
    return DataConfig(
        alpha=alpha,
        num_latent_groups=_validate_positive_int(
            _require(section, "num_latent_groups", "data.num_latent_groups"),
            "data.num_latent_groups",
        ),
        partition_scheme=_validate_choice(
            _require(section, "partition_scheme", "data.partition_scheme"),
            "data.partition_scheme",
            _VALID_PARTITION_SCHEMES,
        ),  # type: ignore[arg-type]
    )


def _parse_model(raw: dict[str, Any]) -> ModelConfig:
    section = _section(raw, "model")
    name = _validate_choice(
        _require(section, "name", "model.name"), "model.name", _VALID_MODELS
    )
    return ModelConfig(name=name)  # type: ignore[arg-type]


def _parse_peers(raw: dict[str, Any]) -> PeersConfig:
    section = _section(raw, "peers")
    hysteresis = float(_require(section, "hysteresis", "peers.hysteresis"))
    if not 0.0 <= hysteresis <= 1.0:
        raise ValueError(f"peers.hysteresis must be in [0, 1], got {hysteresis}")
    threshold = float(_require(section, "threshold", "peers.threshold"))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"peers.threshold must be in [0, 1], got {threshold}")
    return PeersConfig(
        similarity_source=_validate_choice(
            _require(section, "similarity_source", "peers.similarity_source"),
            "peers.similarity_source",
            _VALID_SIMILARITY_SOURCES,
        ),  # type: ignore[arg-type]
        threshold=threshold,
        top_k=_validate_positive_int(
            _require(section, "top_k", "peers.top_k"), "peers.top_k"
        ),
        community_method=_validate_choice(
            _require(section, "community_method", "peers.community_method"),
            "peers.community_method",
            _VALID_COMMUNITY_METHODS,
        ),  # type: ignore[arg-type]
        recluster_every=_validate_positive_int(
            _require(section, "recluster_every", "peers.recluster_every"),
            "peers.recluster_every",
        ),
        warmup_rounds=_validate_positive_int(
            _require(section, "warmup_rounds", "peers.warmup_rounds"),
            "peers.warmup_rounds",
        ),
        hysteresis=hysteresis,
    )


def _parse_training(raw: dict[str, Any]) -> TrainingConfig:
    section = _section(raw, "training")
    lr = float(_require(section, "lr", "training.lr"))
    if lr <= 0:
        raise ValueError(f"training.lr must be positive, got {lr}")
    fedprox_mu = float(_require(section, "fedprox_mu", "training.fedprox_mu"))
    if fedprox_mu < 0:
        raise ValueError(f"training.fedprox_mu must be non-negative, got {fedprox_mu}")
    return TrainingConfig(
        lr=lr,
        batch_size=_validate_positive_int(
            _require(section, "batch_size", "training.batch_size"),
            "training.batch_size",
        ),
        optimizer=_validate_choice(
            _require(section, "optimizer", "training.optimizer"),
            "training.optimizer",
            _VALID_OPTIMIZERS,
        ),  # type: ignore[arg-type]
        local_optimizer=_validate_choice(
            _require(section, "local_optimizer", "training.local_optimizer"),
            "training.local_optimizer",
            _VALID_LOCAL_OPTIMIZERS,
        ),  # type: ignore[arg-type]
        fedprox_mu=fedprox_mu,
    )


def _parse_experiment(raw: dict[str, Any]) -> ExperimentConfig:
    section = _section(raw, "experiment")
    run_name = _require(section, "run_name", "experiment.run_name")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("experiment.run_name must be a non-empty string")
    seed = _require(section, "seed", "experiment.seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"experiment.seed must be an integer, got {seed!r}")
    return ExperimentConfig(
        strategy=_validate_choice(
            _require(section, "strategy", "experiment.strategy"),
            "experiment.strategy",
            _VALID_STRATEGIES,
        ),  # type: ignore[arg-type]
        model_distribution=_validate_choice(
            _require(section, "model_distribution", "experiment.model_distribution"),
            "experiment.model_distribution",
            _VALID_DISTRIBUTIONS,
        ),  # type: ignore[arg-type]
        seed=seed,
        run_name=run_name,
    )


def load_config(path: str | Path) -> Config:
    """Load and validate configuration from a YAML file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")

    return Config(
        dataset=_parse_dataset(raw),
        federation=_parse_federation(raw),
        hierarchy=_parse_hierarchy(raw),
        data=_parse_data(raw),
        model=_parse_model(raw),
        peers=_parse_peers(raw),
        training=_parse_training(raw),
        experiment=_parse_experiment(raw),
    )


def save_config(cfg: Config, path: str | Path) -> Path:
    """Write a validated config to YAML (used by sweep runners)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(cfg), handle, default_flow_style=False, sort_keys=False)
    return out_path


def set_seed(seed: int) -> int:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs.

    Returns the same integer for passing to Flower simulation APIs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

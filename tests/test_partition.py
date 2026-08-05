"""Tests for non-IID latent-group data partitioning."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.config import Config, load_config
from src.data.partition import (
    mean_label_entropy,
    partition_dataset,
    plot_label_distribution,
)


def _cfg_with_alpha(alpha: float) -> Config:
    cfg = load_config("config.yaml")
    return replace(
        cfg,
        data=replace(cfg.data, alpha=alpha, partition_scheme="dirichlet_latent"),
        federation=replace(cfg.federation, num_clients=30),
    )


def _all_assigned_indices(client_datasets) -> list[int]:
    indices: list[int] = []
    for subset in client_datasets:
        indices.extend(subset.indices)
    return indices


def test_every_sample_assigned_exactly_once():
    cfg = _cfg_with_alpha(0.1)
    client_datasets, _, _ = partition_dataset(cfg)
    assigned = _all_assigned_indices(client_datasets)
    train_size = len(client_datasets[0].dataset)

    assert len(assigned) == train_size
    assert len(set(assigned)) == train_size
    assert min(assigned) >= 0
    assert max(assigned) < train_size


def test_client_count_correct():
    cfg = _cfg_with_alpha(0.5)
    client_datasets, group_labels, _ = partition_dataset(cfg)

    assert len(client_datasets) == cfg.federation.num_clients
    assert len(group_labels) == cfg.federation.num_clients
    assert all(len(subset) > 0 for subset in client_datasets)


def test_lower_alpha_increases_label_skew():
    entropies = {}
    for alpha in (1.0, 0.5, 0.1):
        cfg = _cfg_with_alpha(alpha)
        client_datasets, _, _ = partition_dataset(cfg)
        entropies[alpha] = mean_label_entropy(client_datasets)

    assert entropies[0.1] < entropies[0.5] < entropies[1.0]


def test_group_labels_have_expected_cardinality():
    cfg = _cfg_with_alpha(0.1)
    _, group_labels, _ = partition_dataset(cfg)

    unique_groups = set(group_labels)
    assert len(unique_groups) == cfg.data.num_latent_groups
    assert unique_groups == set(range(cfg.data.num_latent_groups))


def test_plot_label_distribution_writes_file(tmp_path):
    cfg = _cfg_with_alpha(0.1)
    client_datasets, group_labels, _ = partition_dataset(cfg)
    out_path = tmp_path / "label_dist.png"
    plot_label_distribution(client_datasets, group_labels, out_path)
    assert out_path.is_file()
    assert out_path.stat().st_size > 0

"""Experiment entrypoint: load config, run Flower simulation, write CSV logs."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from flwr.common import ndarrays_to_parameters
from flwr.server.server_config import ServerConfig
from flwr.simulation import start_simulation

from src.client import make_client_fn
from src.config import Config, load_config, set_seed
from src.data.partition import partition_dataset
from src.metrics.logging import RoundLogger
from src.models.cnn import create_model, get_parameters
from src.strategies.adaptive_peers import build_adaptive_peers_strategy
from src.strategies.flat_fedavg import build_flat_fedavg_strategy
from src.strategies.hier_static import build_hier_static_strategy


def _apply_cli_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    federation = cfg.federation
    experiment = cfg.experiment
    data = cfg.data
    hierarchy = cfg.hierarchy

    if args.num_rounds is not None:
        federation = replace(federation, num_rounds=args.num_rounds)
    if args.num_clients is not None:
        federation = replace(federation, num_clients=args.num_clients)
    if args.run_name is not None:
        experiment = replace(experiment, run_name=args.run_name)
    if args.strategy is not None:
        experiment = replace(experiment, strategy=args.strategy)
    if args.alpha is not None:
        data = replace(data, alpha=args.alpha)
    if args.global_sync_every is not None:
        hierarchy = replace(hierarchy, global_sync_every=args.global_sync_every)
    if args.model_distribution is not None:
        experiment = replace(experiment, model_distribution=args.model_distribution)

    return replace(
        cfg,
        federation=federation,
        experiment=experiment,
        data=data,
        hierarchy=hierarchy,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run FL simulation experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to experiment YAML config.",
    )
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--strategy", choices=["flat", "hier_static", "adaptive"], default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--global-sync-every", type=int, default=None)
    parser.add_argument(
        "--model-distribution",
        choices=["global", "cluster"],
        default=None,
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg = _apply_cli_overrides(cfg, args)
    set_seed(cfg.experiment.seed)

    client_datasets, group_labels, test_loader = partition_dataset(cfg)
    client_fn = make_client_fn(cfg, client_datasets)

    round_logger = RoundLogger(cfg.experiment.run_name, cfg.experiment.strategy)

    model = create_model(cfg)
    initial_parameters = ndarrays_to_parameters(get_parameters(model))

    strategy_name = cfg.experiment.strategy
    if strategy_name == "flat":
        strategy = build_flat_fedavg_strategy(
            cfg, test_loader, round_logger, initial_parameters
        )
    elif strategy_name == "hier_static":
        strategy = build_hier_static_strategy(
            cfg, test_loader, round_logger, group_labels, initial_parameters
        )
    elif strategy_name == "adaptive":
        strategy = build_adaptive_peers_strategy(
            cfg, test_loader, round_logger, group_labels, initial_parameters
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name!r}")

    server_config = ServerConfig(num_rounds=cfg.federation.num_rounds)

    start_simulation(
        client_fn=client_fn,
        num_clients=cfg.federation.num_clients,
        config=server_config,
        strategy=strategy,
    )

    csv_path = round_logger.write_csv()
    print(f"Results written to {csv_path}")


if __name__ == "__main__":
    main()

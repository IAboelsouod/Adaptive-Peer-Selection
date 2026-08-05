"""Multi-run experiment sweep driver (reuses ``src.run`` per configuration)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.config import Config, load_config, save_config
from src.metrics.logging import RUN_METADATA_COLUMNS

STRATEGIES = ("flat", "hier_static", "adaptive")
ALPHAS = (0.1, 0.5, 1.0)
GLOBAL_SYNC_EVERY = (1, 2, 4)
MODEL_DISTRIBUTIONS = ("global", "cluster")
SEEDS = (42, 43, 44)

REDUCED_ALPHAS = (0.1,)
REDUCED_GLOBAL_SYNC_EVERY = (1, 2)
REDUCED_SEEDS = (42, 43)


def make_run_name(
    strategy: str,
    alpha: float,
    global_sync_every: int,
    model_distribution: str,
    seed: int,
) -> str:
    """Deterministic run name encoding sweep coordinates."""
    alpha_s = str(alpha).replace(".", "p")
    return (
        f"sweep_{strategy}_a{alpha_s}_H{global_sync_every}"
        f"_{model_distribution}_s{seed}"
    )


def build_run_config(
    base: Config,
    *,
    strategy: str,
    alpha: float,
    global_sync_every: int,
    model_distribution: str,
    seed: int,
    num_rounds: int | None = None,
    num_clients: int | None = None,
) -> Config:
    """Return a sweep-specific config derived from the base template."""
    federation = base.federation
    if num_rounds is not None:
        federation = replace(federation, num_rounds=num_rounds)
    if num_clients is not None:
        federation = replace(federation, num_clients=num_clients)

    run_name = make_run_name(
        strategy, alpha, global_sync_every, model_distribution, seed
    )
    return replace(
        base,
        federation=federation,
        hierarchy=replace(base.hierarchy, global_sync_every=global_sync_every),
        data=replace(base.data, alpha=alpha),
        experiment=replace(
            base.experiment,
            strategy=strategy,  # type: ignore[arg-type]
            model_distribution=model_distribution,  # type: ignore[arg-type]
            seed=seed,
            run_name=run_name,
        ),
    )


def enrich_run_csv(csv_path: Path, cfg: Config) -> Path:
    """Attach sweep metadata columns so aggregation never relies on filenames."""
    df = pd.read_csv(csv_path)
    metadata = {
        "alpha": cfg.data.alpha,
        "global_sync_every": cfg.hierarchy.global_sync_every,
        "model_distribution": cfg.experiment.model_distribution,
        "seed": cfg.experiment.seed,
        "num_clients": cfg.federation.num_clients,
        "num_rounds": cfg.federation.num_rounds,
    }
    for column, value in metadata.items():
        df[column] = value

    # Stable column order: metadata first, then round metrics.
    metric_columns = [
        c for c in df.columns if c not in RUN_METADATA_COLUMNS and c != "run_name"
    ]
    ordered = ["run_name"] + RUN_METADATA_COLUMNS + metric_columns
    df = df[[c for c in ordered if c in df.columns]]
    df.to_csv(csv_path, index=False)
    return csv_path


def iter_sweep_grid(
    *,
    strategies: tuple[str, ...],
    alphas: tuple[float, ...],
    global_sync_every_values: tuple[int, ...],
    model_distributions: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[tuple[str, float, int, str, int]]:
    """Enumerate all sweep coordinates."""
    return list(
        product(
            strategies,
            alphas,
            global_sync_every_values,
            model_distributions,
            seeds,
        )
    )


def run_single(
    cfg: Config,
    *,
    base_config_path: Path,
    config_dir: Path,
    results_dir: Path,
    dry_run: bool = False,
) -> Path:
    """Write per-run config, invoke ``src.run``, enrich the output CSV."""
    config_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    run_cfg_path = config_dir / f"{cfg.experiment.run_name}.yaml"
    save_config(cfg, run_cfg_path)

    csv_path = results_dir / f"{cfg.experiment.run_name}.csv"
    if dry_run:
        print(f"[dry-run] would run {cfg.experiment.run_name} -> {csv_path}")
        return csv_path

    cmd = [
        sys.executable,
        "-m",
        "src.run",
        "--config",
        str(run_cfg_path),
    ]
    print(f"Running {cfg.experiment.run_name} ...")
    subprocess.run(cmd, check=True, cwd=Path(base_config_path).resolve().parent)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Expected results CSV missing after run: {csv_path}")
    return enrich_run_csv(csv_path, cfg)


def run_sweep(
    base_config_path: Path,
    grid: list[tuple[str, float, int, str, int]],
    *,
    num_rounds: int | None = None,
    num_clients: int | None = None,
    config_dir: Path = Path("results/configs"),
    results_dir: Path = Path("results"),
    dry_run: bool = False,
) -> list[Path]:
    """Execute every grid point and return enriched CSV paths."""
    base = load_config(base_config_path)
    outputs: list[Path] = []
    for strategy, alpha, h_value, distribution, seed in grid:
        cfg = build_run_config(
            base,
            strategy=strategy,
            alpha=alpha,
            global_sync_every=h_value,
            model_distribution=distribution,
            seed=seed,
            num_rounds=num_rounds,
            num_clients=num_clients,
        )
        outputs.append(
            run_single(
                cfg,
                base_config_path=base_config_path,
                config_dir=config_dir,
                results_dir=results_dir,
                dry_run=dry_run,
            )
        )
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a multi-configuration FL experiment sweep via src.run."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Base config template.",
    )
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument(
        "--reduced",
        action="store_true",
        help="Reduced sweep: alpha=0.1, H in {1,2}, 2 seeds, cluster distribution.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.reduced:
        grid = iter_sweep_grid(
            strategies=STRATEGIES,
            alphas=REDUCED_ALPHAS,
            global_sync_every_values=REDUCED_GLOBAL_SYNC_EVERY,
            model_distributions=("cluster",),
            seeds=REDUCED_SEEDS,
        )
    else:
        grid = iter_sweep_grid(
            strategies=STRATEGIES,
            alphas=ALPHAS,
            global_sync_every_values=GLOBAL_SYNC_EVERY,
            model_distributions=MODEL_DISTRIBUTIONS,
            seeds=SEEDS,
        )

    print(f"Sweep size: {len(grid)} runs")
    paths = run_sweep(
        args.config,
        grid,
        num_rounds=args.num_rounds,
        num_clients=args.num_clients,
        dry_run=args.dry_run,
    )
    print(f"Completed {len(paths)} runs.")


if __name__ == "__main__":
    main()

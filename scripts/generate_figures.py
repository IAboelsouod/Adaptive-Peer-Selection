"""Generate headline figures (A/B/C curves, peer graphs, summary table)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.viz.curves import generate_all_curves
from src.viz.graph import run_matched_short_experiments


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 6 headline figures.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--num-rounds", type=int, default=8)
    parser.add_argument("--num-clients", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-prefix", type=str, default="figures")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("results/peer_graphs"),
    )
    parser.add_argument(
        "--gif-path",
        type=Path,
        default=Path("results/peer_graph_evolution.gif"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
    )
    args = parser.parse_args(argv)

    csvs, snapshots, gif = run_matched_short_experiments(
        args.config,
        alpha=args.alpha,
        seed=args.seed,
        num_rounds=args.num_rounds,
        num_clients=args.num_clients,
        run_prefix=args.run_prefix,
        snapshot_dir=args.snapshot_dir,
        gif_path=args.gif_path,
    )
    curve_paths = generate_all_curves(csvs, results_dir=args.results_dir)

    print("Figures generated:")
    for name, path in sorted({**csvs, **curve_paths, "gif": gif}.items()):
        print(f"  {name}: {path}")
    print("Peer-graph snapshots:")
    for path in snapshots:
        print(f"  {path}")


if __name__ == "__main__":
    main()

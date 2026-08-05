"""Aggregate per-run CSV logs into master comparison tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.metrics.logging import RUN_METADATA_COLUMNS, STABLE_ROUND_COLUMNS

# Per-run summary columns aggregated into the master table.
SUMMARY_COLUMNS = [
    "run_name",
    "strategy",
    "alpha",
    "global_sync_every",
    "model_distribution",
    "seed",
    "num_clients",
    "num_rounds",
    "final_global_acc",
    "final_ari",
    "final_nmi",
    "rounds_to_target",
    "target_acc",
    "client_edge_transfers",
    "edge_global_transfers",
    "comm_cost",
    "improvement_vs_flat",
]

MASTER_GROUP_COLUMNS = [
    "strategy",
    "alpha",
    "global_sync_every",
    "model_distribution",
]

DEFAULT_TARGET_FRAC = 0.9
CHEAP_TRANSFER_WEIGHT = 1.0
EXPENSIVE_TRANSFER_WEIGHT = 10.0

EXCLUDE_CSV_NAMES = {
    "master_table.csv",
    "summary.csv",
}


def count_sync_rounds(num_rounds: int, global_sync_every: int) -> int:
    """Count global-sync rounds in ``0 .. num_rounds`` inclusive."""
    return sum(1 for r in range(0, num_rounds + 1) if r % global_sync_every == 0)


def communication_cost(
    strategy: str,
    *,
    num_clients: int,
    num_rounds: int,
    global_sync_every: int,
    num_groups: int,
    cheap_weight: float = CHEAP_TRANSFER_WEIGHT,
    expensive_weight: float = EXPENSIVE_TRANSFER_WEIGHT,
) -> tuple[int, int, float]:
    """Return (client_edge_transfers, edge_global_transfers, weighted comm_cost).

    Flat FedAvg: all client<->global traffic is expensive (no edge tier).
    Hierarchical B/C: client<->edge every round (cheap); edge<->global every
    ``H`` rounds (expensive), following Abad et al.'s latency split.
    """
    if strategy == "flat":
        client_edge = 0
        edge_global = 2 * num_clients * num_rounds
        total = edge_global * expensive_weight
        return client_edge, edge_global, float(total)

    client_edge = 2 * num_clients * num_rounds
    sync_rounds = count_sync_rounds(num_rounds, global_sync_every)
    edge_global = 2 * num_groups * sync_rounds
    total = client_edge * cheap_weight + edge_global * expensive_weight
    return client_edge, edge_global, float(total)


def load_run_csv(path: Path) -> pd.DataFrame:
    """Load one run CSV aligned to :data:`STABLE_ROUND_COLUMNS`."""
    df = pd.read_csv(path)
    for column in STABLE_ROUND_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    return df[STABLE_ROUND_COLUMNS].copy()


def load_all_runs(results_dir: Path) -> pd.DataFrame:
    """Concatenate all per-run CSVs on the stable round-level schema."""
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(results_dir.glob("*.csv")):
        if csv_path.name in EXCLUDE_CSV_NAMES:
            continue
        frames.append(load_run_csv(csv_path))
    if not frames:
        raise FileNotFoundError(f"No run CSV files found in {results_dir}")
    return pd.concat(frames, ignore_index=True, sort=False)


def _final_metric(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return float("nan")
    return float(values.iloc[-1])


def _rounds_to_target(df: pd.DataFrame, target_acc: float) -> int | None:
    hits = df[df["global_acc"].notna() & (df["global_acc"] >= target_acc)]
    if hits.empty:
        return None
    return int(hits.iloc[0]["round"])


def flat_baseline_final_acc(
    summaries: pd.DataFrame,
    *,
    alpha: float,
    seed: int,
    model_distribution: str,
) -> float:
    """Final accuracy of the matched flat run for convergence targeting."""
    mask = (
        (summaries["strategy"] == "flat")
        & (summaries["alpha"] == alpha)
        & (summaries["seed"] == seed)
        & (summaries["model_distribution"] == model_distribution)
    )
    rows = summaries.loc[mask, "final_global_acc"].dropna()
    if rows.empty:
        return float("nan")
    return float(rows.iloc[0])


def summarize_runs(
    rounds_df: pd.DataFrame,
    *,
    target_frac: float = DEFAULT_TARGET_FRAC,
) -> pd.DataFrame:
    """Build per-run summary rows with convergence and communication metrics."""
    complete = rounds_df[rounds_df["alpha"].notna()].copy()
    if complete.empty:
        raise ValueError(
            "No runs with sweep metadata (alpha column). "
            "Run scripts/experiments.py first."
        )

    summaries: list[dict[str, object]] = []
    for run_name, group in complete.groupby("run_name", sort=True):
        row = group.iloc[0]
        strategy = str(row["strategy"])
        alpha = float(row["alpha"])
        seed = int(row["seed"])
        distribution = str(row["model_distribution"])
        h_value = int(row["global_sync_every"])
        num_clients = int(row["num_clients"])
        num_rounds = int(row["num_rounds"])

        groups_metric = _final_metric(group["num_groups"])
        num_groups = int(groups_metric) if groups_metric == groups_metric and groups_metric > 0 else 1

        final_acc = _final_metric(group["global_acc"])
        final_ari = _final_metric(group["ari"])
        final_nmi = _final_metric(group["nmi"])

        ce, eg, cost = communication_cost(
            strategy,
            num_clients=num_clients,
            num_rounds=num_rounds,
            global_sync_every=h_value,
            num_groups=num_groups,
        )

        summaries.append(
            {
                "run_name": run_name,
                "strategy": strategy,
                "alpha": alpha,
                "global_sync_every": h_value,
                "model_distribution": distribution,
                "seed": seed,
                "num_clients": num_clients,
                "num_rounds": num_rounds,
                "final_global_acc": final_acc,
                "final_ari": final_ari,
                "final_nmi": final_nmi,
                "rounds_to_target": None,
                "target_acc": None,
                "client_edge_transfers": ce,
                "edge_global_transfers": eg,
                "comm_cost": cost,
                "improvement_vs_flat": None,
            }
        )

    summary_df = pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)

    for idx, run in summary_df.iterrows():
        target_acc = flat_baseline_final_acc(
            summary_df,
            alpha=float(run["alpha"]),
            seed=int(run["seed"]),
            model_distribution=str(run["model_distribution"]),
        )
        if target_acc == target_acc:
            target = target_frac * target_acc
            run_rounds = complete[complete["run_name"] == run["run_name"]]
            summary_df.at[idx, "target_acc"] = round(target, 6)
            summary_df.at[idx, "rounds_to_target"] = _rounds_to_target(
                run_rounds, target
            )

    for idx, run in summary_df.iterrows():
        flat_mask = (
            (summary_df["strategy"] == "flat")
            & (summary_df["alpha"] == run["alpha"])
            & (summary_df["seed"] == run["seed"])
            & (summary_df["model_distribution"] == run["model_distribution"])
        )
        flat_costs = summary_df.loc[flat_mask, "comm_cost"].dropna()
        if flat_costs.empty or not run["comm_cost"]:
            continue
        summary_df.at[idx, "improvement_vs_flat"] = float(
            flat_costs.iloc[0] / float(run["comm_cost"])
        )

    return summary_df


def _mean_std(series: pd.Series) -> str:
    values = series.dropna()
    if values.empty:
        return ""
    if len(values) == 1:
        return f"{values.mean():.4f}"
    return f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"


def build_master_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-run summaries: mean ± std per sweep coordinate."""
    rows: list[dict[str, object]] = []
    grouped = summary_df.groupby(MASTER_GROUP_COLUMNS, dropna=False, sort=True)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(MASTER_GROUP_COLUMNS, keys, strict=True))
        record["n_seeds"] = len(group)
        record["final_global_acc"] = _mean_std(group["final_global_acc"])
        record["final_ari"] = _mean_std(group["final_ari"])
        record["final_nmi"] = _mean_std(group["final_nmi"])
        record["rounds_to_target"] = _mean_std(
            group["rounds_to_target"].astype(float)
        )
        record["client_edge_transfers"] = _mean_std(
            group["client_edge_transfers"].astype(float)
        )
        record["edge_global_transfers"] = _mean_std(
            group["edge_global_transfers"].astype(float)
        )
        record["comm_cost"] = _mean_std(group["comm_cost"].astype(float))
        record["improvement_vs_flat"] = _mean_std(
            group["improvement_vs_flat"].astype(float)
        )

        # Accuracy-vs-H trade-off within each (strategy, alpha, distribution) slice.
        h_acc = (
            group.groupby("global_sync_every", sort=True)["final_global_acc"]
            .mean()
            .round(4)
        )
        record["accuracy_by_H"] = "; ".join(
            f"H={int(h)}:{acc:.4f}" for h, acc in h_acc.items()
        )
        rows.append(record)

    return pd.DataFrame(rows)


def master_table_to_markdown(master_df: pd.DataFrame) -> str:
    """Render master table as a markdown pipe table."""
    headers = list(master_df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in master_df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines) + "\n"


def aggregate_results(
    results_dir: Path,
    *,
    target_frac: float = DEFAULT_TARGET_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """Load all runs, summarize, and write master CSV + markdown."""
    rounds_df = load_all_runs(results_dir)
    summary_df = summarize_runs(rounds_df, target_frac=target_frac)
    master_df = build_master_table(summary_df)

    csv_path = results_dir / "master_table.csv"
    md_path = results_dir / "master_table.md"
    master_df.to_csv(csv_path, index=False)
    md_path.write_text(master_table_to_markdown(master_df), encoding="utf-8")
    return summary_df, master_df, csv_path, md_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate FL sweep CSVs into master_table.csv."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing per-run CSV logs.",
    )
    parser.add_argument(
        "--target-frac",
        type=float,
        default=DEFAULT_TARGET_FRAC,
        help="Convergence target as fraction of flat final accuracy.",
    )
    args = parser.parse_args(argv)

    _, master_df, csv_path, md_path = aggregate_results(
        args.results_dir, target_frac=args.target_frac
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(master_df.to_csv(index=False))


if __name__ == "__main__":
    main()

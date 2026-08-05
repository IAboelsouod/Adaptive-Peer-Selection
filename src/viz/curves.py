"""Matplotlib curves and convergence summary tables from experiment CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

STRATEGY_LABELS = {
    "flat": "A — Flat FedAvg",
    "hier_static": "B — Static HFL",
    "adaptive": "C — Adaptive (ours)",
}

SYSTEM_LABELS = {
    "A": "A — Flat FedAvg",
    "B": "B — Static HFL",
    "C": "C — Adaptive (ours)",
}


def plot_accuracy(csvs: dict[str, Path], out_path: str | Path) -> Path:
    """Overlay global accuracy vs round for runs A/B/C from matched CSV logs."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}

    for key, csv_path in csvs.items():
        df = pd.read_csv(csv_path)
        label = SYSTEM_LABELS.get(key, df["strategy"].iloc[0])
        rounds = df["round"]
        acc = df["global_acc"]
        ax.plot(
            rounds,
            acc,
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=label,
            color=colors.get(key),
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("Global test accuracy")
    ax.set_title("Global accuracy — A / B / C comparison")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0.0, top=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_cluster_quality(csv_path: str | Path, out_path: str | Path) -> Path:
    """Plot ARI and NMI vs round for the adaptive run."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(9, 5))
    rounds = df["round"]
    ax.plot(rounds, df["ari"], marker="o", markersize=4, linewidth=1.8, label="ARI")
    ax.plot(rounds, df["nmi"], marker="s", markersize=4, linewidth=1.8, label="NMI")
    ax.set_xlabel("Round")
    ax.set_ylabel("Score")
    ax.set_title("Cluster recovery quality (adaptive run)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.1, top=1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _rounds_to_target(df: pd.DataFrame, target_acc: float) -> int | None:
    """First round where global_acc reaches target, or None."""
    hits = df[df["global_acc"].notna() & (df["global_acc"] >= target_acc)]
    if hits.empty:
        return None
    return int(hits.iloc[0]["round"])


def plot_convergence_table(
    csvs: dict[str, Path],
    out_dir: str | Path = "results",
    *,
    target_frac: float = 0.9,
) -> tuple[Path, Path, Path]:
    """Build summary table: rounds-to-target, final acc, final ARI.

    Writes ``results/summary.csv``, a rendered PNG table, and a markdown table.
    Target accuracy is ``target_frac`` × final flat-baseline accuracy.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    flat_df = pd.read_csv(csvs["A"])
    flat_accs = flat_df["global_acc"].dropna()
    flat_final = float(flat_accs.iloc[-1]) if not flat_accs.empty else float("nan")
    target_acc = flat_final * target_frac if flat_final == flat_final else float("nan")

    rows: list[dict[str, object]] = []
    for key, csv_path in csvs.items():
        df = pd.read_csv(csv_path)
        strategy = str(df["strategy"].iloc[0])
        accs = df["global_acc"].dropna()
        final_acc = float(accs.iloc[-1]) if not accs.empty else float("nan")
        rtt = (
            _rounds_to_target(df, target_acc)
            if target_acc == target_acc
            else None
        )
        ari_series = df["ari"].dropna()
        final_ari = float(ari_series.iloc[-1]) if not ari_series.empty else float("nan")

        rows.append(
            {
                "system": key,
                "strategy": strategy,
                "label": SYSTEM_LABELS.get(key, strategy),
                "rounds_to_target": rtt,
                "target_acc": round(target_acc, 4) if target_acc == target_acc else None,
                "final_global_acc": round(final_acc, 4) if final_acc == final_acc else None,
                "final_ari": round(final_ari, 4) if final_ari == final_ari else None,
            }
        )

    summary = pd.DataFrame(rows)
    csv_out = out_dir / "summary.csv"
    summary.to_csv(csv_out, index=False)

    display = summary[
        ["label", "rounds_to_target", "target_acc", "final_global_acc", "final_ari"]
    ].copy()
    display.columns = [
        "System",
        "Rounds to target",
        "Target acc",
        "Final acc",
        "Final ARI",
    ]

    fig, ax = plt.subplots(figsize=(10, 1.2 + 0.45 * len(display)))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)
    png_out = out_dir / "summary_table.png"
    fig.tight_layout()
    fig.savefig(png_out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    md_out = out_dir / "summary_table.md"
    md_lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for _, row in display.iterrows():
        md_lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    md_out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return csv_out, png_out, md_out


def generate_all_curves(
    csvs: dict[str, Path],
    *,
    results_dir: str | Path = "results",
) -> dict[str, Path]:
    """Generate accuracy overlay, cluster-quality plot, and summary table."""
    results_dir = Path(results_dir)
    outputs: dict[str, Path] = {}
    outputs["accuracy"] = plot_accuracy(
        csvs, results_dir / "accuracy_abc_overlay.png"
    )
    outputs["cluster_quality"] = plot_cluster_quality(
        csvs["C"], results_dir / "cluster_quality_adaptive.png"
    )
    summary_csv, summary_png, summary_md = plot_convergence_table(csvs, results_dir)
    outputs["summary_csv"] = summary_csv
    outputs["summary_png"] = summary_png
    outputs["summary_md"] = summary_md
    return outputs

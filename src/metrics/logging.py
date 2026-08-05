"""Per-round experiment metrics logger."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

COLUMNS = [
    "run_name",
    "strategy",
    "round",
    "global_acc",
    "global_loss",
    "num_groups",
    "ari",
    "nmi",
    "wall_time",
]

# Sweep metadata columns appended by scripts/experiments.py (not by RoundLogger).
RUN_METADATA_COLUMNS = [
    "alpha",
    "global_sync_every",
    "model_distribution",
    "seed",
    "num_clients",
    "num_rounds",
]

STABLE_ROUND_COLUMNS = ["run_name", "strategy", *RUN_METADATA_COLUMNS, *COLUMNS[2:]]


class RoundLogger:
    """Accumulate per-round metrics and write a stable-schema CSV."""

    def __init__(
        self,
        run_name: str,
        strategy: str,
        results_dir: str | Path = "results",
    ) -> None:
        self.run_name = run_name
        self.strategy = strategy
        self.results_dir = Path(results_dir)
        self._rows: list[dict[str, object]] = []
        self._start_time = time.perf_counter()

    def log_round(
        self,
        *,
        round: int,
        global_acc: float | None,
        global_loss: float | None,
        num_groups: int | None = None,
        ari: float | None = None,
        nmi: float | None = None,
        wall_time: float | None = None,
    ) -> None:
        """Append one row; unspecified cluster metrics stay null."""
        if wall_time is None:
            wall_time = time.perf_counter() - self._start_time
        self._rows.append(
            {
                "run_name": self.run_name,
                "strategy": self.strategy,
                "round": round,
                "global_acc": global_acc,
                "global_loss": global_loss,
                "num_groups": num_groups,
                "ari": ari,
                "nmi": nmi,
                "wall_time": wall_time,
            }
        )

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return accumulated rows as a DataFrame with a fixed column order."""
        return pd.DataFrame(self._rows, columns=COLUMNS)

    def write_csv(self) -> Path:
        """Write ``results/<run_name>.csv`` and return the output path."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.results_dir / f"{self.run_name}.csv"
        self.dataframe.to_csv(out_path, index=False)
        return out_path

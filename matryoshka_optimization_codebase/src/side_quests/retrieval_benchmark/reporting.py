from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List
import json

import pandas as pd


def to_long_metrics(rows: List[Dict]) -> pd.DataFrame:
    out = []
    for row in rows:
        for metric, value in row["metrics"].items():
            if metric == "name":
                continue
            out.append(
                {
                    "model_id": row["model_id"],
                    "model_type": row["model_type"],
                    "dimension": row["dimension"],
                    "metric": metric,
                    "value": float(value),
                }
            )
    return pd.DataFrame(out)


def to_wide_metrics(long_df: pd.DataFrame) -> pd.DataFrame:
    return long_df.pivot_table(
        index=["model_id", "model_type", "dimension"],
        columns="metric",
        values="value",
    ).reset_index()


def build_ranking_summary(wide_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in wide_df.columns if c not in {"model_id", "model_type", "dimension"}]
    rows = []
    for metric in metrics:
        best = wide_df.sort_values(metric, ascending=False).iloc[0]
        rows.append(
            {
                "metric": metric,
                "best_model_id": best["model_id"],
                "best_dimension": int(best["dimension"]),
                "best_value": float(best[metric]),
            }
        )
    return pd.DataFrame(rows)


def save_outputs(
    *,
    output_dir: Path,
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    metadata: Dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(output_dir / "metrics_long.csv", index=False)
    wide_df.to_csv(output_dir / "metrics_wide.csv", index=False)
    summary_df.to_csv(output_dir / "ranking_summary.csv", index=False)

    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

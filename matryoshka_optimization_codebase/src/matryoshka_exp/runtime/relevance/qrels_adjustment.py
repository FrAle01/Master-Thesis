from __future__ import annotations

import pandas as pd

from .calibration import to_unit_scale


def apply_qrels_hard_override(pair_df: pd.DataFrame, qrels: pd.DataFrame, *, label_column: str, scale_cfg) -> tuple[pd.DataFrame, int]:
    q = qrels.loc[:, ["qid", "docno", label_column]].copy()
    q["qid"] = q["qid"].astype(str)
    q["docno"] = q["docno"].astype(str)
    q[label_column] = pd.to_numeric(q[label_column], errors="coerce")
    q = q.dropna(subset=[label_column])

    out = pair_df.merge(q, on=["qid", "docno"], how="left")
    judged = out[label_column].notna()

    if scale_cfg.mode == "explicit_map":
        mapped = out[label_column].astype("Int64").astype(str).map(scale_cfg.explicit_map)
        scaled = pd.to_numeric(mapped, errors="coerce").fillna(out["relevance_final"])
    else:
        scaled = to_unit_scale(
            pd.to_numeric(out[label_column], errors="coerce").fillna(scale_cfg.min_label),
            min_value=float(scale_cfg.min_label),
            max_value=float(scale_cfg.max_label),
        )

    out.loc[judged, "relevance_final"] = scaled.loc[judged].astype(float)
    out.loc[judged, "is_qrel_overridden"] = True
    out = out.drop(columns=[label_column])
    return out, int(judged.sum())

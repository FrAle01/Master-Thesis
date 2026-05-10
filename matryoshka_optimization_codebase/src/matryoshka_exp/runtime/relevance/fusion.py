from __future__ import annotations

import pandas as pd


def select_uncertain_high_impact_pairs(pair_df: pd.DataFrame, *, fraction: float) -> pd.DataFrame:
    if pair_df.empty:
        return pair_df
    work = pair_df.copy()
    work["impact"] = work["relevance_final"].abs() * (1.0 + work["uncertainty"].fillna(1.0))
    n = max(1, int(len(work) * fraction))
    return work.sort_values(["uncertainty", "impact"], ascending=[False, False]).head(n).loc[:, ["qid", "docno"]]


def merge_refined_scores(base_df: pd.DataFrame, refined_df: pd.DataFrame, *, source_mode: str) -> pd.DataFrame:
    if refined_df.empty:
        return base_df
    key = ["qid", "docno"]
    upd = refined_df.set_index(key)
    out = base_df.set_index(key)
    common = out.index.intersection(upd.index)
    out.loc[common, ["relevance_estimated", "relevance_final", "confidence", "uncertainty"]] = upd.loc[
        common, ["relevance_estimated", "relevance_final", "confidence", "uncertainty"]
    ]
    out.loc[common, "source_mode"] = source_mode
    return out.reset_index()

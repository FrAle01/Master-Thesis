from __future__ import annotations

import numpy as np
import pandas as pd


def to_unit_scale(values: pd.Series, *, min_value: float, max_value: float) -> pd.Series:
    denom = max(max_value - min_value, 1e-9)
    clipped = values.clip(lower=min_value, upper=max_value)
    return (clipped - min_value) / denom


def calibrate_scores(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scores.size == 0:
        return scores, scores, scores
    s_min = float(np.min(scores))
    s_max = float(np.max(scores))
    denom = max(s_max - s_min, 1e-9)
    rel = (scores - s_min) / denom
    margin = np.zeros_like(rel)
    if rel.size > 1:
        sorted_rel = np.sort(rel)[::-1]
        pivot = float(sorted_rel[min(1, len(sorted_rel) - 1)])
        margin = np.abs(rel - pivot)
    confidence = np.clip(margin, 0.0, 1.0)
    return rel, confidence, 1.0 - confidence


def calibrate_rank_log_discount(ranks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ranks.size == 0:
        return ranks.astype(float), ranks.astype(float), ranks.astype(float)
    r = np.asarray(ranks, dtype=float)
    # Discount on 0-based ranks: rank=0 -> 1/log2(2)=1.0
    rel = 1.0 / np.log2(r + 1.0)
    rel = np.clip(rel, 0.0, 1.0)
    margin = np.zeros_like(rel)
    if rel.size > 1:
        sorted_rel = np.sort(rel)[::-1]
        pivot = float(sorted_rel[min(1, len(sorted_rel) - 1)])
        margin = np.abs(rel - pivot)
    confidence = np.clip(margin, 0.0, 1.0)
    return rel, confidence, 1.0 - confidence

from __future__ import annotations

import numpy as np

from ..legacy.score_metrics import (
    compute_pointwise_utility,
    hybrid_score_margin_utility,
    relative_margin_utility,
)


def compute_score_preservation(
    *,
    metric: str,
    full_scores: np.ndarray,
    reduced_scores: np.ndarray,
    epsilon: float,
    alpha: float,
    margin_negatives: int,
) -> np.ndarray:
    if metric in {"relative_score_dissimilarity", "absolute_score_utility", "squared_score_utility"}:
        return np.asarray(
            compute_pointwise_utility(
                metric,
                full_scores,
                reduced_scores,
                epsilon=epsilon,
                alpha=alpha,
            ),
            dtype=float,
        )

    if full_scores.size <= 1:
        # Margin-based metrics are undefined with one candidate; keep neutral preservation.
        return np.ones_like(full_scores, dtype=float)

    n = int(full_scores.shape[0])
    full_neg = np.zeros(n, dtype=float)
    reduced_neg = np.zeros(n, dtype=float)
    for i in range(n):
        full_others = np.delete(full_scores, i)
        reduced_others = np.delete(reduced_scores, i)
        k = min(margin_negatives, full_others.size)
        full_neg[i] = float(np.mean(np.sort(full_others)[-k:]))
        reduced_neg[i] = float(np.mean(np.sort(reduced_others)[-k:]))

    if metric == "relative_margin_utility":
        return np.asarray(
            relative_margin_utility(
                full_scores,
                full_neg,
                reduced_scores,
                reduced_neg,
                epsilon=epsilon,
            ),
            dtype=float,
        )
    if metric == "hybrid_score_margin_utility":
        return np.asarray(
            hybrid_score_margin_utility(
                full_scores,
                reduced_scores,
                full_neg,
                reduced_neg,
                alpha=alpha,
                epsilon=epsilon,
            ),
            dtype=float,
        )
    raise ValueError(f"Unsupported utility metric: {metric}")

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class UtilityMetricConfig:
    name: str
    epsilon: float = 1e-6
    alpha: float = 0.7


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def relative_score_dissimilarity(full_scores, reduced_scores, epsilon: float = 1e-6):
    denom = np.maximum(np.abs(full_scores), epsilon)
    loss = np.abs(full_scores - reduced_scores) / denom
    return 1.0 - _clip01(loss)


def absolute_score_utility(full_scores, reduced_scores):
    loss = np.abs(full_scores - reduced_scores)
    # Convert an unbounded loss into a bounded utility.
    return 1.0 / (1.0 + loss)


def squared_score_utility(full_scores, reduced_scores):
    loss = np.square(full_scores - reduced_scores)
    return 1.0 / (1.0 + loss)


def relative_margin_utility(full_pos, full_neg, red_pos, red_neg, epsilon: float = 1e-6):
    full_margin = full_pos - full_neg
    red_margin = red_pos - red_neg
    denom = np.maximum(np.abs(full_margin), epsilon)
    loss = np.abs(full_margin - red_margin) / denom
    return 1.0 - _clip01(loss)


def hybrid_score_margin_utility(
    full_scores,
    reduced_scores,
    full_neg_scores,
    reduced_neg_scores,
    *,
    alpha: float = 0.7,
    epsilon: float = 1e-6,
):
    score_u = relative_score_dissimilarity(full_scores, reduced_scores, epsilon=epsilon)
    margin_u = relative_margin_utility(full_scores, full_neg_scores, reduced_scores, reduced_neg_scores, epsilon=epsilon)
    return alpha * score_u + (1.0 - alpha) * margin_u


def compute_pointwise_utility(metric_name: str, full_scores, reduced_scores, *, epsilon: float, alpha: float):
    if metric_name == "relative_score_dissimilarity":
        return relative_score_dissimilarity(full_scores, reduced_scores, epsilon=epsilon)
    if metric_name == "absolute_score_utility":
        return absolute_score_utility(full_scores, reduced_scores)
    if metric_name == "squared_score_utility":
        return squared_score_utility(full_scores, reduced_scores)
    raise ValueError(f"Unsupported pointwise utility metric: {metric_name}")

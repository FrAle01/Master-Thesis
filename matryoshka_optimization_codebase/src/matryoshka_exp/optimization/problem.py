from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ..models.base import RepresentationProfile
from .errors import InfeasibleOptimizationError


@dataclass(frozen=True)
class AssignmentProblem:
    profile_names: tuple[str, ...]
    cost_vector: np.ndarray
    utility_matrix: np.ndarray
    docnos: np.ndarray
    budget_bytes: int

    @property
    def num_docs(self) -> int:
        return int(self.utility_matrix.shape[0])

    @property
    def num_profiles(self) -> int:
        return int(self.utility_matrix.shape[1])

    def evaluate(self, chosen_idx: np.ndarray) -> tuple[int, float]:
        rows = np.arange(self.num_docs)
        return (
            int(self.cost_vector[chosen_idx].sum()),
            float(self.utility_matrix[rows, chosen_idx].sum()),
        )

    def to_assignments(self, chosen_idx: np.ndarray) -> pd.DataFrame:
        rows = np.arange(self.num_docs)
        return pd.DataFrame(
            {
                "docno": self.docnos,
                "profile": [self.profile_names[idx] for idx in chosen_idx],
                "utility": self.utility_matrix[rows, chosen_idx],
                "cost_bytes": self.cost_vector[chosen_idx].astype(int),
            }
        )


def build_assignment_problem(
    utility_table: pd.DataFrame,
    profiles: Sequence[RepresentationProfile],
    budget_bytes: int,
) -> AssignmentProblem:
    required = {"docno", "profile", "utility"}
    missing = required.difference(utility_table.columns)
    if missing:
        raise ValueError(f"Utility table is missing columns: {sorted(missing)}")
    if not profiles:
        raise ValueError("At least one representation profile is required.")
    if budget_bytes < 0:
        raise ValueError("budget_bytes must be non-negative.")

    profile_names = tuple(profile.name for profile in profiles)
    if len(set(profile_names)) != len(profile_names):
        raise ValueError("Profile names must be unique.")

    cost_vector = np.asarray([profile.cost_bytes for profile in profiles], dtype=np.int64)
    if np.any(cost_vector <= 0):
        raise ValueError("All profile costs must be positive.")

    pivot = utility_table.pivot(index="docno", columns="profile", values="utility")
    unknown_profiles = set(pivot.columns).difference(profile_names)
    if unknown_profiles:
        raise ValueError(f"Utility table contains unknown profiles: {sorted(unknown_profiles)}")
    pivot = pivot.reindex(columns=profile_names)
    if pivot.empty:
        raise ValueError("Utility table must contain at least one document.")
    if pivot.isnull().any().any():
        raise ValueError("Utility table must contain one utility value per (docno, profile).")

    utility_matrix = pivot.to_numpy(dtype=np.float64)
    if not np.isfinite(utility_matrix).all():
        raise ValueError("Utility table contains non-finite utility values.")

    problem = AssignmentProblem(
        profile_names=profile_names,
        cost_vector=cost_vector,
        utility_matrix=utility_matrix,
        docnos=pivot.index.to_numpy(),
        budget_bytes=int(budget_bytes),
    )
    minimum_total_cost = problem.num_docs * int(np.min(cost_vector))
    if minimum_total_cost > problem.budget_bytes:
        raise InfeasibleOptimizationError(
            budget_bytes=problem.budget_bytes,
            assigned_cost_bytes=minimum_total_cost,
        )
    return problem

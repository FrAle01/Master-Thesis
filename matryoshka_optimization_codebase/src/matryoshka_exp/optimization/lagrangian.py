from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tqdm

from ..models.base import RepresentationProfile


@dataclass
class OptimizationResult:
    assignments: pd.DataFrame
    lambda_star: float
    total_cost_bytes: int
    total_utility: float
    feasible: bool


class LagrangianProfileOptimizer:
    """Solve the relaxed memory-constrained assignment problem with a dual search.

    For a fixed lambda, each document independently selects the profile that maximizes:
        utility(doc, profile) - lambda * cost(profile)

    This is the classical document-wise decomposition induced by the Lagrangian relaxation.
    """

    def __init__(
        self,
        profiles: List[RepresentationProfile],
        budget_bytes: int,
        max_iter: int,
        tolerance: float,
        lambda_low: float = 0.0,
        lambda_high: float = 1.0,
    ):
        self.profiles = profiles
        self.budget_bytes = int(budget_bytes)
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.lambda_low = float(lambda_low)
        self.lambda_high = float(lambda_high)
        if self.lambda_low < 0 or self.lambda_high < self.lambda_low:
            raise ValueError("Invalid lambda bounds: require 0 <= lambda_low <= lambda_high.")
        self.profile_names = [p.name for p in profiles]
        self.cost_lookup = {p.name: p.cost_bytes for p in profiles}

    def solve(self, utility_table: pd.DataFrame) -> OptimizationResult:
        required = {"docno", "profile", "utility"}
        missing = required.difference(utility_table.columns)
        if missing:
            raise ValueError(f"Utility table is missing columns: {sorted(missing)}")

        pivot = utility_table.pivot(index="docno", columns="profile", values="utility").reindex(columns=self.profile_names)
        if pivot.isnull().any().any():
            raise ValueError("Utility table must contain one utility value per (docno, profile).")

        cost_vector = np.array([self.cost_lookup[name] for name in self.profile_names], dtype=np.float64)
        utility_matrix = pivot.values.astype(np.float64)
        docnos = pivot.index.to_numpy()

        lambda_low = self.lambda_low
        dynamic_high = max(1.0, float(np.max(utility_matrix) / max(np.min(cost_vector), 1.0)))
        lambda_high = max(self.lambda_high, dynamic_high)
        self.logger.info("Starting Lagrangian optimization with lambda_low=%.6f lambda_high=%.6f (dynamic_high=%.6f)", lambda_low, lambda_high, dynamic_high)
        
        best_assignments = None
        best_cost = None
        best_lambda = 0.0

        for _ in tqdm(range(self.max_iter), desc="Lagrangian optimization"):
            lam = 0.5 * (lambda_low + lambda_high)
            reduced = utility_matrix - lam * cost_vector[None, :]
            chosen_idx = np.argmax(reduced, axis=1)
            chosen_cost = int(cost_vector[chosen_idx].sum())

            if best_assignments is None or abs(chosen_cost - self.budget_bytes) < abs(best_cost - self.budget_bytes):
                best_assignments = chosen_idx.copy()
                best_cost = chosen_cost
                best_lambda = lam

            relative_gap = abs(chosen_cost - self.budget_bytes) / max(self.budget_bytes, 1)
            if relative_gap <= self.tolerance:
                best_assignments = chosen_idx.copy()
                best_cost = chosen_cost
                best_lambda = lam
                break

            if chosen_cost > self.budget_bytes:
                lambda_low = lam
            else:
                lambda_high = lam

        assigned_profiles = [self.profile_names[idx] for idx in best_assignments]
        assigned_utilities = utility_matrix[np.arange(len(docnos)), best_assignments]
        assigned_costs = cost_vector[best_assignments]

        assignments = pd.DataFrame(
            {
                "docno": docnos,
                "profile": assigned_profiles,
                "utility": assigned_utilities,
                "cost_bytes": assigned_costs.astype(int),
            }
        )

        feasible = int(assignments["cost_bytes"].sum()) <= self.budget_bytes
        self.logger.info("Lagrangian optimization complete. feasible=%s", feasible)
        
        return OptimizationResult(
            assignments=assignments,
            lambda_star=float(best_lambda),
            total_cost_bytes=int(assignments["cost_bytes"].sum()),
            total_utility=float(assignments["utility"].sum()),
            feasible=feasible,
        )

    def choose_profile_online(self, doc_profile_utility: Dict[str, float], lambda_value: float) -> str:
        if not self.profiles:
            raise ValueError("No profiles are available for online profile selection.")
        best_profile = None
        best_value = None
        for profile in self.profiles:
            if profile.name not in doc_profile_utility:
                raise ValueError(f"Missing utility value for profile `{profile.name}`.")
            value = float(doc_profile_utility[profile.name]) - lambda_value * profile.cost_bytes
            if best_value is None or value > best_value:
                best_value = value
                best_profile = profile.name
        if best_profile is None:
            raise RuntimeError("Failed to select an online profile despite non-empty profile catalog.")
        return best_profile

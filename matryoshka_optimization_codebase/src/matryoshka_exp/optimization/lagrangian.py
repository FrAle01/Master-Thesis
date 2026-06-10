from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..models.base import RepresentationProfile
from .candidates import (
    AssignmentCandidate,
    UtilityFirstCandidatePolicy,
    make_candidate,
)
from .assignment_repair import GreedyAssignmentRepair
from .contracts import AssignmentRepairStrategy, CandidatePolicy, RelaxationSearchStrategy
from .problem import AssignmentProblem, build_assignment_problem
from .relaxation_search import LagrangianRelaxationSearch


@dataclass
class OptimizationResult:
    assignments: pd.DataFrame
    lambda_star: float
    total_cost_bytes: int
    total_utility: float
    feasible: bool
    pre_repair_relaxed_cost_bytes: Optional[int] = None
    pre_repair_relaxed_utility: Optional[float] = None
    unused_budget_bytes: int = 0
    budget_utilization_ratio: float = 0.0
    num_upgrades: int = 0
    num_downgrades: int = 0
    search_iterations: int = 0
    selected_candidate_origin: str = "unknown"
    tolerance_reached: bool = False
    positive_gain_moves_remaining: bool = False


class LagrangianProfileOptimizer:
    """Stable facade for Lagrangian relaxation search and assignment repair."""

    def __init__(
        self,
        profiles: List[RepresentationProfile],
        budget_bytes: int,
        max_iter: int,
        tolerance: float,
        lambda_low: float = 0.0,
        lambda_high: float = 1.0,
        logger=None,
        *,
        candidate_policy: Optional[CandidatePolicy] = None,
        assignment_repair: Optional[AssignmentRepairStrategy] = None,
        relaxation_search: Optional[RelaxationSearchStrategy] = None,
    ):
        self.profiles = profiles
        self.budget_bytes = int(budget_bytes)
        self.profile_names = [profile.name for profile in profiles]
        self.cost_lookup = {profile.name: profile.cost_bytes for profile in profiles}
        self.logger = logger or logging.getLogger("matryoshka_exp")
        self.candidate_policy = candidate_policy or UtilityFirstCandidatePolicy()
        self.assignment_repair = assignment_repair or GreedyAssignmentRepair()
        self.relaxation_search = relaxation_search or LagrangianRelaxationSearch(
            max_iter=max_iter,
            tolerance=tolerance,
            lambda_low=lambda_low,
            lambda_high=lambda_high,
            logger=self.logger,
            candidate_policy=self.candidate_policy,
        )

    def solve(self, utility_table: pd.DataFrame) -> OptimizationResult:
        problem = build_assignment_problem(
            utility_table,
            self.profiles,
            self.budget_bytes,
        )
        cheapest = self._build_cheapest_candidate(problem)
        relaxation_result = self.relaxation_search.search(
            problem,
            initial_feasible=cheapest,
        )

        final_candidates = [
            self.assignment_repair.upgrade_feasible(
                problem,
                relaxation_result.best_feasible,
            ),
            self.assignment_repair.upgrade_feasible(problem, cheapest),
        ]
        if relaxation_result.closest_infeasible is not None:
            repaired = self.assignment_repair.downgrade_to_feasible(
                problem,
                relaxation_result.closest_infeasible,
            )
            final_candidates.append(
                self.assignment_repair.upgrade_feasible(problem, repaired)
            )

        selected = self._select_best_feasible(problem, final_candidates)
        utilization = selected.total_cost_bytes / max(problem.budget_bytes, 1)
        if utilization < 0.99 and selected.positive_gain_moves_remaining:
            self.logger.warning(
                "Optimization left %.2f%% of the budget unused while positive-utility upgrades still fit.",
                100.0 * (1.0 - utilization),
            )
        self.logger.info(
            "Lagrangian optimization complete. origin=%s cost=%d utility=%.8g utilization=%.6f upgrades=%d downgrades=%d",
            selected.origin,
            selected.total_cost_bytes,
            selected.total_utility,
            utilization,
            selected.num_upgrades,
            selected.num_downgrades,
        )
        return OptimizationResult(
            assignments=problem.to_assignments(selected.chosen_idx),
            lambda_star=float(selected.lambda_value),
            total_cost_bytes=selected.total_cost_bytes,
            total_utility=selected.total_utility,
            feasible=True,
            pre_repair_relaxed_cost_bytes=selected.pre_repair_relaxed_cost_bytes,
            pre_repair_relaxed_utility=selected.pre_repair_relaxed_utility,
            unused_budget_bytes=problem.budget_bytes - selected.total_cost_bytes,
            budget_utilization_ratio=float(utilization),
            num_upgrades=selected.num_upgrades,
            num_downgrades=selected.num_downgrades,
            search_iterations=relaxation_result.search_iterations,
            selected_candidate_origin=selected.origin,
            tolerance_reached=relaxation_result.tolerance_reached,
            positive_gain_moves_remaining=selected.positive_gain_moves_remaining,
        )

    def _build_cheapest_candidate(
        self,
        problem: AssignmentProblem,
    ) -> AssignmentCandidate:
        cheapest_cost = int(np.min(problem.cost_vector))
        cheapest_profiles = np.flatnonzero(problem.cost_vector == cheapest_cost)
        local_choice = np.argmax(
            problem.utility_matrix[:, cheapest_profiles],
            axis=1,
        )
        return make_candidate(
            problem,
            cheapest_profiles[local_choice],
            lambda_value=0.0,
            origin="cheapest_baseline",
        )

    def _select_best_feasible(
        self,
        problem: AssignmentProblem,
        candidates: list[AssignmentCandidate],
    ) -> AssignmentCandidate:
        selected = candidates[0]
        for candidate in candidates[1:]:
            if self.candidate_policy.better_feasible(
                candidate,
                selected,
                budget_bytes=problem.budget_bytes,
            ):
                selected = candidate
        return selected

    def choose_profile_online(
        self,
        doc_profile_utility: Dict[str, float],
        lambda_value: float,
    ) -> str:
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
            raise RuntimeError(
                "Failed to select an online profile despite non-empty profile catalog."
            )
        return best_profile

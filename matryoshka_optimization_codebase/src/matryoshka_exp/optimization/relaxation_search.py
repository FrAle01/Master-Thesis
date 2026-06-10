from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from tqdm import tqdm

from .candidates import (
    AssignmentCandidate,
    UtilityFirstCandidatePolicy,
    make_candidate,
    mark_as_relaxed,
)
from .problem import AssignmentProblem


@dataclass(frozen=True)
class RelaxationSearchResult:
    best_feasible: AssignmentCandidate
    closest_infeasible: Optional[AssignmentCandidate]
    search_iterations: int
    tolerance_reached: bool


class LagrangianRelaxationSearch:
    def __init__(
        self,
        *,
        max_iter: int,
        tolerance: float,
        lambda_low: float,
        lambda_high: float,
        logger,
        candidate_policy=None,
    ):
        if max_iter < 0:
            raise ValueError("max_iter must be non-negative.")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative.")
        if lambda_low < 0 or lambda_high < lambda_low:
            raise ValueError("Invalid lambda bounds: require 0 <= lambda_low <= lambda_high.")
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.lambda_low = float(lambda_low)
        self.lambda_high = float(lambda_high)
        self.logger = logger
        self.candidate_policy = candidate_policy or UtilityFirstCandidatePolicy()

    def search(
        self,
        problem: AssignmentProblem,
        *,
        initial_feasible: AssignmentCandidate,
    ) -> RelaxationSearchResult:
        best_feasible = initial_feasible
        closest_infeasible = None
        lambda_low = self.lambda_low
        lambda_high = max(self.lambda_high, self._initial_lambda_high(problem))

        high_candidate = self.candidate_for_lambda(
            problem,
            lambda_high,
            origin="relaxation_lambda_high",
        )
        expansion_count = 0
        while high_candidate.total_cost_bytes > problem.budget_bytes and expansion_count < 64:
            lambda_high = max(lambda_high * 2.0, lambda_high + 1e-4)
            high_candidate = self.candidate_for_lambda(
                problem,
                lambda_high,
                origin="relaxation_lambda_high",
            )
            expansion_count += 1

        for candidate in (
            self.candidate_for_lambda(
                problem,
                lambda_low,
                origin="relaxation_lambda_low",
            ),
            high_candidate,
        ):
            best_feasible, closest_infeasible = self._record_candidate(
                problem,
                candidate,
                best_feasible,
                closest_infeasible,
            )

        self.logger.info(
            "Starting Lagrangian optimization with lambda_low=%.8g lambda_high=%.8g budget_bytes=%d",
            lambda_low,
            lambda_high,
            problem.budget_bytes,
        )

        tolerance_reached = False
        search_iterations = 0
        for iteration in tqdm(range(self.max_iter), desc="Lagrangian optimization"):
            search_iterations = iteration + 1
            lam = 0.5 * (lambda_low + lambda_high)
            candidate = self.candidate_for_lambda(
                problem,
                lam,
                origin="relaxation_search",
            )
            best_feasible, closest_infeasible = self._record_candidate(
                problem,
                candidate,
                best_feasible,
                closest_infeasible,
            )
            relative_gap = abs(candidate.total_cost_bytes - problem.budget_bytes) / max(
                problem.budget_bytes,
                1,
            )
            self.logger.info(
                "Iteration %d: lambda=%.8g cost=%d utility=%.8g relative_gap=%.8g feasible=%s",
                iteration,
                lam,
                candidate.total_cost_bytes,
                candidate.total_utility,
                relative_gap,
                candidate.total_cost_bytes <= problem.budget_bytes,
            )
            if relative_gap <= self.tolerance and candidate.total_cost_bytes <= problem.budget_bytes:
                tolerance_reached = True
                break
            if candidate.total_cost_bytes > problem.budget_bytes:
                lambda_low = lam
            else:
                lambda_high = lam

        return RelaxationSearchResult(
            best_feasible=best_feasible,
            closest_infeasible=closest_infeasible,
            search_iterations=search_iterations,
            tolerance_reached=tolerance_reached,
        )

    @staticmethod
    def candidate_for_lambda(
        problem: AssignmentProblem,
        lambda_value: float,
        *,
        origin: str,
    ) -> AssignmentCandidate:
        reduced = problem.utility_matrix - float(lambda_value) * problem.cost_vector[None, :]
        chosen_idx = np.argmax(reduced, axis=1)
        return mark_as_relaxed(
            make_candidate(
                problem,
                chosen_idx,
                lambda_value=lambda_value,
                origin=origin,
            )
        )

    def _record_candidate(
        self,
        problem,
        candidate,
        best_feasible,
        closest_infeasible,
    ):
        if self.candidate_policy.better_feasible(
            candidate,
            best_feasible,
            budget_bytes=problem.budget_bytes,
        ):
            best_feasible = candidate
        elif self.candidate_policy.better_infeasible(
            candidate,
            closest_infeasible,
            budget_bytes=problem.budget_bytes,
        ):
            closest_infeasible = candidate
        return best_feasible, closest_infeasible

    @staticmethod
    def _initial_lambda_high(problem: AssignmentProblem) -> float:
        utility_span = float(
            np.max(problem.utility_matrix) - np.min(problem.utility_matrix)
        )
        unique_costs = np.unique(problem.cost_vector)
        if len(unique_costs) <= 1:
            return 1e-4
        min_cost_gap = int(np.min(np.diff(unique_costs)))
        return max(1e-4, utility_span / max(min_cost_gap, 1))

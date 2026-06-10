from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .problem import AssignmentProblem


@dataclass(frozen=True)
class AssignmentCandidate:
    chosen_idx: np.ndarray
    total_cost_bytes: int
    total_utility: float
    lambda_value: float
    origin: str
    num_upgrades: int = 0
    num_downgrades: int = 0
    pre_repair_relaxed_cost_bytes: Optional[int] = None
    pre_repair_relaxed_utility: Optional[float] = None
    positive_gain_moves_remaining: bool = False


def make_candidate(
    problem: AssignmentProblem,
    chosen_idx: np.ndarray,
    *,
    lambda_value: float,
    origin: str,
    num_upgrades: int = 0,
    num_downgrades: int = 0,
    pre_repair_relaxed_cost_bytes: Optional[int] = None,
    pre_repair_relaxed_utility: Optional[float] = None,
    positive_gain_moves_remaining: bool = False,
) -> AssignmentCandidate:
    chosen_idx = np.asarray(chosen_idx, dtype=np.int64)
    total_cost, total_utility = problem.evaluate(chosen_idx)
    return AssignmentCandidate(
        chosen_idx=chosen_idx.copy(),
        total_cost_bytes=total_cost,
        total_utility=total_utility,
        lambda_value=float(lambda_value),
        origin=origin,
        num_upgrades=int(num_upgrades),
        num_downgrades=int(num_downgrades),
        pre_repair_relaxed_cost_bytes=pre_repair_relaxed_cost_bytes,
        pre_repair_relaxed_utility=pre_repair_relaxed_utility,
        positive_gain_moves_remaining=positive_gain_moves_remaining,
    )


def mark_as_relaxed(candidate: AssignmentCandidate) -> AssignmentCandidate:
    return replace(
        candidate,
        pre_repair_relaxed_cost_bytes=candidate.total_cost_bytes,
        pre_repair_relaxed_utility=candidate.total_utility,
    )


class UtilityFirstCandidatePolicy:
    @staticmethod
    def better_feasible(
        candidate: AssignmentCandidate,
        incumbent: Optional[AssignmentCandidate],
        *,
        budget_bytes: int,
    ) -> bool:
        if candidate.total_cost_bytes > budget_bytes:
            return False
        if incumbent is None:
            return True
        if candidate.total_utility != incumbent.total_utility:
            return candidate.total_utility > incumbent.total_utility
        if candidate.total_cost_bytes != incumbent.total_cost_bytes:
            return candidate.total_cost_bytes > incumbent.total_cost_bytes
        return UtilityFirstCandidatePolicy._wins_deterministic_tie(candidate, incumbent)

    @staticmethod
    def better_infeasible(
        candidate: AssignmentCandidate,
        incumbent: Optional[AssignmentCandidate],
        *,
        budget_bytes: int,
    ) -> bool:
        if candidate.total_cost_bytes <= budget_bytes:
            return False
        if incumbent is None:
            return True
        candidate_overflow = candidate.total_cost_bytes - budget_bytes
        incumbent_overflow = incumbent.total_cost_bytes - budget_bytes
        if candidate_overflow != incumbent_overflow:
            return candidate_overflow < incumbent_overflow
        if candidate.total_utility != incumbent.total_utility:
            return candidate.total_utility > incumbent.total_utility
        return UtilityFirstCandidatePolicy._wins_deterministic_tie(candidate, incumbent)

    @staticmethod
    def _wins_deterministic_tie(
        candidate: AssignmentCandidate,
        incumbent: AssignmentCandidate,
    ) -> bool:
        differences = np.flatnonzero(candidate.chosen_idx != incumbent.chosen_idx)
        if differences.size == 0:
            return False
        first_difference = int(differences[0])
        return int(candidate.chosen_idx[first_difference]) > int(
            incumbent.chosen_idx[first_difference]
        )

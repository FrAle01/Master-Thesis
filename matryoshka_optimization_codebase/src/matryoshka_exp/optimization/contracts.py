from __future__ import annotations

from typing import Optional, Protocol

from .candidates import AssignmentCandidate
from .relaxation_search import RelaxationSearchResult
from .problem import AssignmentProblem


class CandidatePolicy(Protocol):
    def better_feasible(
        self,
        candidate: AssignmentCandidate,
        incumbent: Optional[AssignmentCandidate],
        *,
        budget_bytes: int,
    ) -> bool: ...

    def better_infeasible(
        self,
        candidate: AssignmentCandidate,
        incumbent: Optional[AssignmentCandidate],
        *,
        budget_bytes: int,
    ) -> bool: ...


class AssignmentRepairStrategy(Protocol):
    def upgrade_feasible(
        self,
        problem: AssignmentProblem,
        candidate: AssignmentCandidate,
    ) -> AssignmentCandidate: ...

    def downgrade_to_feasible(
        self,
        problem: AssignmentProblem,
        candidate: AssignmentCandidate,
    ) -> AssignmentCandidate: ...


class RelaxationSearchStrategy(Protocol):
    def search(
        self,
        problem: AssignmentProblem,
        *,
        initial_feasible: AssignmentCandidate,
    ) -> RelaxationSearchResult: ...

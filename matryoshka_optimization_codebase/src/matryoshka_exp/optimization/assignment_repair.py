from __future__ import annotations

import numpy as np

from .candidates import AssignmentCandidate, make_candidate
from .problem import AssignmentProblem


class GreedyAssignmentRepair:
    def upgrade_feasible(
        self,
        problem: AssignmentProblem,
        candidate: AssignmentCandidate,
    ) -> AssignmentCandidate:
        chosen_idx = candidate.chosen_idx.copy()
        current_cost = candidate.total_cost_bytes
        num_upgrades = candidate.num_upgrades

        for _ in range(max(0, problem.num_profiles - 1)):
            moves = self._best_upgrade_moves(problem, chosen_idx)
            if moves is None:
                break
            doc_indices, target_indices, extra_costs = moves
            applied = 0
            for doc_idx, target_idx, extra_cost in zip(
                doc_indices,
                target_indices,
                extra_costs,
            ):
                if current_cost + extra_cost > problem.budget_bytes:
                    continue
                chosen_idx[doc_idx] = target_idx
                current_cost += int(extra_cost)
                num_upgrades += 1
                applied += 1
            if applied == 0:
                break

        remaining_moves = self._best_upgrade_moves(problem, chosen_idx)
        remaining_budget = problem.budget_bytes - current_cost
        positive_gain_moves_remaining = (
            remaining_moves is not None
            and bool(np.any(remaining_moves[2] <= remaining_budget))
        )
        origin = (
            f"{candidate.origin}_upgraded"
            if num_upgrades > candidate.num_upgrades
            else candidate.origin
        )
        return make_candidate(
            problem,
            chosen_idx,
            lambda_value=candidate.lambda_value,
            origin=origin,
            num_upgrades=num_upgrades,
            num_downgrades=candidate.num_downgrades,
            pre_repair_relaxed_cost_bytes=candidate.pre_repair_relaxed_cost_bytes,
            pre_repair_relaxed_utility=candidate.pre_repair_relaxed_utility,
            positive_gain_moves_remaining=positive_gain_moves_remaining,
        )

    def downgrade_to_feasible(
        self,
        problem: AssignmentProblem,
        candidate: AssignmentCandidate,
    ) -> AssignmentCandidate:
        chosen_idx = candidate.chosen_idx.copy()
        current_cost = candidate.total_cost_bytes
        num_downgrades = candidate.num_downgrades

        for _ in range(max(1, problem.num_profiles - 1)):
            if current_cost <= problem.budget_bytes:
                break
            moves = self._best_downgrade_moves(problem, chosen_idx)
            if moves is None:
                break
            doc_indices, target_indices, bytes_saved_values = moves
            applied = 0
            for doc_idx, target_idx, bytes_saved in zip(
                doc_indices,
                target_indices,
                bytes_saved_values,
            ):
                if current_cost <= problem.budget_bytes:
                    break
                chosen_idx[doc_idx] = target_idx
                current_cost -= int(bytes_saved)
                num_downgrades += 1
                applied += 1
            if applied == 0:
                break

        if current_cost > problem.budget_bytes:
            raise RuntimeError(
                "Unable to repair an infeasible assignment despite a feasible cheapest baseline."
            )
        return make_candidate(
            problem,
            chosen_idx,
            lambda_value=candidate.lambda_value,
            origin="repaired_infeasible_relaxed",
            num_upgrades=candidate.num_upgrades,
            num_downgrades=num_downgrades,
            pre_repair_relaxed_cost_bytes=candidate.total_cost_bytes,
            pre_repair_relaxed_utility=candidate.total_utility,
        )

    @staticmethod
    def _best_upgrade_moves(problem: AssignmentProblem, chosen_idx):
        rows = np.arange(problem.num_docs)
        current_costs = problem.cost_vector[chosen_idx]
        current_utilities = problem.utility_matrix[rows, chosen_idx]
        best_ratio = np.full(problem.num_docs, -np.inf, dtype=np.float64)
        best_gain = np.full(problem.num_docs, -np.inf, dtype=np.float64)
        best_extra_cost = np.zeros(problem.num_docs, dtype=np.int64)
        best_target = np.full(problem.num_docs, -1, dtype=np.int64)

        for target_idx, target_cost in enumerate(problem.cost_vector):
            extra_cost = target_cost - current_costs
            utility_gain = problem.utility_matrix[:, target_idx] - current_utilities
            valid = (extra_cost > 0) & (utility_gain > 0.0)
            ratio = np.divide(
                utility_gain,
                extra_cost,
                out=np.full(problem.num_docs, -np.inf, dtype=np.float64),
                where=valid,
            )
            better = valid & (
                (ratio > best_ratio)
                | ((ratio == best_ratio) & (utility_gain > best_gain))
                | (
                    (ratio == best_ratio)
                    & (utility_gain == best_gain)
                    & ((best_target < 0) | (target_idx < best_target))
                )
            )
            best_ratio[better] = ratio[better]
            best_gain[better] = utility_gain[better]
            best_extra_cost[better] = extra_cost[better]
            best_target[better] = target_idx

        valid_docs = np.flatnonzero(best_target >= 0)
        if valid_docs.size == 0:
            return None
        order = np.lexsort(
            (
                best_target[valid_docs],
                valid_docs,
                -best_gain[valid_docs],
                -best_ratio[valid_docs],
            )
        )
        selected_docs = valid_docs[order]
        return (
            selected_docs,
            best_target[selected_docs],
            best_extra_cost[selected_docs],
        )

    @staticmethod
    def _best_downgrade_moves(problem: AssignmentProblem, chosen_idx):
        rows = np.arange(problem.num_docs)
        current_costs = problem.cost_vector[chosen_idx]
        current_utilities = problem.utility_matrix[rows, chosen_idx]
        best_ratio = np.full(problem.num_docs, np.inf, dtype=np.float64)
        best_loss = np.full(problem.num_docs, np.inf, dtype=np.float64)
        best_bytes_saved = np.zeros(problem.num_docs, dtype=np.int64)
        best_target = np.full(problem.num_docs, -1, dtype=np.int64)

        for target_idx, target_cost in enumerate(problem.cost_vector):
            bytes_saved = current_costs - target_cost
            utility_loss = current_utilities - problem.utility_matrix[:, target_idx]
            valid = bytes_saved > 0
            ratio = np.divide(
                utility_loss,
                bytes_saved,
                out=np.full(problem.num_docs, np.inf, dtype=np.float64),
                where=valid,
            )
            better = valid & (
                (ratio < best_ratio)
                | ((ratio == best_ratio) & (utility_loss < best_loss))
                | (
                    (ratio == best_ratio)
                    & (utility_loss == best_loss)
                    & (bytes_saved > best_bytes_saved)
                )
                | (
                    (ratio == best_ratio)
                    & (utility_loss == best_loss)
                    & (bytes_saved == best_bytes_saved)
                    & ((best_target < 0) | (target_idx < best_target))
                )
            )
            best_ratio[better] = ratio[better]
            best_loss[better] = utility_loss[better]
            best_bytes_saved[better] = bytes_saved[better]
            best_target[better] = target_idx

        valid_docs = np.flatnonzero(best_target >= 0)
        if valid_docs.size == 0:
            return None
        order = np.lexsort(
            (
                best_target[valid_docs],
                valid_docs,
                -best_bytes_saved[valid_docs],
                best_loss[valid_docs],
                best_ratio[valid_docs],
            )
        )
        selected_docs = valid_docs[order]
        return (
            selected_docs,
            best_target[selected_docs],
            best_bytes_saved[selected_docs],
        )

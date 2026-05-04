from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class RepairResult:
    assignments: pd.DataFrame
    total_cost_bytes: int
    total_utility: float
    feasible: bool


class DeterministicBudgetRepair:
    """Future hook to make assignments budget-feasible via deterministic downgrades."""

    def repair(
        self,
        assignments: pd.DataFrame,
        utility_table: pd.DataFrame,
        cost_lookup: Dict[str, int],
        budget_bytes: int,
    ) -> RepairResult:
        required_assignment = {"docno", "profile", "utility", "cost_bytes"}
        required_utility = {"docno", "profile", "utility"}
        if required_assignment.difference(assignments.columns):
            raise ValueError("assignments missing required columns")
        if required_utility.difference(utility_table.columns):
            raise ValueError("utility_table missing required columns")

        repaired = assignments.copy()
        current_cost = int(repaired["cost_bytes"].sum())

        if current_cost <= budget_bytes:
            return RepairResult(
                assignments=repaired,
                total_cost_bytes=current_cost,
                total_utility=float(repaired["utility"].sum()),
                feasible=True,
            )

        util_lookup = utility_table.pivot(index="docno", columns="profile", values="utility")

        while current_cost > budget_bytes:
            candidate_moves = []
            for idx, row in repaired.sort_values(["docno", "profile"], kind="mergesort").iterrows():
                docno = row["docno"]
                cur_cost = int(row["cost_bytes"])
                cur_util = float(row["utility"])

                for new_profile, new_cost in sorted(cost_lookup.items(), key=lambda x: (x[1], x[0])):
                    if new_cost >= cur_cost:
                        continue
                    new_util = float(util_lookup.loc[docno, new_profile])
                    util_loss = cur_util - new_util
                    byte_gain = cur_cost - int(new_cost)
                    if byte_gain <= 0:
                        continue
                    loss_per_byte = util_loss / byte_gain
                    candidate_moves.append((loss_per_byte, str(docno), str(new_profile), idx, new_cost, new_util, byte_gain))

            if not candidate_moves:
                break

            candidate_moves.sort(key=lambda x: (x[0], x[1], x[2]))
            _, _, new_profile, idx, new_cost, new_util, byte_gain = candidate_moves[0]
            repaired.at[idx, "profile"] = new_profile
            repaired.at[idx, "cost_bytes"] = int(new_cost)
            repaired.at[idx, "utility"] = float(new_util)
            current_cost -= int(byte_gain)

        return RepairResult(
            assignments=repaired,
            total_cost_bytes=int(repaired["cost_bytes"].sum()),
            total_utility=float(repaired["utility"].sum()),
            feasible=int(repaired["cost_bytes"].sum()) <= budget_bytes,
        )

from __future__ import annotations


class InfeasibleOptimizationError(RuntimeError):
    """Raised when profile assignment exceeds the configured memory budget."""

    def __init__(self, *, budget_bytes: int, assigned_cost_bytes: int):
        overflow = assigned_cost_bytes - budget_bytes
        relative_overflow = (overflow / budget_bytes) if budget_bytes > 0 else float("inf")
        message = (
            "Optimization result is infeasible: assigned_cost_bytes="
            f"{assigned_cost_bytes}, budget_bytes={budget_bytes}, overflow_bytes={overflow}, "
            f"relative_overflow={relative_overflow:.6f}"
        )
        super().__init__(message)
        self.budget_bytes = budget_bytes
        self.assigned_cost_bytes = assigned_cost_bytes
        self.overflow_bytes = overflow
        self.relative_overflow = relative_overflow

from __future__ import annotations

import warnings

warnings.warn(
    "`matryoshka_exp.optimization.repair` is deprecated and kept as a compatibility shim. "
    "Use `matryoshka_exp.legacy.repair` for explicit legacy imports.",
    DeprecationWarning,
    stacklevel=2,
)

from ..legacy.repair import DeterministicBudgetRepair, RepairResult

__all__ = ["DeterministicBudgetRepair", "RepairResult"]

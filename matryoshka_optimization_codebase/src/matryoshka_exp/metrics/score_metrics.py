from __future__ import annotations

import warnings

warnings.warn(
    "`matryoshka_exp.metrics.score_metrics` is deprecated and kept as a compatibility shim. "
    "Use `matryoshka_exp.legacy.score_metrics` for explicit legacy imports.",
    DeprecationWarning,
    stacklevel=2,
)

from ..legacy.score_metrics import *  # noqa: F401,F403

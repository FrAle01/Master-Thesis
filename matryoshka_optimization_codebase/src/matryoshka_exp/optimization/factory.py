from __future__ import annotations

import warnings

from .lagrangian import LagrangianProfileOptimizer


def create_optimizer(config, profiles, logger):
    algorithm = str(config.optimization.algorithm).lower()
    if algorithm == "lagrangian_dual":
        warnings.warn(
            "`optimization.algorithm=lagrangian_dual` is deprecated; "
            "use `lagrangian_relaxation`.",
            DeprecationWarning,
            stacklevel=2,
        )
        algorithm = "lagrangian_relaxation"
    if algorithm != "lagrangian_relaxation":
        raise ValueError(f"Unsupported optimization.algorithm: {config.optimization.algorithm}")
    return LagrangianProfileOptimizer(
        profiles=profiles,
        budget_bytes=config.budget_bytes_resolved(),
        max_iter=config.optimization.max_iter,
        tolerance=config.optimization.tolerance,
        lambda_low=config.optimization.lambda_low,
        lambda_high=config.optimization.lambda_high,
        logger=logger,
    )

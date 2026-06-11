from __future__ import annotations

from .residual_norm import ResidualNormUtilityEstimator


def create_utility_estimator(config, logger):
    estimator = str(config.utility.estimator)
    if estimator == "query_log":
        from ..utility_estimation import QueryLogUtilityEstimator

        return QueryLogUtilityEstimator(config, logger)
    if estimator == "residual_norm":
        return ResidualNormUtilityEstimator(config, logger)
    raise ValueError(f"Unsupported utility estimator: {estimator}")

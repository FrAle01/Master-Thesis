from __future__ import annotations

from typing import Dict


def build_optimization_report(opt_result) -> Dict:
    return {
        "lambda_star": opt_result.lambda_star,
        "feasible": bool(opt_result.feasible),
        "total_utility": float(opt_result.total_utility),
        "total_cost_bytes": int(opt_result.total_cost_bytes),
        "pre_repair_relaxed_cost_bytes": opt_result.pre_repair_relaxed_cost_bytes,
        "pre_repair_relaxed_utility": opt_result.pre_repair_relaxed_utility,
        "unused_budget_bytes": int(opt_result.unused_budget_bytes),
        "budget_utilization_ratio": float(opt_result.budget_utilization_ratio),
        "num_upgrades": int(opt_result.num_upgrades),
        "num_downgrades": int(opt_result.num_downgrades),
        "search_iterations": int(opt_result.search_iterations),
        "selected_candidate_origin": opt_result.selected_candidate_origin,
        "tolerance_reached": bool(opt_result.tolerance_reached),
        "positive_gain_moves_remaining": bool(opt_result.positive_gain_moves_remaining),
    }


def build_memory_summary(config, *, assignments, docnos, full_profile) -> Dict:
    return {
        "budget_bytes": config.budget_bytes_resolved(),
        "optimized_total_cost_bytes": int(assignments["cost_bytes"].sum()),
        "full_total_cost_bytes": len(docnos) * full_profile.cost_bytes,
        "optimized_avg_cost_bytes": float(assignments["cost_bytes"].mean()),
        "full_cost_bytes_per_doc": full_profile.cost_bytes,
    }


def build_experiment_summary(
    config,
    *,
    payload,
    metrics_by_run,
    num_queries,
    num_eval_queries,
) -> Dict:
    opt_result = payload["opt_result"]
    if str(config.utility.estimator) == "query_log":
        full_metrics = metrics_by_run.get("full_embedding", {})
        optimized_metrics = metrics_by_run.get("optimized_embedding", {})
    else:
        full_metrics = metrics_by_run.get("full_run", {})
        optimized_metrics = metrics_by_run.get("optimized_run", {})

    return {
        "lambda_star": opt_result.lambda_star,
        "feasible": opt_result.feasible,
        "optimized_total_utility": opt_result.total_utility,
        "metrics_by_run": metrics_by_run,
        "full_metrics": full_metrics,
        "optimized_metrics": optimized_metrics,
        "num_docs": len(payload["docnos"]),
        "num_queries": int(num_queries),
        "num_eval_queries": int(num_eval_queries),
        "utility_estimator": str(config.utility.estimator),
    }

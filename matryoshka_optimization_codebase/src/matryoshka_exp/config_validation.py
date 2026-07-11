from __future__ import annotations

from typing import Any, Dict, Optional
import warnings


def validate_choice(name: str, value: str, allowed: set[str]) -> None:
    if value not in allowed:
        raise ValueError(f"Unsupported {name}: {value}. Allowed values: {sorted(allowed)}")


def apply_deprecated_data_mappings(
    cfg,
    *,
    raw_data: Optional[Dict[str, Any]],
    raw_retrieval: Optional[Dict[str, Any]],
) -> None:
    raw_data = raw_data or {}
    raw_retrieval = raw_retrieval or {}

    if "candidates_per_query" in raw_data:
        if "candidate_k" not in raw_retrieval:
            cfg.retrieval.candidate_k = int(cfg.data.candidates_per_query)
            warnings.warn(
                "`data.candidates_per_query` is deprecated; mapped to `retrieval.candidate_k`.",
                stacklevel=2,
            )
        elif int(raw_retrieval["candidate_k"]) != int(cfg.data.candidates_per_query):
            warnings.warn(
                "`data.candidates_per_query` is deprecated and ignored because `retrieval.candidate_k` is set.",
                stacklevel=2,
            )

    if "candidate_source" in raw_data:
        candidate_source = str(cfg.data.candidate_source).strip().lower()
        mapped_mode = {
            "pyterrier_bm25": "pyterrier_candidates",
            "dense_exact": "dense_exact",
        }.get(candidate_source)
        if mapped_mode is None:
            raise ValueError(
                f"Unsupported deprecated `data.candidate_source`: {cfg.data.candidate_source}. "
                "Supported values are: ['dense_exact', 'pyterrier_bm25']."
            )
        if "mode" not in raw_retrieval:
            cfg.retrieval.mode = mapped_mode
            warnings.warn(
                "`data.candidate_source` is deprecated; mapped to `retrieval.mode`.",
                stacklevel=2,
            )
        elif str(raw_retrieval["mode"]) != mapped_mode:
            warnings.warn(
                "`data.candidate_source` is deprecated and ignored because `retrieval.mode` is set.",
                stacklevel=2,
            )


def validate_config(cfg) -> None:
    if not cfg.profiles:
        raise ValueError("Config must define at least one profile in `profiles`.")

    profile_names = {item.name for item in cfg.profiles}
    if cfg.model.full_profile_name not in profile_names:
        raise ValueError(
            f"`model.full_profile_name` ({cfg.model.full_profile_name}) is not present in `profiles`."
        )
    if cfg.utility.default_utility_profile_name is not None and cfg.utility.default_utility_profile_name not in profile_names:
        raise ValueError(
            f"`utility.default_utility_profile_name` ({cfg.utility.default_utility_profile_name}) is not present in `profiles`."
        )

    if cfg.utility.default_utility_profile_name is None:
        cfg.utility.default_utility_profile_name = cfg.model.full_profile_name

    full_profile_dimension = next(
        profile.dimension for profile in cfg.profiles if profile.name == cfg.model.full_profile_name
    )
    for profile in cfg.profiles:
        if profile.dimension <= 0:
            raise ValueError(f"Profile `{profile.name}` has non-positive dimension: {profile.dimension}.")
        if profile.dimension > full_profile_dimension:
            raise ValueError(
                f"Profile `{profile.name}` dimension {profile.dimension} exceeds full profile dimension "
                f"{full_profile_dimension}."
            )
        if profile.cost_is_explicit and (profile.cost_bytes is None or profile.cost_bytes <= 0):
            raise ValueError(f"Profile `{profile.name}` has `cost_is_explicit=true` but invalid `cost_bytes`: {profile.cost_bytes}.")

    validate_choice("model.backend", cfg.model.backend, {"sentence_transformers", "transformers"})
    validate_choice("model.similarity", cfg.model.similarity, {"dot", "cosine"})
    validate_choice("optimization.mode", cfg.optimization.mode, {"batch", "streaming"})
    if cfg.optimization.mode == "streaming":
        raise ValueError(
            "`optimization.mode=streaming` is currently disabled. "
            "Use `optimization.mode=batch`."
        )
    validate_choice("execution.retrieval_device", cfg.execution.retrieval_device, {"cuda", "cpu"})
    validate_choice("execution.embedding_checkpoint_resume", cfg.execution.embedding_checkpoint_resume, {"auto", "restart", "fail"})
    validate_choice("retrieval.mode", cfg.retrieval.mode, {"dense_exact", "pyterrier_candidates"})
    validate_choice("utility.estimator", cfg.utility.estimator, {"query_log", "residual_norm"})
    validate_choice(
        "utility.metric",
        cfg.utility.metric,
        {
            "relative_score_dissimilarity",
            "absolute_score_utility",
            "squared_score_utility",
            "relative_margin_utility",
            "hybrid_score_margin_utility",
        },
    )
    validate_choice("utility.relevance.mode", cfg.utility.relevance.mode, {"weak", "model", "hybrid"})
    validate_choice(
        "utility.relevance.calibration",
        cfg.utility.relevance.calibration,
        {"minmax_score", "rank_log_discount", "constant_one"},
    )
    validate_choice(
        "utility.relevance.weak_source",
        cfg.utility.relevance.weak_source,
        {"bm25", "dense", "hybrid_rerank"},
    )
    validate_choice("utility.relevance.uncertainty", cfg.utility.relevance.uncertainty, {"margin"})
    validate_choice("utility.relevance.qrel_adjustment", cfg.utility.relevance.qrel_adjustment, {"hard_override", "relevant_only_to_one"})
    validate_choice("utility.relevance.scale.mode", cfg.utility.relevance.scale.mode, {"minmax", "explicit_map"})
    validate_choice("utility.tail.mode", cfg.utility.tail.mode, {"static_default", "rbp_residual_interval"})
    validate_choice("utility.tail.profile_quality", cfg.utility.tail.profile_quality, {"cosine_padding", "sampled_dense_score_preservation"})
    validate_choice("data.single_log_policy", cfg.data.single_log_policy, {"shared", "split_by_qrels"})
    validate_choice("training.finetune_strategy", cfg.training.finetune_strategy.lower(), {"full", "lora"})
    validate_choice(
        "data.training_format",
        cfg.data.training_format,
        {"hf_triplet", "hf_tevatron_passage", "jsonl_triplet"},
    )
    validate_choice(
        "data.full_embeddings_source",
        cfg.data.full_embeddings_source,
        {"auto", "hf_dataset", "compute"},
    )

    if cfg.retrieval.candidate_k <= 0:
        raise ValueError("`retrieval.candidate_k` must be > 0.")
    if cfg.retrieval.top_k <= 0:
        raise ValueError("`retrieval.top_k` must be > 0.")
    if cfg.utility.margin_negatives <= 0:
        raise ValueError("`utility.margin_negatives` must be > 0.")
    if int(cfg.utility.residual_norm.norm) not in {1, 2}:
        raise ValueError("`utility.residual_norm.norm` must be either 1 or 2.")
    if float(cfg.utility.residual_norm.epsilon) <= 0.0:
        raise ValueError("`utility.residual_norm.epsilon` must be > 0.")
    if cfg.utility.sample_pairs_per_query == 0 or cfg.utility.sample_pairs_per_query < -1:
        raise ValueError("`utility.sample_pairs_per_query` must be -1 (unlimited) or > 0.")
    if cfg.execution.embedding_checkpoint_every_docs <= 0:
        raise ValueError("`execution.embedding_checkpoint_every_docs` must be > 0.")
    if float(cfg.execution.retrieval_vram_temp_overhead_factor) < 1.0:
        raise ValueError("`execution.retrieval_vram_temp_overhead_factor` must be >= 1.0.")
    if cfg.utility.relevance.top_k_candidates <= 0:
        raise ValueError("`utility.relevance.top_k_candidates` must be > 0.")
    if cfg.utility.relevance.rerank_k <= 0:
        raise ValueError("`utility.relevance.rerank_k` must be > 0.")
    if not (0.0 < float(cfg.utility.relevance.high_impact_fraction) <= 1.0):
        raise ValueError("`utility.relevance.high_impact_fraction` must be in (0, 1].")
    if cfg.utility.relevance.validation_qids_limit <= 0:
        raise ValueError("`utility.relevance.validation_qids_limit` must be > 0.")
    if cfg.utility.relevance.scale.max_label <= cfg.utility.relevance.scale.min_label:
        raise ValueError("`utility.relevance.scale.max_label` must be greater than `utility.relevance.scale.min_label`.")
    if cfg.utility.relevance.scale.mode == "explicit_map" and not cfg.utility.relevance.scale.explicit_map:
        raise ValueError("`utility.relevance.scale.explicit_map` must be provided when scale.mode is `explicit_map`.")
    if not (0.0 <= float(cfg.utility.tail.target_residual) <= 1.0):
        raise ValueError("`utility.tail.target_residual` must be in [0, 1].")
    if not (0.0 <= float(cfg.utility.tail.expected_fraction) <= 1.0):
        raise ValueError("`utility.tail.expected_fraction` must be in [0, 1].")
    if cfg.utility.tail.conservatism_beta is not None and float(cfg.utility.tail.conservatism_beta) < 0.0:
        raise ValueError("`utility.tail.conservatism_beta` must be null or >= 0.")
    if cfg.utility.tail.sampled_quality_queries <= 0:
        raise ValueError("`utility.tail.sampled_quality_queries` must be > 0.")
    if cfg.utility.tail.max_tail_relevance < 0.0:
        raise ValueError("`utility.tail.max_tail_relevance` must be >= 0.")
    validate_choice("utility.aggregate", cfg.utility.aggregate, {"mean"})

    if cfg.utility.estimator == "query_log":
        if cfg.data.pyterrier_dataset is None:
            missing = []
            if not cfg.data.local_topics_path:
                missing.append("data.local_topics_path")
            if not cfg.data.local_qrels_path:
                missing.append("data.local_qrels_path")
            if not cfg.data.local_corpus_path:
                missing.append("data.local_corpus_path")
            if missing:
                raise ValueError(
                    "When `data.pyterrier_dataset` is not set, local inputs are required. Missing: "
                    + ", ".join(missing)
                )
    else:
        evaluation_dataset = cfg.data.eval_pyterrier_dataset or cfg.data.pyterrier_dataset
        if evaluation_dataset is None:
            missing = []
            if not (cfg.data.local_eval_topics_path or cfg.data.local_topics_path):
                missing.append("data.local_eval_topics_path")
            if not (cfg.data.local_eval_qrels_path or cfg.data.local_qrels_path):
                missing.append("data.local_eval_qrels_path")
            if not (cfg.data.local_eval_corpus_path or cfg.data.local_corpus_path):
                missing.append("data.local_eval_corpus_path")
            if missing:
                raise ValueError(
                    "Residual-norm utility requires an evaluation dataset or complete local evaluation inputs. "
                    "Missing: " + ", ".join(missing)
                )

    if cfg.training.enabled:
        if cfg.data.training_format == "jsonl_triplet" and not cfg.data.training_local_path:
            raise ValueError("`data.training_local_path` is required when `data.training_format` is `jsonl_triplet`.")
        if cfg.data.training_format in {"hf_triplet", "hf_tevatron_passage"} and not cfg.data.training_dataset_name:
            raise ValueError(
                "`data.training_dataset_name` is required for Hugging Face training formats (`hf_triplet`, "
                "`hf_tevatron_passage`)."
            )
        if cfg.training.hub.enabled:
            if not cfg.training.hub.repo_prefix:
                raise ValueError("`training.hub.repo_prefix` is required when `training.hub.enabled=true`.")
            if not cfg.training.hub.auto_repo_from_experiment:
                raise ValueError("Only `training.hub.auto_repo_from_experiment=true` is currently supported.")

    if cfg.data.full_embeddings_source in {"auto", "hf_dataset"} and not cfg.data.hf_embeddings_repo_id:
        raise ValueError(
            "`data.hf_embeddings_repo_id` is required when `data.full_embeddings_source` is `auto` or `hf_dataset`."
        )

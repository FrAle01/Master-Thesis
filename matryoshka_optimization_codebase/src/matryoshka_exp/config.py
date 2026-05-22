from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings

import yaml


@dataclass
class ProfileConfig:
    name: str
    dimension: int
    layer: Optional[int] = None
    normalize: bool = True
    cost_bytes: Optional[int] = None
    cost_is_explicit: Optional[bool] = False


@dataclass
class DataConfig:
    pyterrier_dataset: Optional[str] = None
    dataset_provider: Optional[str] = None
    topics_variant: Optional[str] = None
    qrels_variant: Optional[str] = None
    eval_pyterrier_dataset: Optional[str] = None
    eval_dataset_provider: Optional[str] = None
    eval_topics_variant: Optional[str] = None
    eval_qrels_variant: Optional[str] = None
    text_fields: List[str] = field(default_factory=lambda: ["text"])
    max_docs: Optional[int] = None
    max_queries: Optional[int] = None
    topic_column: str = "query"
    docno_column: str = "docno"
    local_corpus_path: Optional[str] = None
    local_topics_path: Optional[str] = None
    local_qrels_path: Optional[str] = None
    local_eval_topics_path: Optional[str] = None
    local_eval_qrels_path: Optional[str] = None
    single_log_policy: str = "shared"
    qrels_label_column: str = "label"
    training_dataset_name: Optional[str] = None
    training_local_path: Optional[str] = None
    training_split: str = "train"
    training_format: str = "hf_triplet"
    query_column: str = "query"
    positive_column: str = "positive"
    negative_column: Optional[str] = "negative"
    tevatron_positive_passages_column: str = "positive_passages"
    tevatron_negative_passages_column: str = "negative_passages"
    tevatron_passage_text_field: str = "text"
    candidate_source: str = "pyterrier_bm25"
    candidates_per_query: int = 200
    local_terrier_index_path: Optional[str] = None
    build_local_terrier_index_if_missing: bool = False
    terrier_index_threads: int = 4
    terrier_index_overwrite: bool = False
    terrier_meta_lengths: Dict[str, int] = field(default_factory=lambda: {"docno": 64, "text": 4096})
    full_embeddings_source: str = "auto"
    hf_embeddings_repo_id: Optional[str] = None
    hf_embeddings_split: str = "train"
    hf_embeddings_docno_column: str = "docno"
    hf_embeddings_vector_column: str = "full_embedding"


@dataclass
class ModelConfig:
    mode: str = "pretrained"
    backend: str = "sentence_transformers"
    model_name_or_path: str = "nomic-ai/modernbert-embed-base"
    tokenizer_name_or_path: Optional[str] = None
    adapter_type: Optional[str] = None
    adapter_path: Optional[str] = None
    adapter_name: str = "default"
    trust_remote_code: bool = False
    query_prompt: str = ""
    document_prompt: str = ""
    similarity: str = "dot"
    full_profile_name: str = "full"
    pooling: str = "mean"
    sentence_transformer_truncate_dim: Optional[int] = None


@dataclass
class LoraTrainingConfig:
    enabled: bool = False
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    task_type: str = "FEATURE_EXTRACTION"
    target_modules: List[str] = field(default_factory=list)
    exclude_modules: List[str] = field(default_factory=lambda: ["lm_head", "classifier"])
    modules_to_save: List[str] = field(default_factory=list)
    adapter_name: str = "default"
    output_adapter_dir: str = "outputs/models/lora_adapter"


@dataclass
class HubTrainingConfig:
    enabled: bool = False
    repo_prefix: str = ""
    private: bool = True
    token_env: str = "HF_TOKEN"
    auto_repo_from_experiment: bool = True
    repo_suffix_full: str = "finetuned-model"
    repo_suffix_lora: str = "lora-adapter"


@dataclass
class TrainingConfig:
    enabled: bool = False
    finetune_strategy: str = "full"
    output_model_dir: str = "outputs/models/finetuned"
    base_loss: str = "MultipleNegativesRankingLoss"
    use_matryoshka: bool = True
    use_2d_matryoshka: bool = False
    matryoshka_dimensions: List[int] = field(default_factory=list)
    n_layers_per_step: int = 1
    epochs: int = 1
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_steps: int = -1
    fp16: bool = True
    bf16: bool = False
    save_steps: int = 1000
    eval_steps: int = 1000
    logging_steps: int = 50
    resume_from_checkpoint: Optional[str] = None
    auto_resume_from_last_checkpoint: bool = False
    lora: LoraTrainingConfig = field(default_factory=LoraTrainingConfig)
    hub: HubTrainingConfig = field(default_factory=HubTrainingConfig)


@dataclass
class UtilityConfig:
    @dataclass
    class RelevanceScaleConfig:
        mode: str = "minmax"
        min_label: float = 0.0
        max_label: float = 3.0
        explicit_map: Dict[str, float] = field(default_factory=dict)

    @dataclass
    class RelevanceConfig:
        mode: str = "weak"
        weak_source: str = "bm25"
        calibration: str = "minmax_score"
        cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        top_k_candidates: int = 200
        rerank_k: int = 50
        uncertainty: str = "margin"
        qrel_adjustment: str = "hard_override"
        high_impact_fraction: float = 0.2
        validation_qids_limit: int = 200
        scale: "UtilityConfig.RelevanceScaleConfig" = field(default_factory=lambda: UtilityConfig.RelevanceScaleConfig())

    metric: str = "relative_score_dissimilarity"
    epsilon: float = 1e-6
    alpha: float = 0.7
    margin_negatives: int = 4
    aggregate: str = "mean"
    sample_pairs_per_query: int = 64
    relevance_threshold: float = 1.0
    seed: int = 13
    default_utility_profile_name: Optional[str] = None
    relevance: "UtilityConfig.RelevanceConfig" = field(default_factory=lambda: UtilityConfig.RelevanceConfig())


@dataclass
class OptimizationConfig:
    budget_bytes: Optional[int] = None
    budget_gb: Optional[float] = 2.0
    algorithm: str = "lagrangian_dual"
    max_iter: int = 64
    tolerance: float = 1e-3
    lambda_low: float = 0.0
    lambda_high: float = 1.0
    streaming_reestimate_every_docs: int = 5000
    mode: str = "batch"


@dataclass
class RetrievalConfig:
    mode: str = "dense_exact"
    top_k: int = 100
    candidate_k: int = 2000
    include_bm25_baseline: bool = False
    batch_size_queries: int = 32
    batch_size_docs: int = 256
    use_pyterrier_bm25: bool = True
    terrier_wmodel: str = "BM25"
    use_torch_gpu_exact: bool = True


@dataclass
class ExecutionConfig:
    output_dir: str = "outputs"
    experiment_name: str = "matryoshka_experiment"
    device: str = "cuda"
    retrieval_device: str = "cuda"
    dtype: str = "float16"
    doc_batch_size: int = 256
    query_batch_size: int = 32
    num_workers: int = 0
    seed: int = 13
    save_embeddings: bool = False
    save_score_pairs: bool = True
    save_runs: bool = True
    verbose: bool = True
    retrieval_vram_utilization_limit: float = 0.9
    retrieval_vram_temp_overhead_factor: float = 1.2
    embedding_checkpoint_enabled: bool = True
    embedding_checkpoint_every_docs: int = 1_000_000
    embedding_checkpoint_dir: Optional[str] = None
    embedding_checkpoint_resume: str = "auto"


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    profiles: List[ProfileConfig]
    utility: UtilityConfig = field(default_factory=UtilityConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @property
    def output_path(self) -> Path:
        return Path(self.execution.output_dir) / self.execution.experiment_name

    def budget_bytes_resolved(self) -> int:
        if self.optimization.budget_bytes is not None:
            return int(self.optimization.budget_bytes)
        if self.optimization.budget_gb is None:
            raise ValueError("Either budget_bytes or budget_gb must be provided.")
        return int(self.optimization.budget_gb * (1024 ** 3))


def _construct_dataclass(cls, payload: Optional[Dict[str, Any]]):
    return cls(**(payload or {}))


def _construct_training_config(payload: Optional[Dict[str, Any]]) -> TrainingConfig:
    payload = dict(payload or {})
    lora_payload = payload.pop("lora", {})
    hub_payload = payload.pop("hub", {})
    return TrainingConfig(
        **payload,
        lora=_construct_dataclass(LoraTrainingConfig, lora_payload),
        hub=_construct_dataclass(HubTrainingConfig, hub_payload),
    )


def _construct_utility_config(payload: Optional[Dict[str, Any]]) -> UtilityConfig:
    payload = dict(payload or {})
    relevance_payload = dict(payload.pop("relevance", {}) or {})
    scale_payload = dict(relevance_payload.pop("scale", {}) or {})
    scale_cfg = _construct_dataclass(UtilityConfig.RelevanceScaleConfig, scale_payload)
    relevance_cfg = UtilityConfig.RelevanceConfig(
        **relevance_payload,
        scale=scale_cfg,
    )
    return UtilityConfig(**payload, relevance=relevance_cfg)


def _validate_choice(name: str, value: str, allowed: set[str]) -> None:
    if value not in allowed:
        raise ValueError(f"Unsupported {name}: {value}. Allowed values: {sorted(allowed)}")


def _apply_deprecated_data_mappings(
    cfg: ExperimentConfig,
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


def _validate_config(cfg: ExperimentConfig) -> None:
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
    
    for profile in cfg.profiles:
        if profile.dimension <= 0:
            raise ValueError(f"Profile `{profile.name}` has non-positive dimension: {profile.dimension}.")
        if profile.cost_is_explicit and (profile.cost_bytes is None or profile.cost_bytes <= 0):
            raise ValueError(f"Profile `{profile.name}` has `cost_is_explicit=true` but invalid `cost_bytes`: {profile.cost_bytes}.")

    _validate_choice("model.backend", cfg.model.backend, {"sentence_transformers", "transformers"})
    _validate_choice("model.similarity", cfg.model.similarity, {"dot", "cosine"})
    _validate_choice("optimization.mode", cfg.optimization.mode, {"batch", "streaming"})
    if cfg.optimization.mode == "streaming":
        raise ValueError(
            "`optimization.mode=streaming` is currently disabled. "
            "Use `optimization.mode=batch`."
        )
    _validate_choice("execution.retrieval_device", cfg.execution.retrieval_device, {"cuda", "cpu"})
    _validate_choice("execution.embedding_checkpoint_resume", cfg.execution.embedding_checkpoint_resume, {"auto", "restart", "fail"})
    _validate_choice("retrieval.mode", cfg.retrieval.mode, {"dense_exact", "pyterrier_candidates"})
    _validate_choice(
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
    _validate_choice("utility.relevance.mode", cfg.utility.relevance.mode, {"weak", "model", "hybrid"})
    _validate_choice(
        "utility.relevance.calibration",
        cfg.utility.relevance.calibration,
        {"minmax_score", "rank_log_discount", "constant_one"},
    )
    _validate_choice(
        "utility.relevance.weak_source",
        cfg.utility.relevance.weak_source,
        {"bm25", "dense", "hybrid_rerank"},
    )
    _validate_choice("utility.relevance.uncertainty", cfg.utility.relevance.uncertainty, {"margin"})
    _validate_choice("utility.relevance.qrel_adjustment", cfg.utility.relevance.qrel_adjustment, {"hard_override", "relevant_only_to_one"})
    _validate_choice("utility.relevance.scale.mode", cfg.utility.relevance.scale.mode, {"minmax", "explicit_map"})
    _validate_choice("data.single_log_policy", cfg.data.single_log_policy, {"shared", "split_by_qrels"})
    _validate_choice("training.finetune_strategy", cfg.training.finetune_strategy.lower(), {"full", "lora"})
    _validate_choice(
        "data.training_format",
        cfg.data.training_format,
        {"hf_triplet", "hf_tevatron_passage", "jsonl_triplet"},
    )
    _validate_choice(
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
    _validate_choice("utility.aggregate", cfg.utility.aggregate, {"mean"})

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


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    for key in ("data", "model", "profiles"):
        if key not in raw:
            raise ValueError(f"Missing required top-level config section: `{key}`")

    cfg = ExperimentConfig(
        data=_construct_dataclass(DataConfig, raw["data"]),
        model=_construct_dataclass(ModelConfig, raw["model"]),
        profiles=[_construct_dataclass(ProfileConfig, item) for item in raw["profiles"]],
        utility=_construct_utility_config(raw.get("utility", {})),
        optimization=_construct_dataclass(OptimizationConfig, raw.get("optimization", {})),
        retrieval=_construct_dataclass(RetrievalConfig, raw.get("retrieval", {})),
        execution=_construct_dataclass(ExecutionConfig, raw.get("execution", {})),
        training=_construct_training_config(raw.get("training", {})),
    )
    _apply_deprecated_data_mappings(
        cfg,
        raw_data=raw.get("data"),
        raw_retrieval=raw.get("retrieval"),
    )
    _validate_config(cfg)
    return cfg


def save_config_snapshot(config: ExperimentConfig, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)

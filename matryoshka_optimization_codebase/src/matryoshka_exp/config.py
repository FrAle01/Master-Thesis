from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .config_validation import apply_deprecated_data_mappings, validate_config


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
    local_eval_corpus_path: Optional[str] = None


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
    class ResidualNormConfig:
        norm: int = 2
        epsilon: float = 1e-12

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

    @dataclass
    class TailConfig:
        mode: str = "static_default"
        target_residual: float = 0.01
        conservatism_beta: Optional[float] = 0.5
        expected_fraction: float = 0.5
        max_tail_relevance: float = 0.01
        profile_quality: str = "cosine_padding"
        sampled_quality_queries: int = 32
        utility_low: float = 0.0

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
    tail: "UtilityConfig.TailConfig" = field(default_factory=lambda: UtilityConfig.TailConfig())
    estimator: str = "query_log"
    residual_norm: "UtilityConfig.ResidualNormConfig" = field(default_factory=lambda: UtilityConfig.ResidualNormConfig())


@dataclass
class OptimizationConfig:
    budget_bytes: Optional[int] = None
    budget_gb: Optional[float] = 2.0
    algorithm: str = "lagrangian_relaxation"
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
    tail_payload = dict(payload.pop("tail", {}) or {})
    residual_norm_payload = dict(payload.pop("residual_norm", {}) or {})
    scale_payload = dict(relevance_payload.pop("scale", {}) or {})
    scale_cfg = _construct_dataclass(UtilityConfig.RelevanceScaleConfig, scale_payload)
    relevance_cfg = UtilityConfig.RelevanceConfig(
        **relevance_payload,
        scale=scale_cfg,
    )
    tail_cfg = _construct_dataclass(UtilityConfig.TailConfig, tail_payload)
    residual_norm_cfg = _construct_dataclass(UtilityConfig.ResidualNormConfig, residual_norm_payload)
    return UtilityConfig(
        **payload,
        relevance=relevance_cfg,
        tail=tail_cfg,
        residual_norm=residual_norm_cfg,
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
    apply_deprecated_data_mappings(
        cfg,
        raw_data=raw.get("data"),
        raw_retrieval=raw.get("retrieval"),
    )
    validate_config(cfg)
    return cfg


def save_config_snapshot(config: ExperimentConfig, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)

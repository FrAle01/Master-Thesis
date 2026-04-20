from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ProfileConfig:
    name: str
    dimension: int
    layer: Optional[int] = None
    normalize: bool = True
    cost_bytes: Optional[int] = None


@dataclass
class DataConfig:
    pyterrier_dataset: Optional[str] = None
    dataset_provider: Optional[str] = None
    topics_variant: Optional[str] = None
    qrels_variant: Optional[str] = None
    text_fields: List[str] = field(default_factory=lambda: ["text"])
    max_docs: Optional[int] = None
    max_queries: Optional[int] = None
    topic_column: str = "query"
    docno_column: str = "docno"
    local_corpus_path: Optional[str] = None
    local_topics_path: Optional[str] = None
    local_qrels_path: Optional[str] = None
    training_dataset_name: Optional[str] = None
    training_split: str = "train"
    training_format: str = "hf_triplet"
    query_column: str = "query"
    positive_column: str = "positive"
    negative_column: Optional[str] = "negative"
    candidate_source: str = "pyterrier_bm25"
    candidates_per_query: int = 200
    local_terrier_index_path: Optional[str] = None
    build_local_terrier_index_if_missing: bool = False
    terrier_index_threads: int = 4
    terrier_index_overwrite: bool = False
    terrier_meta_lengths: Dict[str, int] = field(default_factory=lambda: {"docno": 64, "text": 4096})


@dataclass
class ModelConfig:
    mode: str = "pretrained"
    backend: str = "sentence_transformers"
    model_name_or_path: str = "nomic-ai/modernbert-embed-base"
    tokenizer_name_or_path: Optional[str] = None
    trust_remote_code: bool = False
    query_prompt: str = ""
    document_prompt: str = ""
    similarity: str = "dot"
    full_profile_name: str = "full"
    pooling: str = "mean"
    sentence_transformer_truncate_dim: Optional[int] = None


@dataclass
class TrainingConfig:
    enabled: bool = False
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


@dataclass
class UtilityConfig:
    metric: str = "relative_score_dissimilarity"
    epsilon: float = 1e-6
    alpha: float = 0.7
    margin_negatives: int = 4
    aggregate: str = "mean"
    sample_pairs_per_query: int = 64
    seed: int = 13


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
    candidate_k: int = 200
    batch_size_queries: int = 32
    batch_size_docs: int = 256
    use_pyterrier_bm25: bool = True
    terrier_wmodel: str = "BM25"


@dataclass
class ExecutionConfig:
    output_dir: str = "outputs"
    experiment_name: str = "matryoshka_experiment"
    device: str = "cuda"
    dtype: str = "float16"
    doc_batch_size: int = 256
    query_batch_size: int = 32
    num_workers: int = 0
    seed: int = 13
    save_embeddings: bool = False
    save_score_pairs: bool = True
    save_runs: bool = True
    verbose: bool = True


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


def _construct_dataclass(cls, payload: Dict[str, Any]):
    return cls(**payload)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = ExperimentConfig(
        data=_construct_dataclass(DataConfig, raw["data"]),
        model=_construct_dataclass(ModelConfig, raw["model"]),
        profiles=[_construct_dataclass(ProfileConfig, item) for item in raw["profiles"]],
        utility=_construct_dataclass(UtilityConfig, raw.get("utility", {})),
        optimization=_construct_dataclass(OptimizationConfig, raw.get("optimization", {})),
        retrieval=_construct_dataclass(RetrievalConfig, raw.get("retrieval", {})),
        execution=_construct_dataclass(ExecutionConfig, raw.get("execution", {})),
        training=_construct_dataclass(TrainingConfig, raw.get("training", {})),
    )
    return cfg


def save_config_snapshot(config: ExperimentConfig, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)

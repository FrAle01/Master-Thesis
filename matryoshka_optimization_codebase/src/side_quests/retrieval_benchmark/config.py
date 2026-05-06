from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


_ALLOWED_MODEL_TYPES = {"hf_model", "local_model", "lora_adapter"}


@dataclass
class DatasetConfig:
    pyterrier_dataset: Optional[str] = "irds:msmarco-passage/trec-dl-2019"
    topics_variant: Optional[str] = "test"
    qrels_variant: Optional[str] = "test"
    text_fields: List[str] = field(default_factory=lambda: ["text"])
    topic_column: str = "query"
    docno_column: str = "docno"
    max_docs: Optional[int] = None
    max_queries: Optional[int] = None
    local_corpus_path: Optional[str] = None
    local_topics_path: Optional[str] = None
    local_qrels_path: Optional[str] = None


@dataclass
class ModelEntry:
    id: str
    type: str
    path_or_repo: str
    base_model: Optional[str] = None
    adapter_name: str = "default"
    trust_remote_code: bool = False
    query_prompt: str = ""
    document_prompt: str = ""
    normalize: bool = True


@dataclass
class RetrievalConfig:
    top_k: int = 100
    batch_size_queries: int = 32
    batch_size_docs: int = 256
    similarity: str = "dot"
    mode: str = "pyterrier_dr_faiss"
    candidate_k: int = 200
    terrier_wmodel: str = "BM25"
    faiss_enabled: bool = True
    faiss_use_gpu: bool = True
    fallback_to_exact_on_backend_error: bool = True
    faiss_backend: str = "flat"
    require_gpu: bool = True


@dataclass
class EvaluationConfig:
    metrics: List[str] = field(
        default_factory=lambda: ["ndcg_cut_10", "map", "recip_rank", "recall_100"]
    )


@dataclass
class OutputConfig:
    output_dir: str = "outputs/side_quests/retrieval_benchmark"
    run_name: str = "default"


@dataclass
class BenchmarkConfig:
    dataset: DatasetConfig
    models: List[ModelEntry]
    dimensions: List[int]
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    seed: int = 13
    device: str = "cuda"

    @property
    def output_path(self) -> Path:
        return Path(self.output.output_dir) / self.output.run_name


def _dc(cls, payload: Optional[Dict[str, Any]]):
    return cls(**(payload or {}))


def _validate(cfg: BenchmarkConfig) -> None:
    if not cfg.models:
        raise ValueError("`models` must contain at least one model entry.")

    ids = set()
    for model in cfg.models:
        if model.id in ids:
            raise ValueError(f"Duplicate model id found: {model.id}")
        ids.add(model.id)

        if model.type not in _ALLOWED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model type: {model.type}. Allowed: {sorted(_ALLOWED_MODEL_TYPES)}"
            )
        if model.type == "lora_adapter" and not model.base_model:
            raise ValueError(
                f"Model '{model.id}' is type=lora_adapter but has no `base_model` configured."
            )

    if not cfg.dimensions:
        raise ValueError("`dimensions` must contain at least one value.")
    if any(d <= 0 for d in cfg.dimensions):
        raise ValueError("All `dimensions` values must be > 0.")

    if cfg.retrieval.similarity not in {"dot", "cosine"}:
        raise ValueError("`retrieval.similarity` must be one of: ['dot', 'cosine']")
    if cfg.retrieval.mode not in {"pyterrier_dr_faiss", "dense_exact", "bm25_rerank"}:
        raise ValueError(
            "`retrieval.mode` must be one of: ['pyterrier_dr_faiss', 'dense_exact', 'bm25_rerank']"
        )
    if cfg.retrieval.faiss_backend not in {"flat", "hnsw", "ivf"}:
        raise ValueError("`retrieval.faiss_backend` must be one of: ['flat', 'hnsw', 'ivf']")
    if cfg.retrieval.top_k <= 0:
        raise ValueError("`retrieval.top_k` must be > 0")
    if cfg.retrieval.candidate_k <= 0:
        raise ValueError("`retrieval.candidate_k` must be > 0")

    if cfg.dataset.pyterrier_dataset is None:
        missing = []
        if not cfg.dataset.local_corpus_path:
            missing.append("dataset.local_corpus_path")
        if not cfg.dataset.local_topics_path:
            missing.append("dataset.local_topics_path")
        if not cfg.dataset.local_qrels_path:
            missing.append("dataset.local_qrels_path")
        if missing:
            raise ValueError(
                "When `dataset.pyterrier_dataset` is null, local paths are required. Missing: "
                + ", ".join(missing)
            )


def load_config(path: str | Path) -> BenchmarkConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    for key in ("dataset", "models", "dimensions"):
        if key not in raw:
            raise ValueError(f"Missing required top-level config section: `{key}`")

    cfg = BenchmarkConfig(
        dataset=_dc(DatasetConfig, raw.get("dataset")),
        models=[_dc(ModelEntry, item) for item in raw.get("models", [])],
        dimensions=list(raw.get("dimensions", [])),
        retrieval=_dc(RetrievalConfig, raw.get("retrieval")),
        evaluation=_dc(EvaluationConfig, raw.get("evaluation")),
        output=_dc(OutputConfig, raw.get("output")),
        seed=int(raw.get("seed", 13)),
        device=str(raw.get("device", "cuda")),
    )
    _validate(cfg)
    return cfg

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import Dict, Iterator, List, Optional

import pandas as pd

from ..config import DataConfig, RetrievalConfig
from .terrier_index import TerrierIndexResolver


@dataclass
class CorpusRecord:
    docno: str
    text: str
    raw: Dict


class PyTerrierLoader:
    def __init__(self, data_cfg: DataConfig):
        self.data_cfg = data_cfg
        self._pt = None
        self._dataset = None
        self._logger = logging.getLogger("matryoshka_exp")

    def ensure_initialized(self):
        import pyterrier as pt

        if not pt.started():
            pt.init()
        self._pt = pt
        if self.data_cfg.pyterrier_dataset is not None:
            self._dataset = pt.get_dataset(self.data_cfg.pyterrier_dataset)

    @property
    def pt(self):
        self.ensure_initialized()
        return self._pt

    @property
    def dataset(self):
        self.ensure_initialized()
        return self._dataset

    def _dataset_for(self, pyterrier_dataset: Optional[str]):
        if pyterrier_dataset is None:
            return self.dataset
        return self.pt.get_dataset(pyterrier_dataset)

    def _load_topics_from_source(
        self,
        *,
        local_topics_path: Optional[str],
        pyterrier_dataset: Optional[str],
        topics_variant: Optional[str],
    ) -> pd.DataFrame:
        if local_topics_path:
            topics = pd.read_csv(local_topics_path, sep="\t")
        else:
            dataset = self._dataset_for(pyterrier_dataset)
            topics = (
                dataset.get_topics(topics_variant)
                if topics_variant
                else dataset.get_topics()
            )
        if self.data_cfg.max_queries:
            topics = topics.head(self.data_cfg.max_queries).copy()
        return topics

    def _load_qrels_from_source(
        self,
        *,
        local_qrels_path: Optional[str],
        pyterrier_dataset: Optional[str],
        qrels_variant: Optional[str],
        topics: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if local_qrels_path:
            qrels = pd.read_csv(local_qrels_path, sep="\t")
        else:
            dataset = self._dataset_for(pyterrier_dataset)
            try:
                qrels = (
                    dataset.get_qrels(qrels_variant)
                    if qrels_variant
                    else dataset.get_qrels()
                )
            except Exception as exc:
                logging.getLogger("matryoshka_exp").warning(
                    "Failed to load qrels from dataset %s with variant %s; error: %s",
                    pyterrier_dataset,
                    qrels_variant,
                    exc,
                )
                qrels = pd.DataFrame(columns=["qid", "docno", "label"])
        if self.data_cfg.max_queries:
            topics_for_filter = topics if topics is not None else self.load_topics()
            qrels = qrels[qrels["qid"].isin(topics_for_filter["qid"])]
        return qrels

    def load_topics(self) -> pd.DataFrame:
        return self._load_topics_from_source(
            local_topics_path=self.data_cfg.local_topics_path,
            pyterrier_dataset=self.data_cfg.pyterrier_dataset,
            topics_variant=self.data_cfg.topics_variant,
        )

    def load_qrels(self) -> pd.DataFrame:
        topics = self.load_topics()
        return self._load_qrels_from_source(
            local_qrels_path=self.data_cfg.local_qrels_path,
            pyterrier_dataset=self.data_cfg.pyterrier_dataset,
            qrels_variant=self.data_cfg.qrels_variant,
            topics=topics,
        )

    def has_eval_overrides(self) -> bool:
        return any(
            [
                bool(self.data_cfg.eval_pyterrier_dataset),
                bool(self.data_cfg.eval_dataset_provider),
                bool(self.data_cfg.eval_topics_variant),
                bool(self.data_cfg.eval_qrels_variant),
                bool(self.data_cfg.local_eval_topics_path),
                bool(self.data_cfg.local_eval_qrels_path),
                bool(self.data_cfg.local_eval_corpus_path),
            ]
        )

    def for_evaluation(self) -> "PyTerrierLoader":
        eval_cfg = replace(
            self.data_cfg,
            pyterrier_dataset=self.data_cfg.eval_pyterrier_dataset or self.data_cfg.pyterrier_dataset,
            dataset_provider=self.data_cfg.eval_dataset_provider or self.data_cfg.dataset_provider,
            topics_variant=self.data_cfg.eval_topics_variant or self.data_cfg.topics_variant,
            qrels_variant=self.data_cfg.eval_qrels_variant or self.data_cfg.qrels_variant,
            local_corpus_path=self.data_cfg.local_eval_corpus_path or self.data_cfg.local_corpus_path,
            local_topics_path=self.data_cfg.local_eval_topics_path or self.data_cfg.local_topics_path,
            local_qrels_path=self.data_cfg.local_eval_qrels_path or self.data_cfg.local_qrels_path,
            eval_pyterrier_dataset=None,
            eval_dataset_provider=None,
            eval_topics_variant=None,
            eval_qrels_variant=None,
            local_eval_corpus_path=None,
            local_eval_topics_path=None,
            local_eval_qrels_path=None,
        )
        return PyTerrierLoader(eval_cfg)

    def eval_override_fallback_warnings(self) -> List[str]:
        warnings: List[str] = []
        if self.data_cfg.eval_pyterrier_dataset and not (self.data_cfg.eval_topics_variant or self.data_cfg.local_eval_topics_path):
            warnings.append(
                "eval_pyterrier_dataset is set without eval_topics_variant/local_eval_topics_path; using dataset default topics."
            )
        if self.data_cfg.eval_pyterrier_dataset and not (self.data_cfg.eval_qrels_variant or self.data_cfg.local_eval_qrels_path):
            warnings.append(
                "eval_pyterrier_dataset is set without eval_qrels_variant/local_eval_qrels_path; using dataset default qrels."
            )
        if self.data_cfg.local_eval_topics_path and not self.data_cfg.local_eval_qrels_path:
            warnings.append("local_eval_topics_path is set but local_eval_qrels_path is not; qrels will fall back to dataset source.")
        if self.data_cfg.local_eval_qrels_path and not self.data_cfg.local_eval_topics_path:
            warnings.append("local_eval_qrels_path is set but local_eval_topics_path is not; topics will fall back to dataset source.")
        if self.data_cfg.local_eval_corpus_path and not (
            self.data_cfg.local_eval_topics_path or self.data_cfg.eval_pyterrier_dataset
        ):
            warnings.append(
                "local_eval_corpus_path is set without an explicit evaluation topic source; topics will fall back to the primary source."
            )
        return warnings

    def load_eval_topics(self) -> pd.DataFrame:
        dataset = self.data_cfg.eval_pyterrier_dataset or self.data_cfg.pyterrier_dataset
        variant = self.data_cfg.eval_topics_variant or self.data_cfg.topics_variant
        return self._load_topics_from_source(
            local_topics_path=self.data_cfg.local_eval_topics_path,
            pyterrier_dataset=dataset,
            topics_variant=variant,
        )

    def load_eval_qrels(self, topics: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        dataset = self.data_cfg.eval_pyterrier_dataset or self.data_cfg.pyterrier_dataset
        variant = self.data_cfg.eval_qrels_variant or self.data_cfg.qrels_variant
        topics_frame = topics if topics is not None else self.load_eval_topics()
        return self._load_qrels_from_source(
            local_qrels_path=self.data_cfg.local_eval_qrels_path,
            pyterrier_dataset=dataset,
            qrels_variant=variant,
            topics=topics_frame,
        )

    def iter_corpus(self) -> Iterator[CorpusRecord]:
        if self.data_cfg.local_corpus_path:
            df = pd.read_parquet(self.data_cfg.local_corpus_path)
            emitted = 0
            for row in df.to_dict(orient="records"):
                yield CorpusRecord(
                    docno=str(row[self.data_cfg.docno_column]),
                    text=self._compose_text(row),
                    raw=row,
                )
                emitted += 1
                if self.data_cfg.max_docs is not None and emitted >= self.data_cfg.max_docs:
                    break
            return

        iterator = self.dataset.get_corpus_iter(verbose=True)
        emitted = 0
        for record in iterator:
            yield CorpusRecord(
                docno=str(record[self.data_cfg.docno_column]),
                text=self._compose_text(record),
                raw=record,
            )
            emitted += 1
            if self.data_cfg.max_docs is not None and emitted >= self.data_cfg.max_docs:
                break

    def _compose_text(self, record: Dict) -> str:
        parts: List[str] = []
        for field in self.data_cfg.text_fields:
            value = record.get(field)
            if value is not None and str(value).strip():
                parts.append(str(value))
        return "\n".join(parts).strip()

    def _resolve_terrier_index(self):
        return TerrierIndexResolver(self.data_cfg, self.pt, self.dataset, self._logger).resolve()

    def build_bm25_candidates(self, retrieval_cfg: RetrievalConfig, topics: pd.DataFrame) -> pd.DataFrame:
        pt = self.pt
        index = self._resolve_terrier_index()
        topics_for_retrieval = topics.copy()

        # Prevent Terrier from interpreting "foo:bar" user text as fielded query syntax.
        query_col = "query" if "query" in topics_for_retrieval.columns else self.data_cfg.topic_column
        if query_col in topics_for_retrieval.columns:
            topics_for_retrieval[query_col] = (
                topics_for_retrieval[query_col]
                .astype(str)
                .str.replace(":", " ", regex=False)
            )

        retriever = pt.terrier.Retriever(
            index,
            wmodel=retrieval_cfg.terrier_wmodel,
            num_results=retrieval_cfg.candidate_k,
        )
        res = retriever.transform(topics_for_retrieval)

        expected = {"qid", "docno", "score", "rank"}
        missing = expected.difference(res.columns)
        if missing:
            raise ValueError(f"PyTerrier candidate run is missing columns: {sorted(missing)}")

        return res

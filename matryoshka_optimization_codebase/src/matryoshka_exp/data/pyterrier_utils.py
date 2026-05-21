from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pandas as pd

from ..config import DataConfig, RetrievalConfig


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
            qrels = (
                dataset.get_qrels(qrels_variant)
                if qrels_variant
                else dataset.get_qrels()
            )
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
            ]
        )

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

    def _default_index_path(self) -> Path:
        if self.data_cfg.local_terrier_index_path:
            return Path(self.data_cfg.local_terrier_index_path)
        if self.data_cfg.pyterrier_dataset:
            safe_name = self.data_cfg.pyterrier_dataset.replace(":", "_").replace("/", "_")
            return Path("./indices") / safe_name
        return Path("./indices/local_corpus")

    def _build_iterdict_source(self):
        """
        Return an iterator of dicts suitable for pt.IterDictIndexer.
        We make the output explicit so it works for both ir_datasets corpora
        and local corpora.
        """
        if self.data_cfg.local_corpus_path:
            df = pd.read_parquet(self.data_cfg.local_corpus_path)
            emitted = 0
            for row in df.to_dict(orient="records"):
                out = {"docno": str(row[self.data_cfg.docno_column])}
                for field in self.data_cfg.text_fields:
                    out[field] = str(row.get(field, "") or "")
                yield out
                emitted += 1
                if self.data_cfg.max_docs is not None and emitted >= self.data_cfg.max_docs:
                    break
            return

        emitted = 0
        for record in self.dataset.get_corpus_iter(verbose=True):
            out = {"docno": str(record[self.data_cfg.docno_column])}
            for field in self.data_cfg.text_fields:
                out[field] = str(record.get(field, "") or "")
            yield out
            emitted += 1
            if self.data_cfg.max_docs is not None and emitted >= self.data_cfg.max_docs:
                break

    def _load_or_build_local_index(self):
        pt = self.pt
        index_path = self._default_index_path().resolve()
        data_properties = index_path / "data.properties"
        text_fields = [field for field in self.data_cfg.text_fields if str(field).strip()]

        # Reuse an existing local Terrier index if present.
        if data_properties.exists() and not self.data_cfg.terrier_index_overwrite:
            return pt.IndexRef.of(str(index_path))

        if not self.data_cfg.build_local_terrier_index_if_missing:
            raise RuntimeError(
                f"No built-in Terrier index is available and no local index was found at {index_path}."
            )
        if not text_fields:
            raise ValueError(
                "data.text_fields must contain at least one non-empty field to build a Terrier index."
            )

        index_path.mkdir(parents=True, exist_ok=True)

        meta = self.data_cfg.terrier_meta_lengths or {
            "docno": 64,
            **{field: 4096 for field in text_fields},
        }

        indexer = pt.IterDictIndexer(
            str(index_path),
            meta=meta,
            text_attrs=text_fields,
            threads=self.data_cfg.terrier_index_threads,
            overwrite=self.data_cfg.terrier_index_overwrite,
        )

        source_iter = self._build_iterdict_source()
        index_ref = indexer.index(source_iter)
        return index_ref

    def _resolve_terrier_index(self):
        """
        Try built-in dataset index first; if unavailable, fall back to a local index.
        """
        pt = self.pt
        dataset = self.dataset

        if dataset is not None:
            try:
                built_in = dataset.get_index()
                if built_in is not None:
                    return built_in
            except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                self._logger.warning(
                    "Failed to use built-in Terrier index for dataset %s; falling back to local index. Error: %s",
                    self.data_cfg.pyterrier_dataset,
                    exc,
                )

        return self._load_or_build_local_index()

    def build_bm25_candidates(self, retrieval_cfg: RetrievalConfig, topics: pd.DataFrame) -> pd.DataFrame:
        pt = self.pt
        index = self._resolve_terrier_index()

        retriever = pt.terrier.Retriever(
            index,
            wmodel=retrieval_cfg.terrier_wmodel,
            num_results=retrieval_cfg.candidate_k,
        )
        res = retriever.transform(topics)

        expected = {"qid", "docno", "score", "rank"}
        missing = expected.difference(res.columns)
        if missing:
            raise ValueError(f"PyTerrier candidate run is missing columns: {sorted(missing)}")

        return res

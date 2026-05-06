from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import pandas as pd

from .config import DatasetConfig, RetrievalConfig


@dataclass
class CorpusRecord:
    docno: str
    text: str
    raw: Dict


class DatasetLoader:
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self._pt = None
        self._dataset = None
        self.logger = logging.getLogger("side_quests.retrieval_benchmark")

    def ensure_initialized(self):
        import pyterrier as pt

        if not pt.started():
            pt.init()
        self._pt = pt
        if self.cfg.pyterrier_dataset:
            self._dataset = pt.get_dataset(self.cfg.pyterrier_dataset)

    @property
    def pt(self):
        self.ensure_initialized()
        return self._pt

    @property
    def dataset(self):
        self.ensure_initialized()
        return self._dataset

    def load_topics(self) -> pd.DataFrame:
        if self.cfg.local_topics_path:
            topics = pd.read_csv(self.cfg.local_topics_path, sep="\t")
        else:
            topics = self.dataset.get_topics(self.cfg.topics_variant) if self.cfg.topics_variant else self.dataset.get_topics()
        if self.cfg.max_queries:
            topics = topics.head(self.cfg.max_queries).copy()
        topics["qid"] = topics["qid"].astype(str)
        return topics

    def load_qrels(self) -> pd.DataFrame:
        if self.cfg.local_qrels_path:
            qrels = pd.read_csv(self.cfg.local_qrels_path, sep="\t")
        else:
            qrels = self.dataset.get_qrels(self.cfg.qrels_variant) if self.cfg.qrels_variant else self.dataset.get_qrels()
        qrels["qid"] = qrels["qid"].astype(str)
        return qrels

    def iter_corpus(self) -> Iterator[CorpusRecord]:
        if self.cfg.local_corpus_path:
            df = pd.read_parquet(self.cfg.local_corpus_path)
            emitted = 0
            for row in df.to_dict(orient="records"):
                yield CorpusRecord(
                    docno=str(row[self.cfg.docno_column]),
                    text=self._compose_text(row),
                    raw=row,
                )
                emitted += 1
                if self.cfg.max_docs is not None and emitted >= self.cfg.max_docs:
                    break
            return

        emitted = 0
        for record in self.dataset.get_corpus_iter(verbose=True):
            yield CorpusRecord(
                docno=str(record[self.cfg.docno_column]),
                text=self._compose_text(record),
                raw=record,
            )
            emitted += 1
            if self.cfg.max_docs is not None and emitted >= self.cfg.max_docs:
                break

    def _compose_text(self, record: Dict) -> str:
        parts: List[str] = []
        for field in self.cfg.text_fields:
            value = record.get(field)
            if value is not None and str(value).strip():
                parts.append(str(value))
        return "\n".join(parts).strip()

    def bm25_candidates(self, retrieval_cfg: RetrievalConfig, topics: pd.DataFrame) -> pd.DataFrame:
        pt = self.pt
        index = self.dataset.get_index()
        retriever = pt.terrier.Retriever(
            index,
            wmodel=retrieval_cfg.terrier_wmodel,
            num_results=retrieval_cfg.candidate_k,
        )
        return retriever.transform(topics)

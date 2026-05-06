from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

import torch

from .config import BenchmarkConfig
from .dataset import DatasetLoader
from .evaluation import evaluate_run
from .models import ModelResolver, encode_texts
from .reporting import build_ranking_summary, save_outputs, to_long_metrics, to_wide_metrics
from .retrieval import DenseRetriever


class BenchmarkRunner:
    def __init__(self, cfg: BenchmarkConfig):
        self.cfg = cfg
        self.loader = DatasetLoader(cfg.dataset)
        self.retriever = DenseRetriever(cfg.retrieval)
        self.device = self._resolve_device(cfg.device)

    @staticmethod
    def _resolve_device(requested: str) -> str:
        requested = str(requested).lower()
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def run(self) -> Dict:
        topics = self.loader.load_topics()
        qrels = self.loader.load_qrels()
        corpus_records = list(self.loader.iter_corpus())
        docnos = [r.docno for r in corpus_records]
        doc_texts = [r.text for r in corpus_records]

        query_ids = topics["qid"].astype(str).tolist()
        query_texts = topics[self.cfg.dataset.topic_column].astype(str).tolist()

        resolver = ModelResolver(device=self.device)

        result_rows: List[Dict] = []
        for model_entry in self.cfg.models:
            loaded = resolver.load(model_entry)
            for dim in self.cfg.dimensions:
                doc_embeddings = encode_texts(
                    loaded,
                    doc_texts,
                    is_query=False,
                    dimension=int(dim),
                    batch_size=self.cfg.retrieval.batch_size_docs,
                )
                query_embeddings = encode_texts(
                    loaded,
                    query_texts,
                    is_query=True,
                    dimension=int(dim),
                    batch_size=self.cfg.retrieval.batch_size_queries,
                )

                if self.cfg.retrieval.mode == "bm25_rerank":
                    candidates = self.loader.bm25_candidates(self.cfg.retrieval, topics)
                    query_emb_by_id = {
                        qid: emb.unsqueeze(0) for qid, emb in zip(query_ids, query_embeddings)
                    }
                    doc_emb_by_docno = {
                        docno: emb for docno, emb in zip(docnos, doc_embeddings)
                    }
                    run = self.retriever.rerank_candidates(
                        candidates=candidates,
                        query_emb_by_id=query_emb_by_id,
                        doc_emb_by_docno=doc_emb_by_docno,
                    )
                else:
                    idx_path = self.cfg.output_path / "indices" / f"{model_entry.id}__dim{int(dim)}"
                    run = self.retriever.retrieve(
                        query_ids=query_ids,
                        query_texts=query_texts,
                        query_embeddings=query_embeddings,
                        docnos=docnos,
                        doc_embeddings=doc_embeddings,
                        index_path=idx_path,
                    )

                metrics = evaluate_run(
                    pt=self.loader.pt,
                    topics=topics,
                    qrels=qrels,
                    run=run,
                    metrics=self.cfg.evaluation.metrics,
                )
                result_rows.append(
                    {
                        "model_id": model_entry.id,
                        "model_type": model_entry.type,
                        "dimension": int(dim),
                        "metrics": metrics,
                    }
                )

        long_df = to_long_metrics(result_rows)
        wide_df = to_wide_metrics(long_df)
        summary_df = build_ranking_summary(wide_df)
        metadata = {
            "dataset": asdict(self.cfg.dataset),
            "models": [asdict(m) for m in self.cfg.models],
            "dimensions": self.cfg.dimensions,
            "retrieval": asdict(self.cfg.retrieval),
            "evaluation": asdict(self.cfg.evaluation),
            "resolved_device": self.device,
        }

        save_outputs(
            output_dir=self.cfg.output_path,
            long_df=long_df,
            wide_df=wide_df,
            summary_df=summary_df,
            metadata=metadata,
        )

        return {
            "output_dir": str(self.cfg.output_path),
            "num_models": len(self.cfg.models),
            "num_dimensions": len(self.cfg.dimensions),
            "num_runs": len(result_rows),
        }

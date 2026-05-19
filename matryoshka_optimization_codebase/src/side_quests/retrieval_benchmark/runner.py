from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
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
        max_dim = int(max(self.cfg.dimensions))

        result_rows: List[Dict] = []
        for model_entry in self.cfg.models:
            loaded = resolver.load(model_entry)
            doc_embeddings_full = self._load_or_compute_embeddings(
                loaded=loaded,
                texts=doc_texts,
                ids=docnos,
                is_query=False,
                batch_size=self.cfg.retrieval.batch_size_docs,
                dimension=max_dim,
            )
            query_embeddings_full = self._load_or_compute_embeddings(
                loaded=loaded,
                texts=query_texts,
                ids=query_ids,
                is_query=True,
                batch_size=self.cfg.retrieval.batch_size_queries,
                dimension=max_dim,
            )
            for dim in self.cfg.dimensions:
                dim = int(dim)
                doc_embeddings = doc_embeddings_full[:, :dim].contiguous()
                query_embeddings = query_embeddings_full[:, :dim].contiguous()

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

    def _cache_root(self) -> Path:
        if self.cfg.embedding_cache.cache_dir:
            return Path(self.cfg.embedding_cache.cache_dir)
        return self.cfg.output_path / "embedding_cache"

    def _cache_key(self, *, model_id: str, ids: List[str], is_query: bool, dim: int) -> str:
        role = "query" if is_query else "doc"
        h = hashlib.sha1()
        for item in ids:
            h.update(str(item).encode("utf-8"))
            h.update(b"\n")
        digest = h.hexdigest()[:12]
        return f"{model_id}__{role}__dim{dim}__{digest}"

    def _load_or_compute_embeddings(
        self,
        *,
        loaded,
        texts: List[str],
        ids: List[str],
        is_query: bool,
        batch_size: int,
        dimension: int,
    ) -> torch.Tensor:
        if not self.cfg.embedding_cache.enabled:
            return encode_texts(
                loaded,
                texts,
                is_query=is_query,
                dimension=dimension,
                batch_size=batch_size,
            )

        root = self._cache_root()
        root.mkdir(parents=True, exist_ok=True)
        key = self._cache_key(model_id=loaded.id, ids=ids, is_query=is_query, dim=dimension)
        final_path = root / f"{key}.pt"
        ckpt_path = root / f"{key}.ckpt.pt"

        if self.cfg.embedding_cache.reuse_if_available and final_path.exists():
            payload = torch.load(final_path, map_location="cpu")
            emb = payload["embeddings"]
            if len(payload["ids"]) == len(ids):
                return emb.to(self.device)

        start = 0
        prefix = None
        if ckpt_path.exists():
            payload = torch.load(ckpt_path, map_location="cpu")
            start = int(payload.get("next_idx", 0))
            prefix = payload.get("embeddings")

        chunks = []
        if prefix is not None and prefix.numel() > 0:
            chunks.append(prefix)

        for batch_no, begin in enumerate(range(start, len(texts), batch_size), start=1):
            end = min(begin + batch_size, len(texts))
            cur = encode_texts(
                loaded,
                texts[begin:end],
                is_query=is_query,
                dimension=dimension,
                batch_size=batch_size,
            ).detach().cpu()
            chunks.append(cur)

            if batch_no % self.cfg.embedding_cache.checkpoint_every_batches == 0:
                partial = torch.cat(chunks, dim=0) if chunks else torch.empty((0, dimension))
                torch.save({"next_idx": end, "ids": ids[:end], "embeddings": partial}, ckpt_path)

        full = torch.cat(chunks, dim=0) if chunks else torch.empty((0, dimension))
        torch.save({"ids": ids, "embeddings": full}, final_path)
        if ckpt_path.exists():
            ckpt_path.unlink()
        return full.to(self.device)

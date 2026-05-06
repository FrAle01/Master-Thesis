from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from .config import RetrievalConfig


class DenseRetriever:
    def __init__(self, cfg: RetrievalConfig):
        self.cfg = cfg
        self.logger = logging.getLogger("side_quests.retrieval_benchmark")

    def retrieve(
        self,
        *,
        query_ids: List[str],
        query_texts: List[str],
        query_embeddings: torch.Tensor,
        docnos: List[str],
        doc_embeddings: torch.Tensor,
        index_path: Path,
    ) -> pd.DataFrame:
        if self.cfg.mode == "pyterrier_dr_faiss":
            try:
                return self._retrieve_pyterrier_dr_faiss(
                    query_ids=query_ids,
                    query_texts=query_texts,
                    query_embeddings=query_embeddings,
                    docnos=docnos,
                    doc_embeddings=doc_embeddings,
                    index_path=index_path,
                )
            except Exception as exc:  # noqa: BLE001
                if not self.cfg.fallback_to_exact_on_backend_error:
                    raise
                self.logger.warning(
                    "pyterrier_dr/faiss backend failed (%s); falling back to dense_exact retrieval",
                    exc,
                )
                return self._retrieve_exact(query_ids, query_embeddings, docnos, doc_embeddings)
        if self.cfg.faiss_enabled:
            return self._retrieve_faiss(query_ids, query_embeddings, docnos, doc_embeddings)
        return self._retrieve_exact(query_ids, query_embeddings, docnos, doc_embeddings)

    def rerank_candidates(
        self,
        *,
        candidates: pd.DataFrame,
        query_emb_by_id: Dict[str, torch.Tensor],
        doc_emb_by_docno: Dict[str, torch.Tensor],
    ) -> pd.DataFrame:
        rows = []
        for qid, group in candidates.groupby("qid"):
            q = query_emb_by_id[str(qid)]
            cur = group.copy()
            docs = cur["docno"].astype(str).tolist()
            d = torch.cat([doc_emb_by_docno[doc].unsqueeze(0) for doc in docs], dim=0)
            s = self._score(q, d).flatten().detach().cpu().tolist()
            cur["score"] = s
            cur = cur.sort_values("score", ascending=False).head(self.cfg.top_k).copy()
            cur["rank"] = list(range(1, len(cur) + 1))
            rows.append(cur[["qid", "docno", "score", "rank"]])
        return pd.concat(rows, ignore_index=True)

    def _retrieve_exact(
        self,
        query_ids: List[str],
        query_embeddings: torch.Tensor,
        docnos: List[str],
        doc_embeddings: torch.Tensor,
    ) -> pd.DataFrame:
        scores = self._score(query_embeddings, doc_embeddings)
        top_scores, top_idx = torch.topk(scores, k=min(self.cfg.top_k, doc_embeddings.size(0)), dim=1)

        rows = []
        for i, qid in enumerate(query_ids):
            for rank, (s, idx) in enumerate(zip(top_scores[i].tolist(), top_idx[i].tolist()), start=1):
                rows.append({"qid": str(qid), "docno": str(docnos[idx]), "score": float(s), "rank": rank})
        return pd.DataFrame(rows)

    def _retrieve_faiss(
        self,
        query_ids: List[str],
        query_embeddings: torch.Tensor,
        docnos: List[str],
        doc_embeddings: torch.Tensor,
    ) -> pd.DataFrame:
        try:
            import faiss
        except ImportError:
            self.logger.warning("faiss not installed; falling back to dense_exact retrieval")
            return self._retrieve_exact(query_ids, query_embeddings, docnos, doc_embeddings)

        dim = int(doc_embeddings.shape[1])
        doc_np = doc_embeddings.detach().cpu().numpy()
        q_np = query_embeddings.detach().cpu().numpy()

        if self.cfg.faiss_use_gpu:
            if not hasattr(faiss, "StandardGpuResources"):
                raise RuntimeError(
                    "FAISS GPU symbols not found. Install `faiss-gpu-cu12` and run on a CUDA-capable machine."
                )
            try:
                res = faiss.StandardGpuResources()
                # Build index directly on GPU (no CPU index handoff).
                index = faiss.GpuIndexFlatIP(res, dim)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Failed to build FAISS GPU index directly: {exc}") from exc
        else:
            if self.cfg.require_gpu:
                raise RuntimeError("GPU is required by configuration, but `faiss_use_gpu` is false.")
            index = faiss.IndexFlatIP(dim)

        index.add(doc_np)
        k = min(self.cfg.top_k, len(docnos))
        scores, indices = index.search(q_np, k)

        rows = []
        for i, qid in enumerate(query_ids):
            for rank, (s, idx) in enumerate(zip(scores[i], indices[i]), start=1):
                rows.append({"qid": str(qid), "docno": str(docnos[int(idx)]), "score": float(s), "rank": rank})
        return pd.DataFrame(rows)

    def _retrieve_pyterrier_dr_faiss(
        self,
        *,
        query_ids: List[str],
        query_texts: List[str],
        query_embeddings: torch.Tensor,
        docnos: List[str],
        doc_embeddings: torch.Tensor,
        index_path: Path,
    ) -> pd.DataFrame:
        try:
            import pyterrier_dr as ptdr
        except ImportError as exc:
            raise RuntimeError("pyterrier_dr is not installed") from exc

        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        flex_index = ptdr.FlexIndex(str(index_path))

        docs_df = pd.DataFrame(
            {
                "docno": [str(d) for d in docnos],
                "doc_vec": [v for v in doc_embeddings.detach().cpu().numpy()],
            }
        )
        # Ensure deterministic rebuilding for each model x dimension run.
        flex_index.indexer(mode="overwrite").transform(docs_df)

        retriever = self._build_faiss_retriever(flex_index)
        queries_df = pd.DataFrame(
            {
                "qid": [str(q) for q in query_ids],
                "query": [str(q) for q in query_texts],
                "query_vec": [v for v in query_embeddings.detach().cpu().numpy()],
            }
        )
        run = retriever.transform(queries_df)
        return run[["qid", "docno", "score", "rank"]].copy()

    def _build_faiss_retriever(self, flex_index):
        if self.cfg.faiss_backend == "hnsw":
            factory = flex_index.faiss_hnsw_retriever
        elif self.cfg.faiss_backend == "ivf":
            factory = flex_index.faiss_ivf_retriever
        else:
            factory = flex_index.faiss_flat_retriever

        kwargs = {"num_results": self.cfg.top_k}
        if self.cfg.faiss_use_gpu:
            # API surface may vary by version; try common parameter names safely.
            for gpu_kwarg in ("use_gpu", "gpu"):
                try:
                    trial = dict(kwargs)
                    trial[gpu_kwarg] = True
                    return factory(**trial)
                except TypeError:
                    continue
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(f"FAISS GPU init failed via pyterrier_dr: {exc}") from exc
            if self.cfg.require_gpu:
                raise RuntimeError(
                    "Could not enable GPU FAISS retriever via pyterrier_dr; "
                    "set `retrieval.require_gpu=false` only if CPU fallback is acceptable."
                )
        return factory(**kwargs)

    def _score(self, q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        if self.cfg.similarity == "cosine":
            q = torch.nn.functional.normalize(q, p=2, dim=-1)
            d = torch.nn.functional.normalize(d, p=2, dim=-1)
        return q @ d.T

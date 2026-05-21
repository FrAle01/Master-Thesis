from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..models.base import EncoderAdapter, RepresentationProfile


@dataclass
class EmbeddedCorpus:
    docnos_by_profile: Dict[str, List[str]]
    embeddings_by_profile: Dict[str, torch.Tensor]


class DenseGroupedRetriever:
    """Exact dense retrieval with variable document profiles and GPU-first scoring."""

    def __init__(self, adapter: EncoderAdapter, profiles: Dict[str, RepresentationProfile], similarity: str, top_k: int, device: str = "cuda"):
        self.adapter = adapter
        self.profiles = profiles
        self.similarity = similarity
        self.top_k = top_k
        self.device = device
        self._logger = logging.getLogger("matryoshka_exp")

    def _as_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor if str(tensor.device) == self.device else tensor.to(self.device)

    def search_exact(
        self,
        query_ids: List[str],
        query_embeddings_full: torch.Tensor,
        corpus: EmbeddedCorpus,
        *,
        verbose: bool = True,
    ) -> pd.DataFrame:
        rows = []
        query_embeddings_full = self._as_device(query_embeddings_full)
        prepared_doc_cache: Dict[str, torch.Tensor] = {}
        if self.similarity == "cosine":
            for profile_name, doc_matrix in corpus.embeddings_by_profile.items():
                if doc_matrix.numel() == 0:
                    continue
                doc_matrix = self._as_device(doc_matrix)
                doc_matrix.div_(doc_matrix.norm(dim=1, keepdim=True).clamp_min_(1e-12)) # In-place normalization to save memory, since we'll be reusing these for all queries.
                corpus.embeddings_by_profile[profile_name] = doc_matrix 

        for q_offset, qid in enumerate(
            tqdm(query_ids, desc="Dense exact retrieval", disable=not verbose)
        ):
            q_full = query_embeddings_full[q_offset : q_offset + 1]
            all_scores_t = []
            all_docnos = []
            for profile_name, doc_matrix in corpus.embeddings_by_profile.items():
                if doc_matrix.numel() == 0:
                    continue
                profile = self.profiles[profile_name]
                q_view = q_full[:, : profile.dimension]
                if self.similarity == "cosine":
                    q_prepared = self.adapter.prepare_tensor_for_similarity(q_view)
                    doc_matrix = self._as_device(doc_matrix)
                    scores = self.adapter.similarity(
                        q_prepared,
                        doc_matrix,
                        queries_prepared=True,
                        docs_prepared=True,
                    ).squeeze(0)
                else:
                    doc_matrix = self._as_device(doc_matrix)
                    scores = self.adapter.similarity(q_view, doc_matrix).squeeze(0)
                all_scores_t.append(scores)
                all_docnos.extend(corpus.docnos_by_profile[profile_name])

            if not all_scores_t:
                continue

            all_scores = torch.cat(all_scores_t, dim=0)
            k = min(self.top_k, int(all_scores.shape[0]))
            top_scores, top_idx = torch.topk(all_scores, k=k, largest=True, sorted=True)
            top_scores = top_scores.detach().cpu().tolist()
            top_idx = top_idx.detach().cpu().tolist()

            for rank, (score, idx) in enumerate(zip(top_scores, top_idx), start=1):
                rows.append({
                    "qid": qid,
                    "docno": str(all_docnos[idx]),
                    "score": float(score),
                    "rank": rank,
                })
        return pd.DataFrame(rows, columns=["qid", "docno", "score", "rank"])

    def rerank_candidates(
        self,
        candidates: pd.DataFrame,
        query_embeddings_full: Dict[str, torch.Tensor],
        corpus_lookup: Dict[str, Tuple[str, torch.Tensor]],
        *,
        verbose: bool = True,
    ) -> pd.DataFrame:
        if candidates.empty:
            return pd.DataFrame(columns=["qid", "docno", "score", "rank"])

        rows = []
        grouped = list(candidates.groupby("qid"))
        for qid, group in tqdm(grouped, desc="Dense rerank candidates", disable=not verbose):
            q_full = self._as_device(query_embeddings_full[qid])
            group = group.copy()
            group["docno"] = group["docno"].astype(str)
            in_lookup = group["docno"].isin(corpus_lookup)
            dropped_count = int((~in_lookup).sum())
            if dropped_count:
                self._logger.warning(
                    "Dropping %s candidate docs for qid=%s because embeddings are unavailable in corpus lookup.",
                    dropped_count,
                    qid,
                )
            group = group[in_lookup]
            if group.empty:
                continue

            profile_docnos: Dict[str, List[str]] = {}
            for docno in group["docno"].tolist():
                profile_name, _ = corpus_lookup[docno]
                profile_docnos.setdefault(profile_name, []).append(docno)

            score_map: Dict[str, float] = {}
            for profile_name, docnos in profile_docnos.items():
                profile = self.profiles[profile_name]
                q_view = q_full[:, : profile.dimension]
                doc_batch = torch.stack([corpus_lookup[d][1] for d in docnos], dim=0)
                doc_batch = self._as_device(doc_batch)
                if self.similarity == "cosine":
                    q_prepared = self.adapter.prepare_tensor_for_similarity(q_view)
                    d_prepared = self.adapter.prepare_tensor_for_similarity(doc_batch)
                    scores = self.adapter.similarity(
                        q_prepared,
                        d_prepared,
                        queries_prepared=True,
                        docs_prepared=True,
                    ).squeeze(0).detach().cpu().tolist()
                else:
                    scores = self.adapter.similarity(q_view, doc_batch).squeeze(0).detach().cpu().tolist()
                for d, s in zip(docnos, scores):
                    score_map[d] = float(s)

            reranked = group.copy()
            reranked["score"] = reranked["docno"].astype(str).map(score_map).astype(float)
            reranked = reranked.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
            reranked["rank"] = np.arange(1, len(reranked) + 1)
            rows.append(reranked[["qid", "docno", "score", "rank"]])

        if not rows:
            return pd.DataFrame(columns=["qid", "docno", "score", "rank"])
        return pd.concat(rows, ignore_index=True)

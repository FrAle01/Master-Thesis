from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch

from ..models.base import EncoderAdapter, RepresentationProfile


@dataclass
class EmbeddedCorpus:
    docnos_by_profile: Dict[str, List[str]]
    embeddings_by_profile: Dict[str, torch.Tensor]


class DenseGroupedRetriever:
    """Exact dense retrieval with variable document profiles.

    Documents are partitioned by their chosen profile. For each query, the retriever
    scores the corresponding query representation against each profile-specific matrix
    and merges the top-k candidates globally.
    """

    def __init__(self, adapter: EncoderAdapter, profiles: Dict[str, RepresentationProfile], similarity: str, top_k: int):
        self.adapter = adapter
        self.profiles = profiles
        self.similarity = similarity
        self.top_k = top_k

    def search_exact(
        self,
        query_ids: List[str],
        query_embeddings_full: torch.Tensor,
        corpus: EmbeddedCorpus,
    ) -> pd.DataFrame:
        rows = []
        for q_offset, qid in enumerate(query_ids):
            q_full = query_embeddings_full[q_offset : q_offset + 1]
            scored_chunks = []
            doc_chunks = []
            for profile_name, doc_matrix in corpus.embeddings_by_profile.items():
                if doc_matrix.numel() == 0:
                    continue
                profile = self.profiles[profile_name]
                q_view = q_full[:, : profile.dimension]
                scores = self.adapter.similarity(q_view, doc_matrix).squeeze(0).detach().cpu().numpy()
                scored_chunks.append(scores)
                doc_chunks.append(np.array(corpus.docnos_by_profile[profile_name], dtype=object))

            if not scored_chunks:
                continue
            all_scores = np.concatenate(scored_chunks)
            all_docnos = np.concatenate(doc_chunks)
            order = np.argsort(-all_scores)[: self.top_k]
            for rank, idx in enumerate(order, start=1):
                rows.append(
                    {
                        "qid": qid,
                        "docno": str(all_docnos[idx]),
                        "score": float(all_scores[idx]),
                        "rank": rank,
                    }
                )
        return pd.DataFrame(rows)

    def rerank_candidates(
        self,
        candidates: pd.DataFrame,
        query_embeddings_full: Dict[str, torch.Tensor],
        corpus_lookup: Dict[str, Tuple[str, torch.Tensor]],
    ) -> pd.DataFrame:
        rows = []
        for qid, group in candidates.groupby("qid"):
            q_full = query_embeddings_full[qid]
            scores = []
            for _, row in group.iterrows():
                docno = str(row["docno"])
                profile_name, doc_emb = corpus_lookup[docno]
                profile = self.profiles[profile_name]
                q_view = q_full[:, : profile.dimension]
                score = self.adapter.similarity(q_view, doc_emb.unsqueeze(0)).item()
                scores.append(score)
            reranked = group.copy()
            reranked["score"] = scores
            reranked = reranked.sort_values("score", ascending=False).reset_index(drop=True)
            reranked["rank"] = np.arange(1, len(reranked) + 1)
            rows.append(reranked[["qid", "docno", "score", "rank"]])
        return pd.concat(rows, ignore_index=True)

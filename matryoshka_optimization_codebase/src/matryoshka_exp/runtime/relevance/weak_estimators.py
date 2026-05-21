from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from .calibration import calibrate_constant_one, calibrate_rank_log_discount, calibrate_scores


def build_dense_candidates_with_pyterrier_dr(
    *,
    topics: pd.DataFrame,
    topic_column: str,
    subset_docnos,
    subset_doc_embeddings: torch.Tensor,
    full_query_embeddings: torch.Tensor,
    top_k: int,
    index_path: Path,
) -> pd.DataFrame | None:
    try:
        import pyterrier_dr as ptdr
    except Exception:
        return None

    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    flex_index = ptdr.FlexIndex(str(index_path))

    docs_df = pd.DataFrame(
        {
            "docno": [str(d) for d in subset_docnos],
            "doc_vec": [v for v in subset_doc_embeddings.detach().cpu().numpy()],
        }
    )
    flex_index.indexer(mode="overwrite").transform(docs_df)
    retriever = flex_index.faiss_flat_retriever(num_results=int(top_k))

    queries_df = pd.DataFrame(
        {
            "qid": topics["qid"].astype(str).tolist(),
            "query": topics[topic_column].astype(str).tolist(),
            "query_vec": [v for v in full_query_embeddings.detach().cpu().numpy()],
        }
    )
    run = retriever.transform(queries_df)
    return run.loc[:, ["qid", "docno", "score", "rank"]].copy()


def build_dense_candidates(*, topics: pd.DataFrame, subset_docnos, subset_doc_embeddings: torch.Tensor, full_query_embeddings: torch.Tensor, top_k: int, adapter) -> pd.DataFrame:
    docnos = [str(d) for d in subset_docnos]
    scores = adapter.similarity(full_query_embeddings, subset_doc_embeddings)
    if scores.dim() != 2:
        raise ValueError("Dense candidate scoring must produce a [num_queries, num_docs] tensor.")
    k = min(int(top_k), scores.shape[1])
    top_scores, top_idx = torch.topk(scores, k=k, dim=1)

    rows = []
    qids = topics["qid"].astype(str).tolist()
    for qi, qid in enumerate(qids):
        s = top_scores[qi].detach().cpu().numpy()
        idx = top_idx[qi].detach().cpu().numpy()
        order = np.argsort(-s)
        for rank, p in enumerate(order):
            di = int(idx[p])
            rows.append({"qid": qid, "docno": docnos[di], "score": float(s[p]), "rank": int(rank)})
    return pd.DataFrame(rows)


def estimate_from_candidates(candidates: pd.DataFrame, *, source_mode: str, calibration_mode: str = "minmax_score") -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=["qid", "docno", "relevance_estimated", "relevance_final", "confidence", "uncertainty", "source_mode", "is_qrel_overridden"])
    rows = []
    for qid, group in candidates.groupby("qid", sort=False):
        if calibration_mode == "constant_one":
            rel, conf, unc = calibrate_constant_one(len(group))
        elif calibration_mode == "rank_log_discount":
            ranks = group["rank"].to_numpy(dtype=float)
            rel, conf, unc = calibrate_rank_log_discount(ranks)
        else:
            scores = group["score"].to_numpy(dtype=float)
            rel, conf, unc = calibrate_scores(scores)
        for docno, r, c, u in zip(group["docno"].astype(str).tolist(), rel, conf, unc):
            rows.append(
                {
                    "qid": str(qid),
                    "docno": docno,
                    "relevance_estimated": float(r),
                    "relevance_final": float(r),
                    "confidence": float(c),
                    "uncertainty": float(u),
                    "source_mode": source_mode,
                    "is_qrel_overridden": False,
                }
            )
    return pd.DataFrame(rows)


def combine_hybrid_candidates(bm25: pd.DataFrame, dense: pd.DataFrame, *, rerank_k: int) -> pd.DataFrame:
    bm25 = bm25.loc[:, ["qid", "docno", "score", "rank"]].copy()
    dense = dense.loc[:, ["qid", "docno", "score", "rank"]].copy()
    bm25["source"] = "bm25"
    dense["source"] = "dense"
    merged = pd.concat([bm25, dense], ignore_index=True)
    merged["qid"] = merged["qid"].astype(str)
    merged["docno"] = merged["docno"].astype(str)
    merged = merged.sort_values(["qid", "score"], ascending=[True, False])
    merged = merged.drop_duplicates(subset=["qid", "docno"], keep="first")
    out = merged.groupby("qid", sort=False).head(int(rerank_k)).copy()
    out["rank"] = out.groupby("qid").cumcount()
    return out

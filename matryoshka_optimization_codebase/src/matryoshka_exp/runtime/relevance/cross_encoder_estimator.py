from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from .calibration import calibrate_constant_one, calibrate_rank_log_discount, calibrate_scores


def cross_encoder_rescore(*, candidates: pd.DataFrame, topics: pd.DataFrame, doc_text_by_docno: dict[str, str], model_name: str) -> pd.DataFrame:
    if candidates.empty:
        return candidates.assign(score=[])

    try:
        from sentence_transformers import CrossEncoder
    except Exception as exc:
        raise RuntimeError("CrossEncoder is not available. Install sentence-transformers with cross-encoder support.") from exc

    model = CrossEncoder(model_name)
    topic_text = {str(r["qid"]): str(r["query"]) for r in topics.loc[:, ["qid", "query"]].to_dict(orient="records")}

    joined = candidates.copy()
    joined["qid"] = joined["qid"].astype(str)
    joined["docno"] = joined["docno"].astype(str)
    joined["query"] = joined["qid"].map(topic_text)
    joined["text"] = joined["docno"].map(doc_text_by_docno)
    joined = joined.dropna(subset=["query", "text"]).copy()
    if joined.empty:
        return pd.DataFrame(columns=["qid", "docno", "score", "rank"])

    # PyTerrier apply pipeline for scoring whenever possible.
    try:
        import pyterrier as pt

        if not pt.started():
            pt.init()

        def _batch_score(df: pd.DataFrame):
            pairs = list(zip(df["query"].astype(str).tolist(), df["text"].astype(str).tolist()))
            return np.asarray(model.predict(pairs, show_progress_bar=False), dtype=float)

        reranker = pt.apply.doc_score(_batch_score, batch_size=32, required_columns=["qid", "docno", "query", "text"])
        out = reranker.transform(joined)
    except Exception:
        pairs = list(zip(joined["query"].astype(str).tolist(), joined["text"].astype(str).tolist()))
        out = joined.copy()
        out["score"] = np.asarray(model.predict(pairs, show_progress_bar=False), dtype=float)

    out = out.sort_values(["qid", "score"], ascending=[True, False]).copy()
    out["rank"] = out.groupby("qid").cumcount()
    return out.loc[:, ["qid", "docno", "score", "rank"]]


def estimate_from_cross_scores(candidates: pd.DataFrame, *, source_mode: str, calibration_mode: str = "minmax_score") -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=["qid", "docno", "relevance_estimated", "relevance_final", "confidence", "uncertainty", "source_mode", "is_qrel_overridden"])

    rows = []
    for qid, group in tqdm(candidates.groupby("qid", sort=False), desc="Estimating relevance from cross-encoder scores", total=len(candidates["qid"].unique())):
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

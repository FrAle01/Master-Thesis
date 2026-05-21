from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class RoutedQueryData:
    opt_topics: pd.DataFrame
    opt_qrels: pd.DataFrame
    eval_topics: pd.DataFrame
    eval_qrels: pd.DataFrame
    mode: str
    metadata: Dict[str, int | str]


def route_shared(topics: pd.DataFrame, qrels: pd.DataFrame) -> RoutedQueryData:
    return RoutedQueryData(
        opt_topics=topics.copy(),
        opt_qrels=qrels.copy(),
        eval_topics=topics.copy(),
        eval_qrels=qrels.copy(),
        mode="shared",
        metadata={
            "opt_queries": int(len(topics)),
            "eval_queries": int(len(topics)),
            "qid_overlap": int(topics["qid"].astype(str).nunique()),
        },
    )


def route_split_by_qrels(topics: pd.DataFrame, qrels: pd.DataFrame) -> RoutedQueryData:
    qids_with_qrels = set(qrels["qid"].astype(str).tolist())
    topic_qids = topics["qid"].astype(str)
    eval_mask = topic_qids.isin(qids_with_qrels)
    eval_topics = topics[eval_mask].copy()
    opt_topics = topics[~eval_mask].copy()

    if eval_topics.empty:
        raise ValueError(
            "single_log_policy=split_by_qrels produced an empty evaluation set. "
            "At least one topic with qrels is required."
        )
    if opt_topics.empty:
        raise ValueError(
            "single_log_policy=split_by_qrels produced an empty optimization set. "
            "At least one topic without qrels is required."
        )

    eval_qids = set(eval_topics["qid"].astype(str).tolist())
    opt_qids = set(opt_topics["qid"].astype(str).tolist())
    overlap = eval_qids.intersection(opt_qids)
    if overlap:
        raise ValueError("Split-by-qrels routing produced overlapping optimization/evaluation qids.")

    opt_qrels = qrels[qrels["qid"].astype(str).isin(opt_qids)].copy()
    eval_qrels = qrels[qrels["qid"].astype(str).isin(eval_qids)].copy()
    return RoutedQueryData(
        opt_topics=opt_topics,
        opt_qrels=opt_qrels,
        eval_topics=eval_topics,
        eval_qrels=eval_qrels,
        mode="split_by_qrels",
        metadata={
            "opt_queries": int(len(opt_topics)),
            "eval_queries": int(len(eval_topics)),
            "qids_with_qrels": int(len(qids_with_qrels)),
            "qid_overlap": 0,
        },
    )


def route_dual_source(
    opt_topics: pd.DataFrame,
    opt_qrels: pd.DataFrame,
    eval_topics: pd.DataFrame,
    eval_qrels: pd.DataFrame,
) -> RoutedQueryData:
    opt_qids = set(opt_topics["qid"].astype(str).tolist())
    eval_qids = set(eval_topics["qid"].astype(str).tolist())
    return RoutedQueryData(
        opt_topics=opt_topics.copy(),
        opt_qrels=opt_qrels.copy(),
        eval_topics=eval_topics.copy(),
        eval_qrels=eval_qrels.copy(),
        mode="dual_source",
        metadata={
            "opt_queries": int(len(opt_topics)),
            "eval_queries": int(len(eval_topics)),
            "qid_overlap": int(len(opt_qids.intersection(eval_qids))),
        },
    )

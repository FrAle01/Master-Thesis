from __future__ import annotations

from typing import Dict, List

import pandas as pd


def evaluate_run(*, pt, topics: pd.DataFrame, qrels: pd.DataFrame, run: pd.DataFrame, metrics: List[str]) -> Dict[str, float]:
    eval_df = pt.Experiment(
        [run],
        topics,
        qrels,
        eval_metrics=metrics,
        names=["candidate"],
        filter_by_qrels=True,
        verbose=False,
    )
    return eval_df.iloc[0].to_dict()

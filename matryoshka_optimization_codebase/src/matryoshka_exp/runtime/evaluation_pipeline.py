from __future__ import annotations

from typing import Dict

import pandas as pd


class EvaluationPipeline:
    def __init__(self, config):
        self.config = config

    def evaluate_with_pyterrier(self, *, loader, topics, qrels, runs: Dict[str, pd.DataFrame]):
        pt = loader.pt
        metrics = ["ndcg_cut_10",  "AP_rel2", "RR_rel2", "recall_100"]
        names = list(runs.keys())
        run_frames = [runs[name] for name in names]
        eval_df = pt.Experiment(
            run_frames,
            topics,
            qrels,
            eval_metrics=metrics,
            names=names,
            filter_by_qrels=True,
            baseline=0,
            test='t',
            correction = 'b',
            round=4,
            verbose=False,
        )
        metrics_by_run = {
            str(name): eval_df[eval_df["name"] == name].iloc[0].to_dict()
            for name in names
        }
        return eval_df, metrics_by_run

from __future__ import annotations


class EvaluationPipeline:
    def __init__(self, config):
        self.config = config

    def evaluate_with_pyterrier(self, *, loader, topics, qrels, full_run, opt_run):
        pt = loader.pt
        metrics = ["ndcg_cut_10", "map", "recip_rank", "recall_100"]
        eval_df = pt.Experiment(
            [full_run, opt_run],
            topics,
            qrels,
            eval_metrics=metrics,
            names=["full_embedding", "optimized_embedding"],
            filter_by_qrels=True,
            verbose=False,
        )
        full_metrics = eval_df[eval_df["name"] == "full_embedding"].iloc[0].to_dict()
        opt_metrics = eval_df[eval_df["name"] == "optimized_embedding"].iloc[0].to_dict()
        return eval_df, full_metrics, opt_metrics

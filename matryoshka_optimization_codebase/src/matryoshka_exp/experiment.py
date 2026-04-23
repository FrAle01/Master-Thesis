from __future__ import annotations

from dataclasses import asdict
from typing import Dict

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig, save_config_snapshot
from .data.pyterrier_utils import PyTerrierLoader
from .logging_utils import configure_logging
from .metrics.score_metrics import (
    compute_pointwise_utility,
    hybrid_score_margin_utility,
    relative_margin_utility,
)
from .models.factory import create_encoder
from .optimization.lagrangian import LagrangianProfileOptimizer
from .retrieval.dense import DenseGroupedRetriever
from .retrieval.materialization import encode_full_corpus, materialize_grouped_corpus
from .results.persistence import ensure_dir, save_df, save_json
from .training.sbert_trainer import SbertMatryoshkaFinetuner


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = ensure_dir(config.output_path)
        self.logger = configure_logging(self.output_dir / "run.log", verbose=config.execution.verbose)
        save_config_snapshot(config, self.output_dir / "config.snapshot.yaml")

    def train_if_requested(self) -> None:
        if not self.config.training.enabled:
            return
        finetuner = SbertMatryoshkaFinetuner(self.config, self.logger)
        artifact_dir = finetuner.run()
        strategy = self.config.training.finetune_strategy.lower()
        if strategy == "lora":
            self.config.model.adapter_type = "lora"
            self.config.model.adapter_path = str(artifact_dir)
            self.config.model.adapter_name = self.config.training.lora.adapter_name
            self.logger.info("Updated experiment to use base model + LoRA adapter at %s", artifact_dir)
            return

        self.config.model.model_name_or_path = str(artifact_dir)
        self.config.model.adapter_type = None
        self.config.model.adapter_path = None
        self.config.model.adapter_name = "default"
        self.logger.info("Updated experiment to use fine-tuned model at %s", artifact_dir)

    def run(self) -> Dict:
        if self.config.optimization.mode == "streaming":
            raise NotImplementedError("Not implemented: streaming mode is currently disabled.")

        self.train_if_requested()
        adapter, profiles = create_encoder(self.config)
        profile_by_name = {p.name: p for p in profiles}
        full_profile = profile_by_name[self.config.model.full_profile_name]

        loader = PyTerrierLoader(self.config.data)
        topics = loader.load_topics()
        qrels = loader.load_qrels()
        save_df(topics, self.output_dir / "topics.parquet")
        save_df(qrels, self.output_dir / "qrels.parquet")

        # Encode queries once at full dimensionality.
        query_texts = topics[self.config.data.topic_column].astype(str).tolist()
        query_ids = topics["qid"].astype(str).tolist()
        full_query_embeddings = adapter.embed_texts(
            query_texts,
            full_profile,
            prompt_name="query",
            batch_size=self.config.execution.query_batch_size,
        ).cpu()
        self._sync_profile_costs_with_observed_dtype(profiles, full_query_embeddings, stage="query encoding")
        payload = self._run_batch(loader, adapter, profiles, profile_by_name, full_profile, topics, qrels, full_query_embeddings)

        full_run = payload["full_run"]
        opt_run = payload["opt_run"]
        assignments = payload["assignments"]
        utility_pairs_df = payload["utility_pairs_df"]
        utility_table_df = payload["utility_table_df"]
        docnos = payload["docnos"]
        opt_result = payload["opt_result"]

        if self.config.execution.save_score_pairs:
            save_df(utility_pairs_df, self.output_dir / "sampled_score_pairs.parquet")
        save_df(utility_table_df, self.output_dir / "per_document_utility.parquet")
        save_df(pd.DataFrame([asdict(p) for p in profiles]), self.output_dir / "profile_catalog.csv")
        save_df(assignments, self.output_dir / "assignments.parquet")

        if self.config.execution.save_runs:
            save_df(full_run, self.output_dir / "full_run.parquet")
            save_df(opt_run, self.output_dir / "optimized_run.parquet")

        eval_df, full_metrics, opt_metrics = self._evaluate_with_pyterrier(loader, topics, qrels, full_run, opt_run)
        save_df(eval_df, self.output_dir / "pt_experiment.csv")

        memory_summary = {
            "budget_bytes": self.config.budget_bytes_resolved(),
            "optimized_total_cost_bytes": int(assignments["cost_bytes"].sum()),
            "full_total_cost_bytes": len(docnos) * full_profile.cost_bytes,
            "optimized_avg_cost_bytes": float(assignments["cost_bytes"].mean()),
            "full_cost_bytes_per_doc": full_profile.cost_bytes,
        }
        save_json(memory_summary, self.output_dir / "memory_summary.json")

        summary = {
            "lambda_star": opt_result.lambda_star,
            "feasible": opt_result.feasible,
            "optimized_total_utility": opt_result.total_utility,
            "full_metrics": full_metrics,
            "optimized_metrics": opt_metrics,
            "num_docs": len(docnos),
            "num_queries": len(query_ids),
        }
        save_json(summary, self.output_dir / "summary.json")
        self.logger.info("Finished experiment. Summary: %s", summary)
        return summary

    def _run_batch(self, loader, adapter, profiles, profile_by_name, full_profile, topics, qrels, full_query_embeddings):
        docnos, full_doc_embeddings, metadata = encode_full_corpus(
            loader.iter_corpus(),
            adapter,
            full_profile,
            batch_size=self.config.execution.doc_batch_size,
            prompt_name="document",
            verbose=self.config.execution.verbose,
        )
        self._sync_profile_costs_with_observed_dtype(profiles, full_doc_embeddings, stage="batch document encoding")
        save_df(metadata, self.output_dir / "corpus_metadata.parquet")

        utility_pairs_df, utility_table_df = self._estimate_document_utilities_for_subset(
            adapter=adapter,
            profiles=profiles,
            full_profile=full_profile,
            topics=topics,
            qrels=qrels,
            subset_docnos=docnos,
            subset_doc_embeddings=full_doc_embeddings,
            full_query_embeddings=full_query_embeddings,
        )

        optimizer = LagrangianProfileOptimizer(
            profiles=profiles,
            budget_bytes=self.config.budget_bytes_resolved(),
            max_iter=self.config.optimization.max_iter,
            tolerance=self.config.optimization.tolerance,
        )
        opt_result = optimizer.solve(utility_table_df)
        assignments = opt_result.assignments

        full_assignments = pd.DataFrame(
            {
                "docno": docnos,
                "profile": [full_profile.name] * len(docnos),
                "utility": [1.0] * len(docnos),
                "cost_bytes": [full_profile.cost_bytes] * len(docnos),
            }
        )
        full_corpus, full_lookup = materialize_grouped_corpus(docnos, full_doc_embeddings, full_assignments, profile_by_name)
        opt_corpus, opt_lookup = materialize_grouped_corpus(docnos, full_doc_embeddings, assignments, profile_by_name)

        retriever = DenseGroupedRetriever(adapter, profile_by_name, self.config.model.similarity, self.config.retrieval.top_k)
        query_ids = topics["qid"].astype(str).tolist()
        query_emb_by_id = {qid: emb.unsqueeze(0) for qid, emb in zip(query_ids, full_query_embeddings)}

        if self.config.retrieval.mode == "dense_exact":
            full_run = retriever.search_exact(query_ids, full_query_embeddings, full_corpus)
            opt_run = retriever.search_exact(query_ids, full_query_embeddings, opt_corpus)
        else:
            candidates = loader.build_bm25_candidates(self.config.retrieval, topics)
            save_df(candidates, self.output_dir / "bm25_candidates.parquet")
            full_run = retriever.rerank_candidates(candidates, query_emb_by_id, full_lookup)
            opt_run = retriever.rerank_candidates(candidates, query_emb_by_id, opt_lookup)

        return {
            "full_run": full_run,
            "opt_run": opt_run,
            "assignments": assignments,
            "utility_pairs_df": utility_pairs_df,
            "utility_table_df": utility_table_df,
            "docnos": docnos,
            "opt_result": opt_result,
        }

    def _estimate_document_utilities_for_subset(
        self,
        *,
        adapter,
        profiles,
        full_profile,
        topics,
        qrels,
        subset_docnos,
        subset_doc_embeddings,
        full_query_embeddings,
    ):
        doc_index = {docno: i for i, docno in enumerate(subset_docnos)}
        sample_pairs = self._build_score_sampling_frame(topics, qrels, subset_docnos)
        sample_columns = ["qid", "docno", "profile", "full_score", "reduced_score", "utility"]
        sampled_rows = []
        per_doc_profile_utilities = defaultdict(list)

        if sample_pairs.empty:
            utility_pairs_df = pd.DataFrame(columns=sample_columns)
            aggregated_rows = []
            for docno in subset_docnos:
                for profile in profiles:
                    default_utility = 1.0 if profile.name == full_profile.name else 0.0
                    aggregated_rows.append({"docno": docno, "profile": profile.name, "utility": default_utility})
            utility_table_df = pd.DataFrame(aggregated_rows, columns=["docno", "profile", "utility"])
            return utility_pairs_df, utility_table_df

        qid_to_position = {str(qid): i for i, qid in enumerate(topics["qid"].astype(str).tolist())}
        for qid, group in sample_pairs.groupby("qid"):
            q_offset = qid_to_position[str(qid)]
            q_full = full_query_embeddings[q_offset : q_offset + 1]
            group_docnos = group["docno"].astype(str).tolist()
            group_indices = [doc_index[d] for d in group_docnos]
            doc_full = subset_doc_embeddings[group_indices]
            full_scores = adapter.similarity(q_full, doc_full).squeeze(0).detach().cpu().numpy()

            neg_scores_full = np.roll(full_scores, -1) if len(full_scores) > 1 else None
            for profile in profiles:
                q_reduced = q_full[:, : profile.dimension]
                doc_reduced = doc_full[:, : profile.dimension]
                reduced_scores = adapter.similarity(q_reduced, doc_reduced).squeeze(0).detach().cpu().numpy()

                if self.config.utility.metric == "relative_margin_utility":
                    if neg_scores_full is None:
                        utility = np.ones_like(full_scores)
                    else:
                        neg_scores_reduced = np.roll(reduced_scores, -1)
                        utility = relative_margin_utility(full_scores, neg_scores_full, reduced_scores, neg_scores_reduced, epsilon=self.config.utility.epsilon)
                elif self.config.utility.metric == "hybrid_score_margin_utility":
                    if neg_scores_full is None:
                        utility = compute_pointwise_utility("relative_score_dissimilarity", full_scores, reduced_scores, epsilon=self.config.utility.epsilon, alpha=self.config.utility.alpha)
                    else:
                        neg_scores_reduced = np.roll(reduced_scores, -1)
                        utility = hybrid_score_margin_utility(
                            full_scores,
                            reduced_scores,
                            neg_scores_full,
                            neg_scores_reduced,
                            alpha=self.config.utility.alpha,
                            epsilon=self.config.utility.epsilon,
                        )
                else:
                    utility = compute_pointwise_utility(
                        self.config.utility.metric,
                        full_scores,
                        reduced_scores,
                        epsilon=self.config.utility.epsilon,
                        alpha=self.config.utility.alpha,
                    )

                for docno, s_full, s_red, u in zip(group_docnos, full_scores, reduced_scores, utility):
                    sampled_rows.append({
                        "qid": str(qid),
                        "docno": docno,
                        "profile": profile.name,
                        "full_score": float(s_full),
                        "reduced_score": float(s_red),
                        "utility": float(u),
                    })
                    per_doc_profile_utilities[(docno, profile.name)].append(float(u))

        utility_pairs_df = pd.DataFrame(sampled_rows, columns=sample_columns)
        aggregated_rows = []
        for docno in subset_docnos:
            for profile in profiles:
                values = per_doc_profile_utilities.get((docno, profile.name), [])
                agg = float(np.mean(values)) if values else (1.0 if profile.name == full_profile.name else 0.0)
                aggregated_rows.append({"docno": docno, "profile": profile.name, "utility": agg})
        utility_table_df = pd.DataFrame(aggregated_rows, columns=["docno", "profile", "utility"])
        return utility_pairs_df, utility_table_df

    def _build_score_sampling_frame(self, topics, qrels, subset_docnos):
        subset_docnos = [str(d) for d in subset_docnos]
        max_pairs = self.config.utility.sample_pairs_per_query
        qrels = qrels.copy()
        qrels["qid"] = qrels["qid"].astype(str)
        qrels["docno"] = qrels["docno"].astype(str)

        subset_docno_set = set(subset_docnos)
        sampled = []
        for qid in topics["qid"].astype(str).tolist():
            positives = qrels[qrels["qid"] == qid]["docno"].tolist()
            positives = [d for d in positives if d in subset_docno_set]
            if not positives:
                continue
            negative_pool = [d for d in subset_docnos if d not in set(positives)]
            negatives = negative_pool[: max(0, max_pairs - len(positives))]
            selected = (positives + negatives)[:max_pairs]
            for docno in selected:
                sampled.append({"qid": qid, "docno": docno})
        return pd.DataFrame(sampled, columns=["qid", "docno"])

    def _sync_profile_costs_with_observed_dtype(self, profiles, embeddings: torch.Tensor, *, stage: str) -> None:
        if embeddings.numel() == 0:
            return

        expected_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(self.config.execution.dtype, torch.float32)
        observed_dtype = embeddings.dtype
        if observed_dtype != expected_dtype:
            self.logger.warning(
                "Configured execution dtype is `%s` but %s produced `%s`; profile costs will use observed dtype.",
                self.config.execution.dtype,
                stage,
                observed_dtype,
            )

        bytes_per_value = embeddings.element_size()
        for profile in profiles:
            if profile.cost_is_explicit:
                continue
            profile.cost_bytes = int(profile.dimension * bytes_per_value)

    def _evaluate_with_pyterrier(self, loader, topics, qrels, full_run, opt_run):
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

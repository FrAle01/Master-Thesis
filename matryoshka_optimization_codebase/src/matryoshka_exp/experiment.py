from __future__ import annotations

from dataclasses import asdict
from typing import Dict

import pandas as pd
from tqdm import tqdm

from .config import ExperimentConfig, save_config_snapshot
from .data.pyterrier_utils import PyTerrierLoader
from .logging_utils import configure_logging
from .models.factory import create_encoder
from .optimization.errors import InfeasibleOptimizationError
from .optimization.lagrangian import LagrangianProfileOptimizer
from .results.persistence import ensure_dir, save_df, save_json
from .runtime.device_policy import sync_profile_costs_with_observed_dtype
from .runtime.embedding_pipeline import EmbeddingPipeline
from .runtime.evaluation_pipeline import EvaluationPipeline
from .runtime.retrieval_pipeline import RetrievalPipeline
from .runtime.utility_estimation import UtilityEstimator
from .training.sbert_trainer import SbertMatryoshkaFinetuner


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = ensure_dir(config.output_path)
        self.logger = configure_logging(self.output_dir / "run.log", verbose=config.execution.verbose)
        save_config_snapshot(config, self.output_dir / "config.snapshot.yaml")
        self.embedding_pipeline = EmbeddingPipeline(config, self.output_dir, self.logger)
        self.utility_estimator = UtilityEstimator(config, self.logger)
        self.retrieval_pipeline = RetrievalPipeline(config, self.output_dir, self.logger)
        self.evaluation_pipeline = EvaluationPipeline(config)

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

        query_texts = topics[self.config.data.topic_column].astype(str).tolist()
        query_ids = topics["qid"].astype(str).tolist()
        full_query_embeddings = adapter.embed_texts(
            query_texts,
            full_profile,
            prompt_name="query",
            batch_size=self.config.execution.query_batch_size,
        )
        self.embedding_pipeline.save_query_embeddings_checkpoint(query_ids, full_query_embeddings)
        sync_profile_costs_with_observed_dtype(
            self.logger,
            profiles,
            full_query_embeddings,
            expected_dtype_name=self.config.execution.dtype,
            stage="query encoding",
        )

        payload = self._run_batch(
            loader,
            adapter,
            profiles,
            profile_by_name,
            full_profile,
            topics,
            qrels,
            full_query_embeddings,
        )

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

        eval_df, full_metrics, opt_metrics = self.evaluation_pipeline.evaluate_with_pyterrier(
            loader=loader,
            topics=topics,
            qrels=qrels,
            full_run=full_run,
            opt_run=opt_run,
        )
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
        corpus_records = list(
            tqdm(
                loader.iter_corpus(),
                desc="Loading corpus records",
                disable=not self.config.execution.verbose,
            )
        )
        docnos = [record.docno for record in corpus_records]
        metadata = pd.DataFrame(
            [{"docno": record.docno, "text": record.text} for record in corpus_records],
            columns=["docno", "text"],
        )
        full_doc_embeddings = self.embedding_pipeline.load_or_compute_full_doc_embeddings(
            adapter=adapter,
            full_profile=full_profile,
            docnos=docnos,
            corpus_records=corpus_records,
        )
        sync_profile_costs_with_observed_dtype(
            self.logger,
            profiles,
            full_doc_embeddings,
            expected_dtype_name=self.config.execution.dtype,
            stage="batch document encoding",
        )
        save_df(metadata, self.output_dir / "corpus_metadata.parquet")

        utility_pairs_df, utility_table_df = self.utility_estimator.estimate_for_subset(
            loader=loader,
            adapter=adapter,
            profiles=profiles,
            full_profile=full_profile,
            topics=topics,
            qrels=qrels,
            subset_docnos=docnos,
            subset_doc_embeddings=full_doc_embeddings,
            full_query_embeddings=full_query_embeddings,
            corpus_metadata=metadata,
        )
        if self.config.execution.save_score_pairs:
            save_df(utility_pairs_df, self.output_dir / "sampled_score_pairs.parquet")
        save_df(utility_table_df, self.output_dir / "per_document_utility.parquet")

        optimizer = LagrangianProfileOptimizer(
            profiles=profiles,
            budget_bytes=self.config.budget_bytes_resolved(),
            max_iter=self.config.optimization.max_iter,
            tolerance=self.config.optimization.tolerance,
            lambda_low=self.config.optimization.lambda_low,
            lambda_high=self.config.optimization.lambda_high,
            logger=self.logger,
        )
        opt_result = optimizer.solve(utility_table_df)
        assignments = opt_result.assignments
        save_df(assignments, self.output_dir / "assignments.parquet")
        save_json(
            {
                "lambda_star": opt_result.lambda_star,
                "feasible": bool(opt_result.feasible),
                "total_utility": float(opt_result.total_utility),
                "total_cost_bytes": int(opt_result.total_cost_bytes),
            },
            self.output_dir / "optimization_result.json",
        )
        save_json(self.utility_estimator.last_report, self.output_dir / "relevance_estimation_report.json")

        if not opt_result.feasible:
            budget = self.config.budget_bytes_resolved()
            assigned = int(assignments["cost_bytes"].sum())
            self.logger.error(
                "Optimization infeasible. budget_bytes=%s assigned_cost_bytes=%s relative_overflow=%.6f",
                budget,
                assigned,
                (assigned - budget) / max(budget, 1),
            )
            raise InfeasibleOptimizationError(budget_bytes=budget, assigned_cost_bytes=assigned)

        full_run, opt_run = self.retrieval_pipeline.run(
            loader=loader,
            adapter=adapter,
            profile_by_name=profile_by_name,
            full_profile=full_profile,
            docnos=docnos,
            full_doc_embeddings=full_doc_embeddings,
            assignments=assignments,
            topics=topics,
            full_query_embeddings=full_query_embeddings,
        )

        return {
            "full_run": full_run,
            "opt_run": opt_run,
            "assignments": assignments,
            "utility_pairs_df": utility_pairs_df,
            "utility_table_df": utility_table_df,
            "docnos": docnos,
            "opt_result": opt_result,
        }

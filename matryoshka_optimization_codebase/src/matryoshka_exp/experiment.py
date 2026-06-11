from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict

import pandas as pd
from tqdm import tqdm

from .config import ExperimentConfig, save_config_snapshot
from .data.pyterrier_utils import PyTerrierLoader
from .data.query_routing import route_dual_source, route_shared, route_split_by_qrels
from .logging_utils import configure_logging
from .models.factory import create_encoder
from .optimization.errors import InfeasibleOptimizationError
from .optimization.factory import create_optimizer
from .results.experiment_reporting import (
    build_experiment_summary,
    build_memory_summary,
    build_optimization_report,
)
from .results.persistence import ensure_dir, save_df, save_json
from .runtime.device_policy import sync_profile_costs_with_observed_dtype
from .runtime.embedding_pipeline import EmbeddingPipeline
from .runtime.evaluation_pipeline import EvaluationPipeline
from .runtime.retrieval_pipeline import RetrievalPipeline
from .runtime.utility_estimators import UtilityEstimationRequest, create_utility_estimator
from .training.sbert_trainer import SbertMatryoshkaFinetuner


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = ensure_dir(config.output_path)
        self.logger = configure_logging(self.output_dir / "run.log", verbose=config.execution.verbose)
        save_config_snapshot(config, self.output_dir / "config.snapshot.yaml")
        self.embedding_pipeline = EmbeddingPipeline(config, self.output_dir, self.logger)
        self.utility_estimator = create_utility_estimator(config, self.logger)
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
        if not self.utility_estimator.requires_query_data:
            return self._run_residual_norm(
                loader=loader,
                adapter=adapter,
                profiles=profiles,
                profile_by_name=profile_by_name,
                full_profile=full_profile,
            )

        routed = self._load_and_route_query_data(loader)
        self._persist_query_data(routed)
        utility_qrels = self._utility_qrels(routed)
        query_ids, full_query_embeddings = self._encode_queries(
            adapter=adapter,
            full_profile=full_profile,
            topics=routed.opt_topics,
            usage="opt",
        )
        sync_profile_costs_with_observed_dtype(
            self.logger,
            profiles,
            full_query_embeddings,
            expected_dtype_name=self.config.execution.dtype,
            stage="query encoding",
        )
        self.logger.info(
            "Starting optimization using optimization query log (queries=%d).",
            len(routed.opt_topics),
        )

        payload = self._run_batch(
            loader,
            adapter,
            profiles,
            profile_by_name,
            full_profile,
            routed.opt_topics,
            utility_qrels,
            full_query_embeddings,
        )
        return self._evaluate_and_finalize(
            loader=self._evaluation_loader(loader),
            adapter=adapter,
            profiles=profiles,
            profile_by_name=profile_by_name,
            full_profile=full_profile,
            payload=payload,
            eval_topics=routed.eval_topics,
            eval_qrels=routed.eval_qrels,
            num_queries=len(query_ids),
            evaluation_log_message="Starting final evaluation using evaluation query log (queries=%d).",
        )

    def _run_residual_norm(self, *, loader, adapter, profiles, profile_by_name, full_profile) -> Dict:
        eval_loader = self._evaluation_loader(loader)
        eval_topics = eval_loader.load_topics()
        eval_qrels = eval_loader.load_qrels()
        self._persist_evaluation_data(eval_topics, eval_qrels)

        self.logger.info(
            "Starting query-independent optimization with residual norm p=%d using the evaluation corpus.",
            int(self.config.utility.residual_norm.norm),
        )
        payload = self._run_batch(
            eval_loader,
            adapter,
            profiles,
            profile_by_name,
            full_profile,
            topics=None,
            qrels=None,
            full_query_embeddings=None,
        )
        return self._evaluate_and_finalize(
            loader=eval_loader,
            adapter=adapter,
            profiles=profiles,
            profile_by_name=profile_by_name,
            full_profile=full_profile,
            payload=payload,
            eval_topics=eval_topics,
            eval_qrels=eval_qrels,
            num_queries=0,
            evaluation_log_message="Starting final evaluation using evaluation queries only (queries=%d).",
        )

    def _load_and_route_query_data(self, loader):
        topics = loader.load_topics()
        qrels = loader.load_qrels()
        if loader.has_eval_overrides():
            for warning in loader.eval_override_fallback_warnings():
                self.logger.warning("Evaluation source fallback: %s", warning)
            if self.config.data.single_log_policy == "split_by_qrels":
                self.logger.warning(
                    "Evaluation source overrides are set; ignoring data.single_log_policy=split_by_qrels "
                    "and using dual_source routing."
                )
            eval_topics = loader.load_eval_topics()
            eval_qrels = loader.load_eval_qrels(eval_topics)
            routed = route_dual_source(topics, qrels, eval_topics, eval_qrels)
        elif self.config.data.single_log_policy == "split_by_qrels":
            routed = route_split_by_qrels(topics, qrels)
        else:
            routed = route_shared(topics, qrels)

        self._log_query_routing(routed)
        return routed

    def _log_query_routing(self, routed) -> None:
        self.logger.info("Query routing mode selected: %s", routed.mode)
        self.logger.info(
            "Optimization query log summary: dataset=%s topics_variant=%s local_topics=%s "
            "qrels_variant=%s local_qrels=%s num_queries=%s",
            self.config.data.pyterrier_dataset,
            self.config.data.topics_variant,
            self.config.data.local_topics_path,
            self.config.data.qrels_variant,
            self.config.data.local_qrels_path,
            routed.metadata.get("opt_queries"),
        )
        self.logger.info(
            "Evaluation query log summary: dataset=%s topics_variant=%s local_topics=%s "
            "qrels_variant=%s local_qrels=%s num_queries=%s",
            self.config.data.eval_pyterrier_dataset or self.config.data.pyterrier_dataset,
            self.config.data.eval_topics_variant or self.config.data.topics_variant,
            self.config.data.local_eval_topics_path or self.config.data.local_topics_path,
            self.config.data.eval_qrels_variant or self.config.data.qrels_variant,
            self.config.data.local_eval_qrels_path or self.config.data.local_qrels_path,
            routed.metadata.get("eval_queries"),
        )
        self.logger.info(
            "Routing qid overlap between optimization and evaluation logs: %s",
            routed.metadata.get("qid_overlap"),
        )
        if routed.mode == "split_by_qrels":
            self.logger.info(
                "Split-by-qrels stats: qids_with_qrels=%s opt_queries=%s eval_queries=%s",
                routed.metadata.get("qids_with_qrels"),
                routed.metadata.get("opt_queries"),
                routed.metadata.get("eval_queries"),
            )

    def _utility_qrels(self, routed) -> pd.DataFrame:
        qid_overlap = int(routed.metadata.get("qid_overlap", 0))
        if qid_overlap == 0:
            self.logger.info(
                "Qrel override in relevance estimation is ENABLED "
                "(optimization/evaluation qid overlap = 0)."
            )
            return routed.opt_qrels

        self.logger.info(
            "Qrel override in relevance estimation is DISABLED because optimization/evaluation "
            "qids overlap (%d shared qids).",
            qid_overlap,
        )
        return routed.opt_qrels.iloc[0:0].copy()

    def _persist_query_data(self, routed) -> None:
        save_df(routed.opt_topics, self.output_dir / "topics.parquet")
        save_df(routed.opt_qrels, self.output_dir / "qrels.parquet")
        self._persist_evaluation_data(routed.eval_topics, routed.eval_qrels)

    def _persist_evaluation_data(self, topics, qrels) -> None:
        save_df(topics, self.output_dir / "eval_topics.parquet")
        save_df(qrels, self.output_dir / "eval_qrels.parquet")

    def _evaluation_loader(self, loader):
        if not hasattr(loader, "for_evaluation"):
            return loader
        if not self.utility_estimator.requires_query_data:
            return loader.for_evaluation()
        if loader.has_eval_overrides():
            return loader.for_evaluation()
        return loader

    def _encode_queries(self, *, adapter, full_profile, topics, usage):
        query_ids = topics["qid"].astype(str).tolist()
        query_texts = topics[self.config.data.topic_column].astype(str).tolist()
        embeddings = adapter.embed_texts(
            query_texts,
            full_profile,
            prompt_name="query",
            batch_size=self.config.execution.query_batch_size,
        )
        self.embedding_pipeline.save_query_embeddings_checkpoint(
            query_ids,
            embeddings,
            usage=usage,
        )
        return query_ids, embeddings

    def _evaluate_and_finalize(
        self,
        *,
        loader,
        adapter,
        profiles,
        profile_by_name,
        full_profile,
        payload,
        eval_topics,
        eval_qrels,
        num_queries,
        evaluation_log_message,
    ) -> Dict:
        eval_query_ids, eval_query_embeddings = self._encode_queries(
            adapter=adapter,
            full_profile=full_profile,
            topics=eval_topics,
            usage="eval",
        )
        runs = self.retrieval_pipeline.run(
            loader=loader,
            adapter=adapter,
            profile_by_name=profile_by_name,
            full_profile=full_profile,
            docnos=payload["docnos"],
            full_doc_embeddings=payload["full_doc_embeddings"],
            assignments=payload["assignments"],
            topics=eval_topics,
            full_query_embeddings=eval_query_embeddings,
        )
        self._save_profile_catalog(profiles)

        self.logger.info(evaluation_log_message, len(eval_topics))
        eval_df, metrics_by_run = self.evaluation_pipeline.evaluate_with_pyterrier(
            loader=loader,
            topics=eval_topics,
            qrels=eval_qrels,
            runs=runs,
        )
        save_df(eval_df, self.output_dir / "pt_experiment.csv")

        save_json(
            build_memory_summary(
                self.config,
                assignments=payload["assignments"],
                docnos=payload["docnos"],
                full_profile=full_profile,
            ),
            self.output_dir / "memory_summary.json",
        )
        summary = build_experiment_summary(
            self.config,
            payload=payload,
            metrics_by_run=metrics_by_run,
            num_queries=num_queries,
            num_eval_queries=len(eval_query_ids),
        )
        save_json(summary, self.output_dir / "summary.json")
        self.logger.info("Finished experiment. Summary: %s", summary)
        return summary

    def _save_profile_catalog(self, profiles) -> None:
        catalog = pd.DataFrame([asdict(profile) for profile in profiles])
        save_df(catalog, self.output_dir / "profile_catalog.csv")

    def _run_batch(
        self,
        loader,
        adapter,
        profiles,
        _profile_by_name,
        full_profile,
        topics=None,
        qrels=None,
        full_query_embeddings=None,
    ):
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
        save_df(metadata, Path(self.config.execution.output_dir) / "corpus_metadata.parquet")

        estimation = self.utility_estimator.estimate(
            UtilityEstimationRequest(
                profiles=profiles,
                full_profile=full_profile,
                docnos=docnos,
                doc_embeddings=full_doc_embeddings,
                loader=loader,
                adapter=adapter,
                topics=topics,
                qrels=qrels,
                query_embeddings=full_query_embeddings,
                corpus_metadata=metadata,
            )
        )
        utility_pairs_df = estimation.pair_details
        utility_table_df = estimation.utility_table
        if self.config.execution.save_score_pairs and not utility_pairs_df.empty:
            save_df(utility_pairs_df, self.output_dir / "sampled_score_pairs.parquet")
        save_df(utility_table_df, self.output_dir / "per_document_utility.parquet")

        optimizer = create_optimizer(self.config, profiles, self.logger)
        opt_result = optimizer.solve(utility_table_df)
        assignments = opt_result.assignments
        save_df(assignments, self.output_dir / "assignments.parquet")
        save_json(
            build_optimization_report(opt_result),
            self.output_dir / "optimization_result.json",
        )
        save_json(estimation.report, self.output_dir / "utility_estimation_report.json")
        if self.utility_estimator.requires_query_data:
            save_json(estimation.report, self.output_dir / "relevance_estimation_report.json")

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

        return {
            "assignments": assignments,
            "utility_pairs_df": utility_pairs_df,
            "utility_table_df": utility_table_df,
            "docnos": docnos,
            "opt_result": opt_result,
            "full_doc_embeddings": full_doc_embeddings,
        }

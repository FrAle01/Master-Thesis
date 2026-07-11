from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..optimization.errors import InfeasibleOptimizationError
from ..optimization.factory import create_optimizer
from ..results.experiment_reporting import build_optimization_report
from ..results.persistence import save_df, save_json
from .device_policy import sync_profile_costs_with_observed_dtype
from .utility_estimators import UtilityEstimationRequest


class BatchOptimizationPipeline:
    def __init__(self, config, output_dir, logger, embedding_pipeline, utility_estimator):
        self.config = config
        self.output_dir = output_dir
        self.logger = logger
        self.embedding_pipeline = embedding_pipeline
        self.utility_estimator = utility_estimator

    def run(
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

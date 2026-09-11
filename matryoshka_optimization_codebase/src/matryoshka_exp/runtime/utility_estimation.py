from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..results.pair_writer import PairRowsParquetWriter
from .relevance.estimation import PairRelevanceEstimator
from .relevance.tail_utility import build_rbp_residual_tail_utilities
from .score_preservation import compute_score_preservation
from .utility_tables import aggregate_utility_table, default_utility_table, tail_lookup_from_table
from .utility_estimators.contracts import (
    UtilityEstimationRequest,
    UtilityEstimationResult,
    UtilityEstimator as UtilityEstimatorContract,
)


class QueryLogUtilityEstimator(UtilityEstimatorContract):
    requires_query_data = True

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.pair_relevance_estimator = PairRelevanceEstimator(config, logger)

    def estimate(self, request: UtilityEstimationRequest) -> UtilityEstimationResult:
        required = {
            "loader": request.loader,
            "adapter": request.adapter,
            "topics": request.topics,
            "qrels": request.qrels,
            "query_embeddings": request.query_embeddings,
            "corpus_metadata": request.corpus_metadata,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"Query-log utility estimation requires: {missing}")
        pair_details, utility_table = self.estimate_for_subset(
            loader=request.loader,
            adapter=request.adapter,
            profiles=request.profiles,
            full_profile=request.full_profile,
            topics=request.topics,
            qrels=request.qrels,
            subset_docnos=request.docnos,
            subset_doc_embeddings=request.doc_embeddings,
            full_query_embeddings=request.query_embeddings,
            corpus_metadata=request.corpus_metadata,
        )
        self.last_report = {
            "estimator": "query_log",
            "requires_query_data": True,
            **self.last_report,
        }
        return UtilityEstimationResult(
            utility_table=utility_table,
            pair_details=pair_details,
            report=self.last_report,
        )

    def _compute_score_preservation(
        self,
        *,
        full_scores: np.ndarray,
        reduced_scores: np.ndarray,
    ) -> np.ndarray:
        return compute_score_preservation(
            metric=str(self.config.utility.metric),
            full_scores=full_scores,
            reduced_scores=reduced_scores,
            epsilon=float(self.config.utility.epsilon),
            alpha=float(self.config.utility.alpha),
            margin_negatives=max(1, int(self.config.utility.margin_negatives)),
        )

    def estimate_for_subset(
        self,
        *,
        loader,
        adapter,
        profiles,
        full_profile,
        topics,
        qrels,
        subset_docnos,
        subset_doc_embeddings,
        full_query_embeddings,
        corpus_metadata,
    ):
        sample_columns = [
            "qid",
            "docno",
            "profile",
            "full_score",
            "reduced_score",
            "score_preservation",
            "utility",
            "relevance_estimated",
            "relevance_final",
            "confidence",
            "uncertainty",
            "source_mode",
            "is_qrel_overridden",
        ]
        doc_index = {str(docno): i for i, docno in enumerate(subset_docnos)}
        pair_relevance_df = self.pair_relevance_estimator.estimate(
            loader=loader,
            adapter=adapter,
            topics=topics,
            qrels=qrels,
            subset_docnos=subset_docnos,
            subset_doc_embeddings=subset_doc_embeddings,
            full_query_embeddings=full_query_embeddings,
            corpus_metadata=corpus_metadata,
        )
        if pair_relevance_df.empty:
            self.logger.warning("No relevance pairs available. Returning default utilities.")
            utility_pairs_df = pd.DataFrame(
                columns=sample_columns
            )
            tail_table_df, tail_report = self._build_tail_utility_table(
                adapter=adapter,
                profiles=profiles,
                full_profile=full_profile,
                topics=topics,
                subset_docnos=subset_docnos,
                subset_doc_embeddings=subset_doc_embeddings,
                full_query_embeddings=full_query_embeddings,
                unranked_docnos=[str(d) for d in subset_docnos],
            )
            tail_lookup = tail_lookup_from_table(tail_table_df)
            utility_table_df = default_utility_table(
                subset_docnos,
                profiles,
                full_profile,
                default_utility_profile_name=self.config.utility.default_utility_profile_name,
                tail_lookup=tail_lookup,
            )
            self.last_report = {
                "relevance_mode": self.config.utility.relevance.mode,
                "coverage_pairs": 0,
                "num_docs": len(subset_docnos),
                "qrel_overrides": 0,
                "default_docs": int(len(subset_docnos)),
                **tail_report,
            }
            return utility_pairs_df, utility_table_df
        else:
            self.logger.info("Estimating utilities for %d relevance pairs.", len(pair_relevance_df))

        stream_pairs_to_disk = bool(self.config.execution.save_score_pairs)
        pair_writer = None
        if stream_pairs_to_disk:
            try:
                pair_writer = PairRowsParquetWriter(self.config.output_path / "sampled_score_pairs.parquet", sample_columns)
            except Exception as exc:
                self.logger.warning(
                    "Failed to initialize streaming score-pair writer (%s). Falling back to in-memory collection.",
                    exc,
                )
                stream_pairs_to_disk = False
        sampled_rows = [] if not stream_pairs_to_disk else None
        per_doc_profile_utilities_sum = defaultdict(float)
        per_doc_profile_utilities_count = defaultdict(float)
        qid_to_position = {str(qid): i for i, qid in enumerate(topics["qid"].astype(str).tolist())}

        for qid, group in tqdm(
            pair_relevance_df.groupby("qid", sort=False),
            desc="Estimating per-doc utilities",
            disable=not self.config.execution.verbose,
        ):
            q_offset = qid_to_position.get(str(qid))
            if q_offset is None:
                continue
            q_full = full_query_embeddings[q_offset : q_offset + 1]
            group_docnos = group["docno"].astype(str).tolist()
            group_indices = [doc_index[d] for d in group_docnos if d in doc_index]
            valid_docnos = [d for d in group_docnos if d in doc_index]
            if not group_indices:
                continue

            rel_map = dict(zip(group["docno"].astype(str), group["relevance_final"].astype(float)))
            est_rel_map = dict(zip(group["docno"].astype(str), group["relevance_estimated"].astype(float)))
            conf_map = dict(zip(group["docno"].astype(str), group["confidence"].astype(float)))
            unc_map = dict(zip(group["docno"].astype(str), group["uncertainty"].astype(float)))
            src_map = dict(zip(group["docno"].astype(str), group["source_mode"].astype(str)))
            over_map = dict(zip(group["docno"].astype(str), group["is_qrel_overridden"].astype(bool)))

            doc_full = subset_doc_embeddings[group_indices]
            full_scores = adapter.similarity(q_full, doc_full).squeeze(0)
            full_scores_np = full_scores.detach().cpu().numpy()

            for profile in profiles:
                q_reduced = q_full[:, : profile.dimension]
                doc_reduced = doc_full[:, : profile.dimension]
                reduced_scores = adapter.similarity(q_reduced, doc_reduced).squeeze(0)

                reduced_scores_np = reduced_scores.detach().cpu().numpy()
                preservation_np = self._compute_score_preservation(
                    full_scores=full_scores_np,
                    reduced_scores=reduced_scores_np,
                )

                for docno, s_full, s_red, score_pres in zip(valid_docnos, full_scores_np, reduced_scores_np, preservation_np):
                    rel = float(rel_map.get(docno, 0.0))
                    utility = float(rel * score_pres)
                    row = {
                        "qid": str(qid),
                        "docno": docno,
                        "profile": profile.name,
                        "full_score": float(s_full),
                        "reduced_score": float(s_red),
                        "score_preservation": float(score_pres),
                        "utility": utility,
                        "relevance_estimated": float(est_rel_map.get(docno, rel)),
                        "relevance_final": rel,
                        "confidence": float(conf_map.get(docno, 0.0)),
                        "uncertainty": float(unc_map.get(docno, 1.0)),
                        "source_mode": str(src_map.get(docno, "unknown")),
                        "is_qrel_overridden": bool(over_map.get(docno, False)),
                    }
                    if pair_writer is not None:
                        pair_writer.add_row(row)
                    else:
                        sampled_rows.append(row)
                    per_doc_profile_utilities_count[(docno, profile.name)] += 1
                    per_doc_profile_utilities_sum[(docno, profile.name)] += utility

        if pair_writer is not None:
            pair_writer.close()
            utility_pairs_df = pd.DataFrame(columns=sample_columns)
        else:
            utility_pairs_df = pd.DataFrame(sampled_rows, columns=sample_columns)
        covered_docnos = set(pair_relevance_df["docno"].astype(str).unique())
        unranked_docnos = [str(d) for d in subset_docnos if str(d) not in covered_docnos]
        tail_table_df, tail_report = self._build_tail_utility_table(
            adapter=adapter,
            profiles=profiles,
            full_profile=full_profile,
            topics=topics,
            subset_docnos=subset_docnos,
            subset_doc_embeddings=subset_doc_embeddings,
            full_query_embeddings=full_query_embeddings,
            unranked_docnos=unranked_docnos,
        )
        utility_table_df, default_count = aggregate_utility_table(
            subset_docnos,
            profiles,
            full_profile,
            per_doc_profile_utilities_sum,
            per_doc_profile_utilities_count,
            default_utility_profile_name=self.config.utility.default_utility_profile_name,
            tail_lookup=tail_lookup_from_table(tail_table_df),
        )

        overrides = int(pair_relevance_df["is_qrel_overridden"].sum()) if not pair_relevance_df.empty else 0
        self.last_report = {
            "relevance_mode": self.config.utility.relevance.mode,
            "weak_source": self.config.utility.relevance.weak_source,
            "coverage_pairs": int(len(pair_relevance_df)),
            "covered_docs": int(pair_relevance_df["docno"].nunique()),
            "num_docs": int(len(subset_docnos)),
            "qrel_overrides": overrides,
            "default_docs": int(default_count),
            "default_doc_fraction": float(default_count / max(1, len(subset_docnos))),
            **tail_report,
        }
        self.logger.info("Relevance estimation report: %s", self.last_report)
        return utility_pairs_df, utility_table_df

    def _build_tail_utility_table(
        self,
        *,
        adapter,
        profiles,
        full_profile,
        topics,
        subset_docnos,
        subset_doc_embeddings,
        full_query_embeddings,
        unranked_docnos,
    ):
        if str(self.config.utility.tail.mode) != "rbp_residual_interval":
            self.logger.info(
                "Tail utility mode is %s; using static fallback for %d unranked docs.",
                self.config.utility.tail.mode,
                len(unranked_docnos),
            )
            return pd.DataFrame(columns=["docno", "profile", "utility", "profile_quality"]), self._static_tail_report(unranked_docnos)
        self.logger.info(
            "Building RBP residual tail utilities for %d unranked docs with target_residual=%s beta=%s profile_quality=%s.",
            len(unranked_docnos),
            self.config.utility.tail.target_residual,
            self.config.utility.tail.conservatism_beta,
            self.config.utility.tail.profile_quality,
        )
        return_value = build_rbp_residual_tail_utilities(
            config=self.config,
            logger=self.logger,
            adapter=adapter,
            profiles=profiles,
            full_profile=full_profile,
            topics=topics,
            subset_docnos=subset_docnos,
            subset_doc_embeddings=subset_doc_embeddings,
            full_query_embeddings=full_query_embeddings,
            unranked_docnos=unranked_docnos,
            score_preservation_fn=self._compute_score_preservation,
        )
        return return_value.utility_table, return_value.report

    def _static_tail_report(self, unranked_docnos) -> dict:
        return {
            "tail_mode": str(self.config.utility.tail.mode),
            "tail_unranked_docs": int(len(unranked_docnos)),
            "tail_target_residual": float(self.config.utility.tail.target_residual),
            "tail_expected_relevance_per_doc": 0.0,
            "tail_uncertainty_per_doc": 0.0,
            "tail_profile_quality": str(self.config.utility.tail.profile_quality),
            "tail_profile_quality_mean_by_profile": {},
            "tail_beta": None if self.config.utility.tail.conservatism_beta is None else float(self.config.utility.tail.conservatism_beta),
            "tail_utility_min": 0.0,
            "tail_utility_mean": 0.0,
            "tail_utility_max": 0.0,
        }

UtilityEstimator = QueryLogUtilityEstimator

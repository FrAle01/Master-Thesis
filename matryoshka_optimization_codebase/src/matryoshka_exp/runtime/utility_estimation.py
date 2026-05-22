from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..legacy.score_metrics import (
    compute_pointwise_utility,
    hybrid_score_margin_utility,
    relative_margin_utility,
)
from .relevance.base import RELEVANCE_COLUMNS
from .relevance.cross_encoder_estimator import cross_encoder_rescore, estimate_from_cross_scores
from .relevance.fusion import merge_refined_scores, select_uncertain_high_impact_pairs
from .relevance.qrels_adjustment import apply_qrels_hard_override, apply_qrels_relevant_only_override
from .relevance.weak_estimators import (
    build_dense_candidates,
    build_dense_candidates_with_pyterrier_dr,
    combine_hybrid_candidates,
    estimate_from_candidates,
)


class UtilityEstimator:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.last_report = {}

    def _compute_score_preservation(
        self,
        *,
        full_scores: np.ndarray,
        reduced_scores: np.ndarray,
    ) -> np.ndarray:
        metric = str(self.config.utility.metric)
        epsilon = float(self.config.utility.epsilon)
        alpha = float(self.config.utility.alpha)
        m_neg = max(1, int(self.config.utility.margin_negatives))

        if metric in {"relative_score_dissimilarity", "absolute_score_utility", "squared_score_utility"}:
            return np.asarray(
                compute_pointwise_utility(
                    metric,
                    full_scores,
                    reduced_scores,
                    epsilon=epsilon,
                    alpha=alpha,
                ),
                dtype=float,
            )

        if full_scores.size <= 1:
            # Margin-based metrics are undefined with one candidate; keep neutral preservation.
            return np.ones_like(full_scores, dtype=float)

        n = int(full_scores.shape[0])
        full_neg = np.zeros(n, dtype=float)
        reduced_neg = np.zeros(n, dtype=float)
        for i in range(n):
            full_others = np.delete(full_scores, i)
            reduced_others = np.delete(reduced_scores, i)
            k = min(m_neg, full_others.size)
            full_neg[i] = float(np.mean(np.sort(full_others)[-k:]))
            reduced_neg[i] = float(np.mean(np.sort(reduced_others)[-k:]))

        if metric == "relative_margin_utility":
            return np.asarray(
                relative_margin_utility(
                    full_scores,
                    full_neg,
                    reduced_scores,
                    reduced_neg,
                    epsilon=epsilon,
                ),
                dtype=float,
            )
        if metric == "hybrid_score_margin_utility":
            return np.asarray(
                hybrid_score_margin_utility(
                    full_scores,
                    reduced_scores,
                    full_neg,
                    reduced_neg,
                    alpha=alpha,
                    epsilon=epsilon,
                ),
                dtype=float,
            )
        raise ValueError(f"Unsupported utility metric: {metric}")

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
        doc_index = {str(docno): i for i, docno in enumerate(subset_docnos)}
        pair_relevance_df = self._estimate_pair_relevance(
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
                columns=[
                    "qid",
                    "docno",
                    "profile",
                    "full_score",
                    "reduced_score",
                    "score_preservation",
                    "utility",
                    *RELEVANCE_COLUMNS[2:],
                ]
            )
            utility_table_df = self._default_utility_table(subset_docnos, profiles, full_profile)
            self.last_report = {
                "relevance_mode": self.config.utility.relevance.mode,
                "coverage_pairs": 0,
                "num_docs": len(subset_docnos),
                "qrel_overrides": 0,
                "default_docs": int(len(subset_docnos)),
            }
            return utility_pairs_df, utility_table_df
        else:
            self.logger.info("Estimating utilities for %d relevance pairs.", len(pair_relevance_df))

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
        sampled_rows = []
        per_doc_profile_utilities = defaultdict(list)
        qid_to_position = {str(qid): i for i, qid in enumerate(topics["qid"].astype(str).tolist())}

        grouped_pairs = list(pair_relevance_df.groupby("qid"))
        for qid, group in tqdm(grouped_pairs, desc="Estimating per-doc utilities", disable=not self.config.execution.verbose):
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

            for profile in profiles:
                q_reduced = q_full[:, : profile.dimension]
                doc_reduced = doc_full[:, : profile.dimension]
                reduced_scores = adapter.similarity(q_reduced, doc_reduced).squeeze(0)

                full_scores_np = full_scores.detach().cpu().numpy()
                reduced_scores_np = reduced_scores.detach().cpu().numpy()
                preservation_np = self._compute_score_preservation(
                    full_scores=full_scores_np,
                    reduced_scores=reduced_scores_np,
                )

                for docno, s_full, s_red, score_pres in zip(valid_docnos, full_scores_np, reduced_scores_np, preservation_np):
                    rel = float(rel_map.get(docno, 0.0))
                    utility = float(rel * score_pres)
                    sampled_rows.append(
                        {
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
                    )
                    per_doc_profile_utilities[(docno, profile.name)].append(utility)

        utility_pairs_df = pd.DataFrame(sampled_rows, columns=sample_columns)
        utility_table_df, default_count = self._aggregate_table(subset_docnos, profiles, full_profile, per_doc_profile_utilities)

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
        }
        self.logger.info("Relevance estimation report: %s", self.last_report)
        return utility_pairs_df, utility_table_df

    def _default_utility_table(self, subset_docnos, profiles, full_profile):
        aggregated_rows = []
        for docno in subset_docnos:
            for profile in profiles:
                default_utility = self._default_utility_for_profile(profile.name, full_profile.name)
                aggregated_rows.append({"docno": str(docno), "profile": profile.name, "utility": default_utility})
        return pd.DataFrame(aggregated_rows, columns=["docno", "profile", "utility"])

    def _aggregate_table(self, subset_docnos, profiles, full_profile, per_doc_profile_utilities):
        aggregated_rows = []
        default_count = 0
        for docno in [str(d) for d in subset_docnos]:
            has_any = False
            for profile in profiles:
                values = per_doc_profile_utilities.get((docno, profile.name), [])
                if values:
                    agg = float(np.sum(values))
                    has_any = True
                else:
                    agg = self._default_utility_for_profile(profile.name, full_profile.name)
                aggregated_rows.append({"docno": docno, "profile": profile.name, "utility": agg})
            if not has_any:
                default_count += 1
        return pd.DataFrame(aggregated_rows, columns=["docno", "profile", "utility"]), default_count

    def _default_utility_for_profile(self, profile_name: str, full_profile_name: str) -> float:
        preferred_profile = self.config.utility.default_utility_profile_name or full_profile_name
        # TODO: extend this to support custom fallback functions for never-retrieved documents.
        return 1.0 if profile_name == preferred_profile else 0.0

    def _estimate_pair_relevance(
        self,
        *,
        loader,
        adapter,
        topics,
        qrels,
        subset_docnos,
        subset_doc_embeddings,
        full_query_embeddings,
        corpus_metadata,
    ) -> pd.DataFrame:
        mode = str(self.config.utility.relevance.mode)
        weak_source = str(self.config.utility.relevance.weak_source)
        calibration_mode = str(self.config.utility.relevance.calibration)
        top_k = int(self.config.utility.relevance.top_k_candidates)
        rerank_k = int(self.config.utility.relevance.rerank_k)
        self.logger.info("Estimating relevance with mode=%s weak_source=%s calibration=%s", mode, weak_source, calibration_mode)
        self.logger.info("Estimating relevance pairs for %d candidate documents on %s queries.", len(subset_docnos), len(topics))

        bm25_candidates = pd.DataFrame(columns=["qid", "docno", "score", "rank"])
        dense_candidates = pd.DataFrame(columns=["qid", "docno", "score", "rank"])

        if mode in {"weak", "hybrid"} and weak_source in {"bm25", "hybrid_rerank"}:
            bm25_candidates = loader.build_bm25_candidates(self.config.retrieval, topics)
            bm25_candidates = bm25_candidates[bm25_candidates["docno"].astype(str).isin(set(map(str, subset_docnos)))].copy()
            bm25_candidates = bm25_candidates.groupby("qid", sort=False).head(top_k)
            self.logger.info("BM25 candidate generation complete. Candidates: %d", len(bm25_candidates))

        if mode in {"weak", "hybrid"} and weak_source in {"dense", "hybrid_rerank"}:
            dense_candidates = build_dense_candidates_with_pyterrier_dr(
                topics=topics,
                topic_column=self.config.data.topic_column,
                subset_docnos=subset_docnos,
                subset_doc_embeddings=subset_doc_embeddings,
                full_query_embeddings=full_query_embeddings,
                top_k=top_k,
                index_path=self.config.output_path / "utility_relevance_dense_index",
            )
            if dense_candidates is None:
                self.logger.info("pyterrier_dr not available for utility dense candidates; falling back to exact dense scoring.")
                dense_candidates = build_dense_candidates(
                    topics=topics,
                    subset_docnos=subset_docnos,
                    subset_doc_embeddings=subset_doc_embeddings,
                    full_query_embeddings=full_query_embeddings,
                    top_k=top_k,
                    adapter=adapter,
                )
            self.logger.info("Dense candidate generation complete. Candidates: %d", len(dense_candidates))

        self.logger.info("Candidate generation complete. BM25 candidates: %d Dense candidates: %d", len(bm25_candidates), len(dense_candidates))
        if mode == "weak":
            if weak_source == "bm25":
                pair_df = estimate_from_candidates(bm25_candidates, source_mode="weak_bm25", calibration_mode=calibration_mode)
            elif weak_source == "dense":
                pair_df = estimate_from_candidates(dense_candidates, source_mode="weak_dense", calibration_mode=calibration_mode)
            else:
                merged = combine_hybrid_candidates(bm25_candidates, dense_candidates, rerank_k=rerank_k)
                pair_df = estimate_from_candidates(merged, source_mode="weak_hybrid_rerank", calibration_mode=calibration_mode)

        elif mode == "model":
            # Prefer PyTerrier BM25 candidates as the pipeline retrieval source for model judging.
            seed_candidates = loader.build_bm25_candidates(self.config.retrieval, topics)
            seed_candidates = seed_candidates[seed_candidates["docno"].astype(str).isin(set(map(str, subset_docnos)))].copy()
            seed_candidates = seed_candidates.groupby("qid", sort=False).head(top_k)
            topic_df = topics.loc[:, ["qid", self.config.data.topic_column]].rename(columns={self.config.data.topic_column: "query"})
            doc_text = {
                str(r["docno"]): str(r["text"])
                for r in corpus_metadata.loc[:, ["docno", "text"]].to_dict(orient="records")
            }
            rescored = cross_encoder_rescore(
                candidates=seed_candidates,
                topics=topic_df,
                doc_text_by_docno=doc_text,
                model_name=str(self.config.utility.relevance.cross_encoder_model_name),
            )
            pair_df = estimate_from_cross_scores(rescored, source_mode="model_cross_encoder", calibration_mode=calibration_mode)

        elif mode == "hybrid":
            if weak_source == "bm25":
                weak_pairs = estimate_from_candidates(bm25_candidates, source_mode="hybrid_weak_bm25", calibration_mode=calibration_mode)
                seed_candidates = bm25_candidates
            elif weak_source == "dense":
                weak_pairs = estimate_from_candidates(dense_candidates, source_mode="hybrid_weak_dense", calibration_mode=calibration_mode)
                seed_candidates = dense_candidates
            else:
                merged = combine_hybrid_candidates(bm25_candidates, dense_candidates, rerank_k=rerank_k)
                weak_pairs = estimate_from_candidates(merged, source_mode="hybrid_weak_hybrid_rerank", calibration_mode=calibration_mode)
                seed_candidates = merged

            uncertain = select_uncertain_high_impact_pairs(
                weak_pairs,
                fraction=float(self.config.utility.relevance.high_impact_fraction),
            )
            refine_keys = set((str(r["qid"]), str(r["docno"])) for r in uncertain.to_dict(orient="records"))
            refine_candidates = seed_candidates[
                seed_candidates.apply(lambda x: (str(x["qid"]), str(x["docno"])) in refine_keys, axis=1)
            ].copy()

            topic_df = topics.loc[:, ["qid", self.config.data.topic_column]].rename(columns={self.config.data.topic_column: "query"})
            doc_text = {
                str(r["docno"]): str(r["text"])
                for r in corpus_metadata.loc[:, ["docno", "text"]].to_dict(orient="records")
            }
            rescored = cross_encoder_rescore(
                candidates=refine_candidates,
                topics=topic_df,
                doc_text_by_docno=doc_text,
                model_name=str(self.config.utility.relevance.cross_encoder_model_name),
            )
            refined = estimate_from_cross_scores(rescored, source_mode="hybrid_refined_cross_encoder", calibration_mode=calibration_mode)
            pair_df = merge_refined_scores(weak_pairs, refined, source_mode="hybrid_refined_cross_encoder")
        else:
            raise ValueError(f"Unsupported utility.relevance.mode: {mode}")

        label_column = self._resolve_qrels_label_column(qrels)
        if str(self.config.utility.relevance.qrel_adjustment) == "relevant_only_to_one":
            pair_df, n_overrides = apply_qrels_relevant_only_override(
                pair_df,
                qrels,
                label_column=label_column,
                relevance_threshold=float(self.config.utility.relevance_threshold),
            )
        else:
            pair_df, n_overrides = apply_qrels_hard_override(
                pair_df,
                qrels,
                label_column=label_column,
                scale_cfg=self.config.utility.relevance.scale,
            )
        self.logger.info(
            "Relevance estimation complete. mode=%s pairs=%s overrides=%s",
            mode,
            len(pair_df),
            n_overrides,
        )
        return pair_df.loc[:, RELEVANCE_COLUMNS].copy()

    def _resolve_qrels_label_column(self, qrels: pd.DataFrame) -> str:
        column = str(self.config.data.qrels_label_column)
        if column not in qrels.columns:
            raise ValueError(
                f"Qrels label column `{column}` not found. Available columns: {sorted(qrels.columns.tolist())}"
            )
        return column

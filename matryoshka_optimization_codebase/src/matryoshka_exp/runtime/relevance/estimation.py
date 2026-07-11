from __future__ import annotations

import pandas as pd

from .base import RELEVANCE_COLUMNS
from .cross_encoder_estimator import cross_encoder_rescore, estimate_from_cross_scores
from .fusion import merge_refined_scores, select_uncertain_high_impact_pairs
from .qrels_adjustment import apply_qrels_hard_override, apply_qrels_relevant_only_override
from .weak_estimators import (
    build_dense_candidates,
    build_dense_candidates_with_pyterrier_dr,
    combine_hybrid_candidates,
    estimate_from_candidates,
)


class PairRelevanceEstimator:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def estimate(
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

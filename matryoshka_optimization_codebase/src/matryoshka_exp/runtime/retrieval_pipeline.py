from __future__ import annotations

import pandas as pd
import torch

from ..retrieval.dense import DenseGroupedRetriever
from ..retrieval.materialization import materialize_grouped_corpus
from ..results.persistence import save_df
from .device_policy import assert_retrieval_fits_vram, resolve_retrieval_device


class RetrievalPipeline:
    def __init__(self, config, output_dir, logger):
        self.config = config
        self.output_dir = output_dir
        self.logger = logger

    def run(
        self,
        *,
        loader,
        adapter,
        profile_by_name,
        full_profile,
        docnos,
        full_doc_embeddings,
        assignments,
        topics,
        full_query_embeddings,
    ):
        retrieval_device = resolve_retrieval_device(self.config)

        full_assignments = pd.DataFrame(
            {
                "docno": docnos,
                "profile": [full_profile.name] * len(docnos),
                "utility": [1.0] * len(docnos),
                "cost_bytes": [full_profile.cost_bytes] * len(docnos),
            }
        )
        assert_retrieval_fits_vram(
            self.config,
            max(
                int(assignments["cost_bytes"].sum()),
                int(full_assignments["cost_bytes"].sum()),
            ),
            retrieval_device=retrieval_device,
            logger=self.logger,
        )

        retriever = DenseGroupedRetriever(
            adapter,
            profile_by_name,
            self.config.model.similarity,
            self.config.retrieval.top_k,
            device=retrieval_device,
        )
        query_ids = topics["qid"].astype(str).tolist()
        query_emb_by_id = {qid: emb.unsqueeze(0) for qid, emb in zip(query_ids, full_query_embeddings)}

        if self.config.retrieval.mode == "dense_exact":
            full_corpus, _ = materialize_grouped_corpus(
                docnos,
                full_doc_embeddings,
                full_assignments,
                profile_by_name,
                target_device="cpu",
            )
            if retrieval_device != "cpu":
                self._move_corpus_to_device(full_corpus, retrieval_device)
            self._log_materialized_profiles("full", full_corpus)
            full_run = self._search_exact_with_oom_context(
                retriever=retriever,
                query_ids=query_ids,
                full_query_embeddings=full_query_embeddings,
                corpus=full_corpus,
                retrieval_device=retrieval_device,
                run_label="full",
            )
            if self.config.execution.save_runs:
                save_df(full_run, self.output_dir / "full_run.parquet")
            del full_corpus
            self._empty_cache_if_needed(retrieval_device)

            opt_corpus, _ = materialize_grouped_corpus(
                docnos,
                full_doc_embeddings,
                assignments,
                profile_by_name,
                target_device="cpu",
            )
            if retrieval_device != "cpu":
                self._move_corpus_to_device(opt_corpus, retrieval_device)
            self._log_materialized_profiles("optimized", opt_corpus)
            opt_run = self._search_exact_with_oom_context(
                retriever=retriever,
                query_ids=query_ids,
                full_query_embeddings=full_query_embeddings,
                corpus=opt_corpus,
                retrieval_device=retrieval_device,
                run_label="optimized",
            )
            del opt_corpus
            self._empty_cache_if_needed(retrieval_device)
            return full_run, opt_run

        candidates = loader.build_bm25_candidates(self.config.retrieval, topics)
        save_df(candidates, self.output_dir / "bm25_candidates.parquet")
        _, full_lookup = materialize_grouped_corpus(
            docnos,
            full_doc_embeddings,
            full_assignments,
            profile_by_name,
            target_device=retrieval_device,
        )
        full_run = retriever.rerank_candidates(
            candidates,
            query_emb_by_id,
            full_lookup,
            verbose=self.config.execution.verbose,
        )
        if self.config.execution.save_runs:
            save_df(full_run, self.output_dir / "full_run.parquet")
        del full_lookup
        self._empty_cache_if_needed(retrieval_device)

        _, opt_lookup = materialize_grouped_corpus(
            docnos,
            full_doc_embeddings,
            assignments,
            profile_by_name,
            target_device=retrieval_device,
        )
        opt_run = retriever.rerank_candidates(
            candidates,
            query_emb_by_id,
            opt_lookup,
            verbose=self.config.execution.verbose,
        )
        del opt_lookup
        self._empty_cache_if_needed(retrieval_device)
        return full_run, opt_run

    @staticmethod
    def _empty_cache_if_needed(retrieval_device: str) -> None:
        if retrieval_device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _move_corpus_to_device(self, corpus, device: str) -> None:
        for profile_name, matrix in corpus.embeddings_by_profile.items():
            if str(matrix.device) == device:
                continue
            corpus.embeddings_by_profile[profile_name] = matrix.to(device)

    def _log_materialized_profiles(self, run_label: str, corpus) -> None:
        profiles = sorted(corpus.embeddings_by_profile.keys())
        self.logger.info(
            "Dense retrieval [%s]: materialized profiles=%s",
            run_label,
            profiles,
        )
        for profile_name in profiles:
            matrix = corpus.embeddings_by_profile[profile_name]
            doc_count = len(corpus.docnos_by_profile.get(profile_name, []))
            self.logger.info(
                "Dense retrieval [%s]: profile=%s docs=%d stacked_shape=%s dtype=%s device=%s",
                run_label,
                profile_name,
                doc_count,
                tuple(matrix.shape),
                matrix.dtype,
                matrix.device,
            )

    def _search_exact_with_oom_context(
        self,
        *,
        retriever,
        query_ids,
        full_query_embeddings,
        corpus,
        retrieval_device: str,
        run_label: str,
    ):
        try:
            return retriever.search_exact(
                query_ids,
                full_query_embeddings,
                corpus,
                verbose=self.config.execution.verbose,
            )
        except torch.OutOfMemoryError as exc:
            raise RuntimeError(
                "CUDA OOM during dense exact retrieval "
                f"({run_label} run). Consider one of: "
                "`execution.retrieval_device=cpu`, "
                "`retrieval.mode=pyterrier_candidates`, "
                "or reducing corpus size/profile dimensions."
            ) from exc

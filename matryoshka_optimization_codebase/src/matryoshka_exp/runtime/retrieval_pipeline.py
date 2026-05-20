from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

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
    ) -> Dict[str, pd.DataFrame]:
        retrieval_device = resolve_retrieval_device(self.config)
        runs_assignments = self._build_run_assignments(
            docnos=docnos,
            full_profile=full_profile,
            profile_by_name=profile_by_name,
            optimized_assignments=assignments,
        )

        max_required_bytes = max(int(df["cost_bytes"].sum()) for df in runs_assignments.values())
        assert_retrieval_fits_vram(
            self.config,
            max_required_bytes,
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

        runs: Dict[str, pd.DataFrame] = OrderedDict()
        if self.config.retrieval.include_bm25_baseline:
            baseline = loader.build_bm25_candidates(self.config.retrieval, topics)
            runs["bm25_baseline"] = baseline
            if self.config.execution.save_runs:
                save_df(baseline, self.output_dir / "bm25_baseline_run.parquet")

        if self.config.retrieval.mode == "dense_exact":
            for run_name, run_assignments in runs_assignments.items():
                corpus, _ = materialize_grouped_corpus(
                    docnos,
                    full_doc_embeddings,
                    run_assignments,
                    profile_by_name,
                    target_device="cpu",
                )
                if retrieval_device != "cpu":
                    self._move_corpus_to_device(corpus, retrieval_device)
                self._log_materialized_profiles(run_name, corpus)
                run_df = self._search_exact_with_oom_context(
                    retriever=retriever,
                    query_ids=query_ids,
                    full_query_embeddings=full_query_embeddings,
                    corpus=corpus,
                    retrieval_device=retrieval_device,
                    run_label=run_name,
                )
                runs[run_name] = run_df
                if self.config.execution.save_runs:
                    save_df(run_df, self.output_dir / f"{run_name}.parquet")
                del corpus
                self._empty_cache_if_needed(retrieval_device)
            return runs

        candidates = loader.build_bm25_candidates(self.config.retrieval, topics)
        if self.config.execution.save_runs:
            save_df(candidates, self.output_dir / "bm25_candidates.parquet")

        for run_name, run_assignments in runs_assignments.items():
            _, lookup = materialize_grouped_corpus(
                docnos,
                full_doc_embeddings,
                run_assignments,
                profile_by_name,
                target_device=retrieval_device,
            )
            run_df = retriever.rerank_candidates(
                candidates,
                query_emb_by_id,
                lookup,
                verbose=self.config.execution.verbose,
            )
            runs[run_name] = run_df
            if self.config.execution.save_runs:
                save_df(run_df, self.output_dir / f"{run_name}.parquet")
            del lookup
            self._empty_cache_if_needed(retrieval_device)
        return runs

    def _build_run_assignments(
        self,
        *,
        docnos: List[str],
        full_profile,
        profile_by_name,
        optimized_assignments: pd.DataFrame,
    ) -> "OrderedDict[str, pd.DataFrame]":
        runs_assignments: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
        runs_assignments["full_embedding"] = self._build_uniform_assignments(docnos, full_profile.name, profile_by_name)

        non_full_profiles = sorted(
            (profile for profile in profile_by_name.values() if profile.name != full_profile.name),
            key=lambda p: (int(p.dimension), str(p.name)),
            reverse=True,
        )
        for profile in non_full_profiles:
            run_name = f"profile_{profile.name}"
            runs_assignments[run_name] = self._build_uniform_assignments(docnos, profile.name, profile_by_name)

        runs_assignments["optimized_embedding"] = optimized_assignments.copy()
        return runs_assignments

    @staticmethod
    def _build_uniform_assignments(docnos: List[str], profile_name: str, profile_by_name) -> pd.DataFrame:
        profile = profile_by_name[profile_name]
        return pd.DataFrame(
            {
                "docno": docnos,
                "profile": [profile_name] * len(docnos),
                "utility": [1.0] * len(docnos),
                "cost_bytes": [profile.cost_bytes] * len(docnos),
            }
        )

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

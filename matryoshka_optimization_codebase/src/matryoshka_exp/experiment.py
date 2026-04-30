from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import hashlib
import re
from typing import Dict

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig, save_config_snapshot
from .data.pyterrier_utils import PyTerrierLoader
from .logging_utils import configure_logging
from .models.factory import create_encoder
from .optimization.errors import InfeasibleOptimizationError
from .optimization.lagrangian import LagrangianProfileOptimizer
from .retrieval.dense import DenseGroupedRetriever
from .retrieval.embedding_store import load_full_embeddings_from_hf, save_full_embeddings_to_hf
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

        query_texts = topics[self.config.data.topic_column].astype(str).tolist()
        query_ids = topics["qid"].astype(str).tolist()
        full_query_embeddings = adapter.embed_texts(
            query_texts,
            full_profile,
            prompt_name="query",
            batch_size=self.config.execution.query_batch_size,
        )
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
        corpus_records = list(loader.iter_corpus())
        docnos = [record.docno for record in corpus_records]
        metadata = pd.DataFrame(
            [{"docno": record.docno, "text": record.text} for record in corpus_records],
            columns=["docno", "text"],
        )
        full_doc_embeddings = self._load_or_compute_full_doc_embeddings(
            adapter=adapter,
            full_profile=full_profile,
            docnos=docnos,
            corpus_records=corpus_records,
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
            lambda_low=self.config.optimization.lambda_low,
            lambda_high=self.config.optimization.lambda_high,
        )
        opt_result = optimizer.solve(utility_table_df)
        assignments = opt_result.assignments

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

        retrieval_device = self._resolve_retrieval_device()

        full_assignments = pd.DataFrame(
            {
                "docno": docnos,
                "profile": [full_profile.name] * len(docnos),
                "utility": [1.0] * len(docnos),
                "cost_bytes": [full_profile.cost_bytes] * len(docnos),
            }
        )
        self._assert_retrieval_fits_vram(
            max(
                int(assignments["cost_bytes"].sum()),
                int(full_assignments["cost_bytes"].sum()),
            ),
            retrieval_device=retrieval_device,
        )
        full_corpus, full_lookup = materialize_grouped_corpus(
            docnos,
            full_doc_embeddings,
            full_assignments,
            profile_by_name,
            target_device=retrieval_device,
        )
        opt_corpus, opt_lookup = materialize_grouped_corpus(
            docnos,
            full_doc_embeddings,
            assignments,
            profile_by_name,
            target_device=retrieval_device,
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

    def _load_or_compute_full_doc_embeddings(self, *, adapter, full_profile, docnos, corpus_records):
        source = self.config.data.full_embeddings_source
        target_repo_id = self._resolve_model_specific_embeddings_repo_id()

        if source in {"auto", "hf_dataset"}:
            try:
                loaded_docnos, loaded_embeddings = load_full_embeddings_from_hf(
                    repo_id=target_repo_id,
                    split=self.config.data.hf_embeddings_split,
                    docno_column=self.config.data.hf_embeddings_docno_column,
                    vector_column=self.config.data.hf_embeddings_vector_column,
                    expected_docnos=docnos,
                    expected_dimension=full_profile.dimension,
                    target_dtype=self._target_dtype_from_execution(),
                )
                if loaded_docnos != [str(docno) for docno in docnos]:
                    raise ValueError("Loaded Hugging Face embeddings docno order does not match the current corpus.")
                self.logger.info(
                    "Loaded %s full document embeddings from Hugging Face dataset `%s` (split `%s`).",
                    len(loaded_docnos),
                    target_repo_id,
                    self.config.data.hf_embeddings_split,
                )
                return loaded_embeddings
            except (ValueError, OSError, RuntimeError) as exc:
                self.logger.warning(
                    "Failed to load full embeddings from Hugging Face (source=%s, repo=%s, split=%s): %s. "
                    "Falling back to local encoding.",
                    source,
                    target_repo_id,
                    self.config.data.hf_embeddings_split,
                    exc,
                )

        _docnos, computed_embeddings, _ = encode_full_corpus(
            iter(corpus_records),
            adapter,
            full_profile,
            batch_size=self.config.execution.doc_batch_size,
            prompt_name="document",
            verbose=self.config.execution.verbose,
        )

        if target_repo_id:
            try:
                save_full_embeddings_to_hf(
                    repo_id=target_repo_id,
                    split=self.config.data.hf_embeddings_split,
                    docnos=docnos,
                    embeddings=computed_embeddings,
                    docno_column=self.config.data.hf_embeddings_docno_column,
                    vector_column=self.config.data.hf_embeddings_vector_column,
                )
                self.logger.info(
                    "Saved %s full document embeddings to Hugging Face dataset `%s` (split `%s`).",
                    len(docnos),
                    target_repo_id,
                    self.config.data.hf_embeddings_split,
                )
            except (ValueError, OSError, RuntimeError) as exc:
                self.logger.warning(
                    "Failed to save computed full embeddings to Hugging Face repo `%s` (split `%s`): %s",
                    target_repo_id,
                    self.config.data.hf_embeddings_split,
                    exc,
                )
        else:
            self.logger.warning(
                "Skipping Hugging Face embedding persistence because `data.hf_embeddings_repo_id` is not configured."
            )

        return computed_embeddings

    def _resolve_model_specific_embeddings_repo_id(self) -> str | None:
        base_repo_id = self.config.data.hf_embeddings_repo_id
        if not base_repo_id:
            return None

        signature_parts = [str(self.config.model.model_name_or_path)]
        if self.config.model.adapter_type:
            signature_parts.extend(
                [
                    f"adapter-{self.config.model.adapter_type}",
                    str(self.config.model.adapter_name),
                    str(self.config.model.adapter_path or ""),
                ]
            )
        signature_raw = "__".join(signature_parts)
        model_slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", signature_raw.lower()).strip(".-_")
        if not model_slug:
            model_slug = "model"

        max_slug_len = 64
        if len(model_slug) > max_slug_len:
            digest = hashlib.sha1(model_slug.encode("utf-8")).hexdigest()[:10]
            model_slug = f"{model_slug[: max_slug_len - 11]}-{digest}"

        if "/" in base_repo_id:
            namespace, repo_name = base_repo_id.split("/", 1)
            return f"{namespace}/{repo_name}--{model_slug}"
        return f"{base_repo_id}--{model_slug}"

    def _target_dtype_from_execution(self) -> torch.dtype:
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(self.config.execution.dtype, torch.float32)

    def _resolve_retrieval_device(self) -> str:
        requested = self.config.execution.retrieval_device
        if not self.config.retrieval.use_torch_gpu_exact:
            return "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "execution.retrieval_device='cuda' but CUDA is not available. "
                "Verify the pinned cu121 PyTorch installation and run the GPU preflight checks."
            )
        return requested

    def _assert_retrieval_fits_vram(self, required_bytes: int, *, retrieval_device: str) -> None:
        if retrieval_device != "cuda":
            return
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA retrieval requested but CUDA is not available. "
                "Verify the pinned cu121 PyTorch installation and run the GPU preflight checks."
            )

        free_bytes, total_bytes = torch.cuda.mem_get_info()
        usable_bytes = int(total_bytes * float(self.config.execution.retrieval_vram_utilization_limit))

        if required_bytes > usable_bytes or required_bytes > free_bytes:
            raise RuntimeError(
                "Optimized corpus does not fit configured VRAM limits for GPU retrieval: "
                f"required_bytes={required_bytes}, free_bytes={free_bytes}, total_bytes={total_bytes}, "
                f"utilization_limit={self.config.execution.retrieval_vram_utilization_limit}"
            )

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
            full_scores = adapter.similarity(q_full, doc_full).squeeze(0)

            neg_docnos = self._sample_margin_negatives(str(qid), qrels, subset_docnos, group_docnos)
            neg_indices = [doc_index[d] for d in neg_docnos if d in doc_index]
            neg_scores_full = None
            if neg_indices:
                neg_doc_full = subset_doc_embeddings[neg_indices]
                neg_scores_full = adapter.similarity(q_full, neg_doc_full).squeeze(0)

            for profile in profiles:
                q_reduced = q_full[:, : profile.dimension]
                doc_reduced = doc_full[:, : profile.dimension]
                reduced_scores = adapter.similarity(q_reduced, doc_reduced).squeeze(0)
                utility = self._compute_utility_tensor(
                    full_scores,
                    reduced_scores,
                    q_reduced,
                    profile.dimension,
                    neg_indices,
                    subset_doc_embeddings,
                    neg_scores_full,
                    adapter,
                )

                full_scores_np = full_scores.detach().cpu().numpy()
                reduced_scores_np = reduced_scores.detach().cpu().numpy()
                utility_np = utility.detach().cpu().numpy()

                for docno, s_full, s_red, u in zip(group_docnos, full_scores_np, reduced_scores_np, utility_np):
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
                if values:
                    agg = float(np.mean(values))
                else:
                    agg = 1.0 if profile.name == full_profile.name else 0.0
                aggregated_rows.append({"docno": docno, "profile": profile.name, "utility": agg})
        utility_table_df = pd.DataFrame(aggregated_rows, columns=["docno", "profile", "utility"])
        return utility_pairs_df, utility_table_df

    def _compute_utility_tensor(
        self,
        full_scores: torch.Tensor,
        reduced_scores: torch.Tensor,
        q_reduced: torch.Tensor,
        profile_dim: int,
        neg_indices,
        subset_doc_embeddings: torch.Tensor,
        neg_scores_full: torch.Tensor | None,
        adapter,
    ) -> torch.Tensor:
        eps = float(self.config.utility.epsilon)
        metric = self.config.utility.metric

        if metric == "relative_score_dissimilarity":
            denom = torch.maximum(full_scores.abs(), torch.tensor(eps, device=full_scores.device, dtype=full_scores.dtype))
            loss = (full_scores - reduced_scores).abs() / denom
            return 1.0 - torch.clamp(loss, 0.0, 1.0)

        if metric == "absolute_score_utility":
            return 1.0 / (1.0 + (full_scores - reduced_scores).abs())

        if metric == "squared_score_utility":
            return 1.0 / (1.0 + torch.square(full_scores - reduced_scores))

        if metric in {"relative_margin_utility", "hybrid_score_margin_utility"}:
            if not neg_indices or neg_scores_full is None or neg_scores_full.numel() == 0:
                score_u = self._compute_utility_tensor(
                    full_scores,
                    reduced_scores,
                    q_reduced,
                    profile_dim,
                    [],
                    subset_doc_embeddings,
                    None,
                    adapter,
                )
                if metric == "relative_margin_utility":
                    return torch.ones_like(full_scores)
                return score_u

            neg_doc_reduced = subset_doc_embeddings[neg_indices][:, :profile_dim]
            neg_scores_reduced = adapter.similarity(q_reduced, neg_doc_reduced).squeeze(0)

            if self.config.utility.aggregate != "mean":
                raise ValueError(f"Unsupported utility.aggregate: {self.config.utility.aggregate}")

            full_margin = full_scores.unsqueeze(1) - neg_scores_full.unsqueeze(0)
            reduced_margin = reduced_scores.unsqueeze(1) - neg_scores_reduced.unsqueeze(0)
            margin_denom = torch.maximum(full_margin.abs(), torch.tensor(eps, device=full_margin.device, dtype=full_margin.dtype))
            margin_loss = (full_margin - reduced_margin).abs() / margin_denom
            margin_u = 1.0 - torch.clamp(margin_loss, 0.0, 1.0)
            margin_u = margin_u.mean(dim=1)

            if metric == "relative_margin_utility":
                return margin_u

            score_u = self._compute_utility_tensor(
                full_scores,
                reduced_scores,
                q_reduced,
                profile_dim,
                [],
                subset_doc_embeddings,
                None,
                adapter,
            )
            alpha = float(self.config.utility.alpha)
            return alpha * score_u + (1.0 - alpha) * margin_u

        raise ValueError(f"Unsupported utility metric: {metric}")

    def _sample_margin_negatives(self, qid: str, qrels: pd.DataFrame, subset_docnos, positive_docnos):
        qrels_q = qrels[qrels["qid"].astype(str) == str(qid)]
        positive_set = set(qrels_q["docno"].astype(str).tolist())
        positive_set.update([str(d) for d in positive_docnos])
        pool = [str(d) for d in subset_docnos if str(d) not in positive_set]
        if not pool:
            return []

        digest = hashlib.sha1(str(qid).encode("utf-8")).hexdigest()
        qid_hash = int(digest[:8], 16)
        rng_seed = int(self.config.utility.seed + (qid_hash & 0xFFFF))
        rng = np.random.default_rng(rng_seed)
        n = min(int(self.config.utility.margin_negatives), len(pool))
        picks = rng.choice(pool, size=n, replace=False)
        return [str(x) for x in picks]

    def _build_score_sampling_frame(self, topics, qrels, subset_docnos):
        subset_docnos = [str(d) for d in subset_docnos]
        max_pairs = self.config.utility.sample_pairs_per_query
        qrels = qrels.copy()
        qrels["qid"] = qrels["qid"].astype(str)
        qrels["docno"] = qrels["docno"].astype(str)

        subset_docno_set = set(subset_docnos)
        rng = np.random.default_rng(int(self.config.utility.seed))
        sampled = []
        for qid in topics["qid"].astype(str).tolist():
            positives = qrels[qrels["qid"] == qid]["docno"].tolist()
            positives = [d for d in positives if d in subset_docno_set]
            if not positives:
                continue
            positive_set = set(positives)
            negative_pool = [d for d in subset_docnos if d not in positive_set]
            n_neg = max(0, max_pairs - len(positives))
            if n_neg > 0 and negative_pool:
                take = min(n_neg, len(negative_pool))
                negatives = [str(x) for x in rng.choice(negative_pool, size=take, replace=False)]
            else:
                negatives = []
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

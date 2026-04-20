from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import ExperimentConfig, save_config_snapshot
from .data.pyterrier_utils import CorpusRecord, PyTerrierLoader
from .logging_utils import configure_logging
from .metrics.score_metrics import (
    compute_pointwise_utility,
    hybrid_score_margin_utility,
    relative_margin_utility,
)
from .models.factory import create_encoder
from .models.base import RepresentationProfile
from .optimization.lagrangian import LagrangianProfileOptimizer
from .retrieval.dense import DenseGroupedRetriever, EmbeddedCorpus
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
        model_dir = finetuner.run()
        self.config.model.model_name_or_path = str(model_dir)
        self.logger.info("Updated experiment to use fine-tuned model at %s", model_dir)

    def run(self) -> Dict:
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
        query_emb_by_id = {qid: emb.unsqueeze(0) for qid, emb in zip(query_ids, full_query_embeddings)}

        if self.config.optimization.mode == "streaming":
            payload = self._run_streaming(loader, adapter, profiles, profile_by_name, full_profile, topics, qrels, full_query_embeddings)
        else:
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

    def _run_streaming(self, loader, adapter, profiles, profile_by_name, full_profile, topics, qrels, full_query_embeddings):
        self.logger.info("Running streaming mode with online profile assignment.")
        budget_bytes = self.config.budget_bytes_resolved()
        streaming_reestimate_every_docs = self.config.optimization.streaming_reestimate_every_docs

        all_docnos: List[str] = []
        metadata_rows: List[Dict] = []
        utility_pairs_parts: List[pd.DataFrame] = []
        utility_rows: List[Dict] = []
        assignment_rows: List[Dict] = []

        full_docnos_by_profile = defaultdict(list)
        full_embs_by_profile = defaultdict(list)
        opt_docnos_by_profile = defaultdict(list)
        opt_embs_by_profile = defaultdict(list)
        full_lookup = {}
        opt_lookup = {}

        optimizer = LagrangianProfileOptimizer(
            profiles=profiles,
            budget_bytes=budget_bytes,
            max_iter=self.config.optimization.max_iter,
            tolerance=self.config.optimization.tolerance,
        )

        pending_records: List[CorpusRecord] = []
        current_lambda = 0.0
        processed_docs = 0
        warmup_done = False

        def flush_batch(records: List[CorpusRecord], lambda_value: float):
            nonlocal processed_docs
            if not records:
                return lambda_value

            batch_docnos = [r.docno for r in records]
            batch_texts = [r.text for r in records]
            batch_full_emb = adapter.embed_texts(
                batch_texts,
                full_profile,
                prompt_name="document",
                batch_size=self.config.execution.doc_batch_size,
            ).cpu()

            batch_pairs_df, batch_util_df = self._estimate_document_utilities_for_subset(
                adapter=adapter,
                profiles=profiles,
                full_profile=full_profile,
                topics=topics,
                qrels=qrels,
                subset_docnos=batch_docnos,
                subset_doc_embeddings=batch_full_emb,
                full_query_embeddings=full_query_embeddings,
            )
            utility_pairs_parts.append(batch_pairs_df)
            utility_rows.extend(batch_util_df.to_dict(orient="records"))

            # Re-estimate lambda periodically using the utility accumulated so far.
            if (not warmup_done) or (processed_docs > 0 and processed_docs % streaming_reestimate_every_docs == 0):
                current_utility_df = pd.DataFrame(utility_rows)
                if not current_utility_df.empty:
                    partial_result = optimizer.solve(current_utility_df)
                    lambda_value = partial_result.lambda_star
                    self.logger.info("Updated streaming lambda to %.6f after %d documents.", lambda_value, processed_docs)

            for idx, record in enumerate(records):
                docno = record.docno
                full_emb = batch_full_emb[idx].detach().cpu()
                doc_profile_rows = [row for row in batch_util_df.to_dict(orient="records") if row["docno"] == docno]
                profile_util_map = {row["profile"]: row["utility"] for row in doc_profile_rows}
                chosen_profile_name = optimizer.choose_profile_online(profile_util_map, lambda_value)
                chosen_profile = profile_by_name[chosen_profile_name]
                reduced = full_emb[: chosen_profile.dimension].clone().detach().cpu()

                all_docnos.append(docno)
                metadata_rows.append({"docno": docno, "text": record.text})
                assignment_rows.append(
                    {
                        "docno": docno,
                        "profile": chosen_profile_name,
                        "utility": float(profile_util_map[chosen_profile_name]),
                        "cost_bytes": chosen_profile.cost_bytes,
                    }
                )

                full_docnos_by_profile[full_profile.name].append(docno)
                full_embs_by_profile[full_profile.name].append(full_emb)
                opt_docnos_by_profile[chosen_profile_name].append(docno)
                opt_embs_by_profile[chosen_profile_name].append(reduced)
                full_lookup[docno] = (full_profile.name, full_emb)
                opt_lookup[docno] = (chosen_profile_name, reduced)
                processed_docs += 1
            return lambda_value

        for record in tqdm(loader.iter_corpus(), desc="Streaming corpus", disable=not self.config.execution.verbose):
            pending_records.append(record)
            if len(pending_records) >= self.config.execution.doc_batch_size:
                current_lambda = flush_batch(pending_records, current_lambda)
                pending_records = []
                warmup_done = True
        if pending_records:
            current_lambda = flush_batch(pending_records, current_lambda)

        metadata = pd.DataFrame(metadata_rows)
        save_df(metadata, self.output_dir / "corpus_metadata.parquet")

        full_corpus = EmbeddedCorpus(
            docnos_by_profile={k: v for k, v in full_docnos_by_profile.items()},
            embeddings_by_profile={k: torch.stack(v, dim=0) for k, v in full_embs_by_profile.items()},
        )
        opt_corpus = EmbeddedCorpus(
            docnos_by_profile={k: v for k, v in opt_docnos_by_profile.items()},
            embeddings_by_profile={k: torch.stack(v, dim=0) for k, v in opt_embs_by_profile.items()},
        )

        assignments = pd.DataFrame(assignment_rows)
        utility_pairs_df = pd.concat(utility_pairs_parts, ignore_index=True) if utility_pairs_parts else pd.DataFrame()
        utility_table_df = pd.DataFrame(utility_rows)
        opt_result = type("StreamingOptimizationResult", (), {
            "lambda_star": float(current_lambda),
            "feasible": int(assignments["cost_bytes"].sum()) <= budget_bytes,
            "total_utility": float(assignments["utility"].sum()),
        })()

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
            "docnos": all_docnos,
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
        sampled_rows = []
        per_doc_profile_utilities = defaultdict(list)

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

        utility_pairs_df = pd.DataFrame(sampled_rows)
        aggregated_rows = []
        for docno in subset_docnos:
            for profile in profiles:
                values = per_doc_profile_utilities.get((docno, profile.name), [])
                agg = float(np.mean(values)) if values else (1.0 if profile.name == full_profile.name else 0.0)
                aggregated_rows.append({"docno": docno, "profile": profile.name, "utility": agg})
        utility_table_df = pd.DataFrame(aggregated_rows)
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
        return pd.DataFrame(sampled)

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

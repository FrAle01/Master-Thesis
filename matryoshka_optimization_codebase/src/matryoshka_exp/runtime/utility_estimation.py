from __future__ import annotations

from collections import defaultdict
import hashlib

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class UtilityEstimator:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def estimate_for_subset(
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
        qrels_views = self._prepare_qrels_views(qrels)
        sample_pairs = self._build_score_sampling_frame(topics, qrels_views, subset_docnos)
        sample_columns = ["qid", "docno", "profile", "full_score", "reduced_score", "utility"]
        sampled_rows = []
        per_doc_profile_utilities = defaultdict(list)

        if sample_pairs.empty:
            self.logger.warning("No sample pairs for utility estimation. Returning default utilities.")
            utility_pairs_df = pd.DataFrame(columns=sample_columns)
            aggregated_rows = []
            for docno in subset_docnos:
                for profile in profiles:
                    default_utility = 1.0 if profile.name == full_profile.name else 0.0
                    aggregated_rows.append({"docno": docno, "profile": profile.name, "utility": default_utility})
            utility_table_df = pd.DataFrame(aggregated_rows, columns=["docno", "profile", "utility"])
            return utility_pairs_df, utility_table_df

        qid_to_position = {str(qid): i for i, qid in enumerate(topics["qid"].astype(str).tolist())}
        grouped_pairs = list(sample_pairs.groupby("qid"))
        for qid, group in tqdm(
            grouped_pairs,
            desc="Estimating per-doc utilities",
            disable=not self.config.execution.verbose,
        ):
            q_offset = qid_to_position[str(qid)]
            q_full = full_query_embeddings[q_offset : q_offset + 1]
            group_docnos = group["docno"].astype(str).tolist()
            group_indices = [doc_index[d] for d in group_docnos]
            doc_full = subset_doc_embeddings[group_indices]
            full_scores = adapter.similarity(q_full, doc_full).squeeze(0)

            neg_docnos = self._sample_margin_negatives(str(qid), qrels_views, subset_docnos, group_docnos)
            neg_indices = [doc_index[d] for d in neg_docnos if d in doc_index]
            neg_scores_full = None
            if neg_indices:
                self.logger.debug("For qid %s, sampled %s margin negatives for utility estimation.", qid, len(neg_indices))
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
                self.logger.debug(
                    "For qid %s and profile %s, computed utilities for %s docs.",
                    qid,
                    profile.name,
                    len(group_docnos),
                )

                full_scores_np = full_scores.detach().cpu().numpy()
                reduced_scores_np = reduced_scores.detach().cpu().numpy()
                utility_np = utility.detach().cpu().numpy()

                for docno, s_full, s_red, u in zip(group_docnos, full_scores_np, reduced_scores_np, utility_np):
                    sampled_rows.append(
                        {
                            "qid": str(qid),
                            "docno": docno,
                            "profile": profile.name,
                            "full_score": float(s_full),
                            "reduced_score": float(s_red),
                            "utility": float(u),
                        }
                    )
                    per_doc_profile_utilities[(docno, profile.name)].append(float(u))

        utility_pairs_df = pd.DataFrame(sampled_rows, columns=sample_columns)
        aggregated_rows = []
        self.logger.info("Aggregating utilities per doc and profile for %s docs and %s profiles.", len(subset_docnos), len(profiles))
        default_count = 0
        for docno in subset_docnos:
            for profile in profiles:
                values = per_doc_profile_utilities.get((docno, profile.name), [])
                if values:
                    agg = float(np.sum(values))
                else:
                    agg = 1.0 if profile.name == full_profile.name else 0.0
                    default_count = default_count + 1 if profile.name == full_profile.name else default_count
                    self.logger.debug(
                        "No utility samples for docno %s and profile %s. Using default utility of %s.",
                        docno,
                        profile.name,                        
                        agg,
                    )

                aggregated_rows.append({"docno": docno, "profile": profile.name, "utility": agg})
        self.logger.info(
            "Completed utility aggregation. %s out of %s docs used default utility values.",
            default_count,
            len(subset_docnos),
        )
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
                self.logger.debug("No margin negatives for qid %s. Using default utility computation.", qid)
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

    def _sample_margin_negatives(self, qid: str, qrels_views, subset_docnos, positive_docnos):
        qid = str(qid)
        positive_set = set(qrels_views["positive_by_qid"].get(qid, set()))
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

    def _build_score_sampling_frame(self, topics, qrels_views, subset_docnos):
        subset_docnos = [str(d) for d in subset_docnos]
        max_pairs = self.config.utility.sample_pairs_per_query

        subset_docno_set = set(subset_docnos)
        rng = np.random.default_rng(int(self.config.utility.seed))
        sampled = []
        for qid in topics["qid"].astype(str).tolist():
            positives = [d for d in qrels_views["positive_by_qid"].get(qid, set()) if d in subset_docno_set]
            if not positives:
                continue
            self.logger.debug("For qid %s, found %s positive docnos in subset.", qid, len(positives))
            positive_set = set(positives)
            if max_pairs == -1:
                selected = positives
            else:
                if len(positives) >= max_pairs:
                    selected = [str(x) for x in rng.choice(positives, size=max_pairs, replace=False)]
                else:
                    qrel_zero_set = qrels_views["zero_by_qid"].get(qid, set())
                    qrel_docno_set = qrels_views["all_by_qid"].get(qid, set())
                    negative_pool = [
                        d for d in subset_docnos if (d in qrel_zero_set or d not in qrel_docno_set) and d not in positive_set
                    ]
                    n_neg = max(0, max_pairs - len(positives))
                    self.logger.debug(
                        "For qid %s, sampling up to %s negatives from pool of %s candidates.",
                        qid,
                        n_neg,
                        len(negative_pool),
                    )
                    if n_neg > 0 and negative_pool:
                        take = min(n_neg, len(negative_pool))
                        negatives = [str(x) for x in rng.choice(negative_pool, size=take, replace=False)]
                    else:
                        negatives = []
                    selected = positives + negatives
            self.logger.debug("For qid %s, selected %s docnos for utility estimation.", qid, len(selected))
            for docno in selected:
                sampled.append({"qid": qid, "docno": docno})
        return pd.DataFrame(sampled, columns=["qid", "docno"])

    def _prepare_qrels_views(self, qrels: pd.DataFrame):
        qrels_slim = qrels.loc[:, ["qid", "docno", self._resolve_qrels_label_column(qrels)]].copy()
        qrels_slim["qid"] = qrels_slim["qid"].astype(str)
        qrels_slim["docno"] = qrels_slim["docno"].astype(str)
        label_col = self._resolve_qrels_label_column(qrels_slim)
        qrels_slim[label_col] = pd.to_numeric(qrels_slim[label_col], errors="coerce")
        if qrels_slim[label_col].isna().any():
            raise ValueError(f"Qrels column `{label_col}` contains non-numeric relevance labels.")

        threshold = float(self.config.utility.relevance_threshold)
        positive_by_qid = {}
        zero_by_qid = {}
        all_by_qid = {}
        for qid, group in qrels_slim.groupby("qid", sort=False):
            docs = group["docno"].tolist()
            labels = group[label_col].to_numpy()
            all_by_qid[qid] = set(docs)
            positive_by_qid[qid] = {doc for doc, lab in zip(docs, labels) if lab >= threshold}
            zero_by_qid[qid] = {doc for doc, lab in zip(docs, labels) if lab == 0}

        return {
            "positive_by_qid": positive_by_qid,
            "zero_by_qid": zero_by_qid,
            "all_by_qid": all_by_qid,
        }

    def _resolve_qrels_label_column(self, qrels: pd.DataFrame) -> str:
        column = str(self.config.data.qrels_label_column)
        if column not in qrels.columns:
            raise ValueError(
                f"Qrels label column `{column}` not found. Available columns: {sorted(qrels.columns.tolist())}"
            )
        return column

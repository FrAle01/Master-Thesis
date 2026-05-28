from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch


@dataclass
class TailUtilityResult:
    utility_table: pd.DataFrame
    report: dict


def build_rbp_residual_tail_utilities(
    *,
    config,
    logger,
    adapter,
    profiles,
    full_profile,
    topics,
    subset_docnos,
    subset_doc_embeddings: torch.Tensor,
    full_query_embeddings: torch.Tensor,
    unranked_docnos,
    score_preservation_fn: Callable[..., np.ndarray],
) -> TailUtilityResult:
    unranked_docnos = [str(d) for d in unranked_docnos]
    columns = ["docno", "profile", "utility", "profile_quality"]
    if not unranked_docnos:
        return TailUtilityResult(pd.DataFrame(columns=columns), _empty_report(config))

    tail_cfg = config.utility.tail
    n_unranked = len(unranked_docnos)
    target_residual = float(tail_cfg.target_residual)
    expected_mass = target_residual * float(tail_cfg.expected_fraction)
    uncertainty_mass = max(0.0, target_residual - expected_mass)
    expected_relevance = min(float(tail_cfg.max_tail_relevance), expected_mass / max(n_unranked, 1))
    uncertainty_relevance = uncertainty_mass / max(n_unranked, 1)

    if target_residual == 0.0:
        logger.warning("RBP residual tail utility is enabled with target_residual=0; all tail expected utilities will be zero.")
    if float(tail_cfg.expected_fraction) == 0.0:
        logger.warning("RBP residual tail utility has expected_fraction=0; beta-calibrated mode can only decrease from zero.")
    if float(tail_cfg.max_tail_relevance) == 0.0:
        logger.warning("RBP residual tail utility has max_tail_relevance=0; expected tail relevance is capped at zero.")
    if tail_cfg.conservatism_beta is None:
        logger.info("RBP residual tail utility will use lower-bound utility_low=%s because conservatism_beta is null.", tail_cfg.utility_low)
    else:
        logger.info(
            "RBP residual tail utility will use beta-calibrated expected utility: max(0, expected - beta * uncertainty), beta=%s.",
            tail_cfg.conservatism_beta,
        )
    logger.info(
        "Tail residual distribution: docs=%d target_residual=%.8g expected_mass=%.8g uncertainty_mass=%.8g expected_relevance_per_doc=%.8g uncertainty_per_doc=%.8g.",
        n_unranked,
        target_residual,
        expected_mass,
        uncertainty_mass,
        expected_relevance,
        uncertainty_relevance,
    )

    doc_index = {str(docno): i for i, docno in enumerate(subset_docnos)}
    unranked_indices = [doc_index[d] for d in unranked_docnos if d in doc_index]
    unranked_docnos = [d for d in unranked_docnos if d in doc_index]
    if not unranked_indices:
        return TailUtilityResult(pd.DataFrame(columns=columns), _empty_report(config))

    doc_embeddings = subset_doc_embeddings[unranked_indices]
    if str(tail_cfg.profile_quality) == "sampled_dense_score_preservation":
        logger.info(
            "Computing tail profile quality with sampled dense score preservation using up to %d sampled queries.",
            tail_cfg.sampled_quality_queries,
        )
        quality_by_profile = _sampled_dense_score_preservation(
            config=config,
            adapter=adapter,
            profiles=profiles,
            doc_embeddings=doc_embeddings,
            full_query_embeddings=full_query_embeddings,
            score_preservation_fn=score_preservation_fn,
        )
    else:
        logger.info("Computing tail profile quality with cosine-padding preservation.")
        quality_by_profile = _cosine_padding_quality(profiles=profiles, doc_embeddings=doc_embeddings)

    beta = tail_cfg.conservatism_beta
    rows = []
    profile_quality_summary = {}
    utilities = []
    for profile in profiles:
        qualities = np.asarray(quality_by_profile[profile.name], dtype=float)
        profile_quality_summary[profile.name] = float(np.mean(qualities)) if qualities.size else 0.0
        if qualities.size and float(np.mean(qualities)) <= 0.0:
            logger.warning("Tail profile quality mean for profile %s is %.8g; tail utilities for this profile may be zero.", profile.name, float(np.mean(qualities)))
        for docno, quality in zip(unranked_docnos, qualities):
            expected_utility = expected_relevance * float(quality)
            uncertainty_utility = uncertainty_relevance * float(quality)
            if beta is None:
                utility = float(tail_cfg.utility_low)
            else:
                utility = max(0.0, expected_utility - float(beta) * uncertainty_utility)
            rows.append(
                {
                    "docno": docno,
                    "profile": profile.name,
                    "utility": float(utility),
                    "profile_quality": float(quality),
                }
            )
            utilities.append(float(utility))

    table = pd.DataFrame(rows, columns=columns)
    report = {
        "tail_mode": str(tail_cfg.mode),
        "tail_unranked_docs": int(len(unranked_docnos)),
        "tail_target_residual": target_residual,
        "tail_expected_relevance_per_doc": float(expected_relevance),
        "tail_uncertainty_per_doc": float(uncertainty_relevance),
        "tail_profile_quality": str(tail_cfg.profile_quality),
        "tail_profile_quality_mean_by_profile": profile_quality_summary,
        "tail_beta": None if beta is None else float(beta),
        "tail_utility_min": float(np.min(utilities)) if utilities else 0.0,
        "tail_utility_mean": float(np.mean(utilities)) if utilities else 0.0,
        "tail_utility_max": float(np.max(utilities)) if utilities else 0.0,
    }
    logger.info(
        "RBP residual tail utility applied to %d docs. expected_relevance_per_doc=%.8g uncertainty_per_doc=%.8g beta=%s quality=%s",
        len(unranked_docnos),
        expected_relevance,
        uncertainty_relevance,
        beta,
        tail_cfg.profile_quality,
    )
    logger.info("Tail utility profile-quality means: %s", profile_quality_summary)
    logger.info(
        "Tail utility distribution: min=%.8g mean=%.8g max=%.8g.",
        report["tail_utility_min"],
        report["tail_utility_mean"],
        report["tail_utility_max"],
    )
    return TailUtilityResult(table, report)


def _cosine_padding_quality(*, profiles, doc_embeddings: torch.Tensor) -> dict[str, np.ndarray]:
    full_norm = torch.linalg.vector_norm(doc_embeddings, dim=1).clamp_min(1e-12)
    out = {}
    for profile in profiles:
        reduced = torch.zeros_like(doc_embeddings)
        dim = min(int(profile.dimension), int(doc_embeddings.shape[1]))
        reduced[:, :dim] = doc_embeddings[:, :dim]
        reduced_norm = torch.linalg.vector_norm(reduced, dim=1).clamp_min(1e-12)
        cosine = torch.sum(doc_embeddings * reduced, dim=1) / (full_norm * reduced_norm)
        out[profile.name] = torch.clamp(cosine, 0.0, 1.0).detach().cpu().numpy()
    return out


def _sampled_dense_score_preservation(
    *,
    config,
    adapter,
    profiles,
    doc_embeddings: torch.Tensor,
    full_query_embeddings: torch.Tensor,
    score_preservation_fn: Callable[..., np.ndarray],
) -> dict[str, np.ndarray]:
    n_queries = int(full_query_embeddings.shape[0])
    sample_n = min(int(config.utility.tail.sampled_quality_queries), n_queries)
    if sample_n == 0:
        return {profile.name: np.zeros(int(doc_embeddings.shape[0]), dtype=float) for profile in profiles}
    rng = np.random.default_rng(int(config.utility.seed))
    if sample_n < n_queries:
        q_indices = rng.choice(np.arange(n_queries), size=sample_n, replace=False)
    else:
        q_indices = np.arange(n_queries)

    query_sample = full_query_embeddings[q_indices]
    out = {profile.name: np.zeros(int(doc_embeddings.shape[0]), dtype=float) for profile in profiles}
    counts = {profile.name: np.zeros(int(doc_embeddings.shape[0]), dtype=float) for profile in profiles}

    chunk_size = 512
    for start in range(0, int(doc_embeddings.shape[0]), chunk_size):
        end = min(start + chunk_size, int(doc_embeddings.shape[0]))
        docs = doc_embeddings[start:end]
        full_scores = adapter.similarity(query_sample, docs).detach().cpu().numpy()
        for profile in profiles:
            q_reduced = query_sample[:, : profile.dimension]
            d_reduced = docs[:, : profile.dimension]
            reduced_scores = adapter.similarity(q_reduced, d_reduced).detach().cpu().numpy()
            quality_sum = np.zeros(end - start, dtype=float)
            for q_idx in range(full_scores.shape[0]):
                quality_sum += score_preservation_fn(
                    full_scores=full_scores[q_idx],
                    reduced_scores=reduced_scores[q_idx],
                )
            out[profile.name][start:end] += quality_sum
            counts[profile.name][start:end] += full_scores.shape[0]

    for profile in profiles:
        denom = np.maximum(counts[profile.name], 1.0)
        out[profile.name] = np.clip(out[profile.name] / denom, 0.0, 1.0)
    return out


def _empty_report(config) -> dict:
    return {
        "tail_mode": str(config.utility.tail.mode),
        "tail_unranked_docs": 0,
        "tail_target_residual": float(config.utility.tail.target_residual),
        "tail_expected_relevance_per_doc": 0.0,
        "tail_uncertainty_per_doc": 0.0,
        "tail_profile_quality": str(config.utility.tail.profile_quality),
        "tail_profile_quality_mean_by_profile": {},
        "tail_beta": None if config.utility.tail.conservatism_beta is None else float(config.utility.tail.conservatism_beta),
        "tail_utility_min": 0.0,
        "tail_utility_mean": 0.0,
        "tail_utility_max": 0.0,
    }

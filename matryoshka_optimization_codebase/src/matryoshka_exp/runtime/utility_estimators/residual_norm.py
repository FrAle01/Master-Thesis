from __future__ import annotations

import pandas as pd
import torch
from tqdm import tqdm

from .contracts import UtilityEstimationRequest, UtilityEstimationResult, UtilityEstimator


class ResidualNormUtilityEstimator(UtilityEstimator):
    requires_query_data = False
    _CHUNK_SIZE = 65536

    def estimate(self, request: UtilityEstimationRequest) -> UtilityEstimationResult:
        embeddings = request.doc_embeddings
        if embeddings.ndim != 2:
            raise ValueError("Document embeddings must be a rank-2 tensor.")
        if len(request.docnos) != int(embeddings.shape[0]):
            raise ValueError("Document ids and embedding rows must have the same length.")

        norm_order = int(self.config.utility.residual_norm.norm)
        epsilon = float(self.config.utility.residual_norm.epsilon)
        full_dimension = int(embeddings.shape[1])
        if int(request.full_profile.dimension) != full_dimension:
            raise ValueError(
                "Full profile dimension must match the document embedding width: "
                f"{request.full_profile.dimension} != {full_dimension}."
            )
        oversized = [
            str(profile.name)
            for profile in request.profiles
            if int(profile.dimension) > full_dimension
        ]
        if oversized:
            raise ValueError(f"Profiles exceed the document embedding width: {oversized}")

        rows = []
        utility_count = 0
        utility_sum = 0.0
        utility_min = float("inf")
        utility_max = float("-inf")

        for start in tqdm(range(0, len(request.docnos), self._CHUNK_SIZE), desc="Estimating utilities", unit="docs"):
            end = min(start + self._CHUNK_SIZE, len(request.docnos))
            chunk = embeddings[start:end].detach().to(dtype=torch.float32)
            full_norm = torch.linalg.vector_norm(chunk, ord=norm_order, dim=1)
            zero_mask = full_norm <= epsilon
            denominator = full_norm.clamp_min(epsilon)

            for profile in request.profiles:
                dimension = int(profile.dimension)
                if profile.name == request.full_profile.name or dimension >= full_dimension:
                    utilities = torch.ones(end - start, dtype=torch.float32, device=chunk.device)
                else:
                    residual_norm = torch.linalg.vector_norm(
                        chunk[:, dimension:],
                        ord=norm_order,
                        dim=1,
                    )
                    utilities = torch.clamp(1.0 - residual_norm / denominator, min=0.0, max=1.0)
                    utilities = torch.where(zero_mask, torch.ones_like(utilities), utilities)

                values = utilities.detach().cpu().tolist()
                if values:
                    utility_count += len(values)
                    utility_sum += float(sum(values))
                    utility_min = min(utility_min, float(min(values)))
                    utility_max = max(utility_max, float(max(values)))
                rows.extend(
                    {
                        "docno": str(docno),
                        "profile": str(profile.name),
                        "utility": float(utility),
                    }
                    for docno, utility in zip(request.docnos[start:end], values)
                )

        utility_table = pd.DataFrame(rows, columns=["docno", "profile", "utility"])
        report = {
            "estimator": "residual_norm",
            "requires_query_data": False,
            "norm": norm_order,
            "epsilon": epsilon,
            "num_docs": int(len(request.docnos)),
            "num_profiles": int(len(request.profiles)),
            "utility_min": utility_min if utility_count else 0.0,
            "utility_mean": utility_sum / utility_count if utility_count else 0.0,
            "utility_max": utility_max if utility_count else 0.0,
        }
        self.last_report = report
        self.logger.info("Residual-norm utility estimation report: %s", report)
        return UtilityEstimationResult(
            utility_table=utility_table,
            pair_details=pd.DataFrame(),
            report=report,
        )

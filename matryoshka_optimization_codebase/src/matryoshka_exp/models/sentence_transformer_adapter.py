from __future__ import annotations

from typing import List

import torch

from .base import EncoderAdapter, RepresentationProfile


class SentenceTransformerAdapter(EncoderAdapter):
    """Adapter for SentenceTransformer-compatible models.

    This is the preferred path for existing Matryoshka-capable embedding models such as:
    - nomic-ai/modernbert-embed-base
    - nomic-ai/nomic-embed-text-v1.5
    - mixedbread-ai/mxbai-embed-large-v1

    The SentenceTransformer API supports `truncate_dim`, which is directly useful for
    standard dimension-wise Matryoshka experiments.
    """

    def __init__(self, model_cfg, device: str, dtype: str):
        super().__init__(model_cfg, device, dtype)
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_cfg.model_name_or_path,
            device=device,
            trust_remote_code=model_cfg.trust_remote_code,
        )

    def embed_texts(
        self,
        texts: List[str],
        profile: RepresentationProfile,
        *,
        prompt_name: str,
        batch_size: int,
    ) -> torch.Tensor:
        if prompt_name == "query" and self.model_cfg.query_prompt:
            texts = [self.model_cfg.query_prompt + t for t in texts]
        elif prompt_name == "document" and self.model_cfg.document_prompt:
            texts = [self.model_cfg.document_prompt + t for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=profile.normalize,
            truncate_dim=profile.dimension,
        )
        return embeddings

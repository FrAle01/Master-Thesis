from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from ..config import ModelConfig, ProfileConfig


@dataclass
class RepresentationProfile:
    name: str
    dimension: int
    layer: Optional[int]
    normalize: bool
    cost_bytes: int


class EncoderAdapter(ABC):
    """Abstract encoder interface.

    The rest of the codebase interacts with profiles through this interface, so the
    experiment works with both regular Matryoshka models and 2D Starbucks-like models.
    """

    def __init__(self, model_cfg: ModelConfig, device: str, dtype: str):
        self.model_cfg = model_cfg
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def embed_texts(
        self,
        texts: List[str],
        profile: RepresentationProfile,
        *,
        prompt_name: str,
        batch_size: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    def similarity(self, queries: torch.Tensor, docs: torch.Tensor) -> torch.Tensor:
        if self.model_cfg.similarity == "dot":
            return queries @ docs.T
        if self.model_cfg.similarity == "cosine":
            q = torch.nn.functional.normalize(queries, p=2, dim=-1)
            d = torch.nn.functional.normalize(docs, p=2, dim=-1)
            return q @ d.T
        raise ValueError(f"Unsupported similarity: {self.model_cfg.similarity}")


def build_profile_catalog(profiles: List[ProfileConfig], dtype: str) -> List[RepresentationProfile]:
    bytes_per_value = {
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
    }.get(dtype, 4)

    catalog: List[RepresentationProfile] = []
    for item in profiles:
        cost = item.cost_bytes if item.cost_bytes is not None else item.dimension * bytes_per_value
        catalog.append(
            RepresentationProfile(
                name=item.name,
                dimension=item.dimension,
                layer=item.layer,
                normalize=item.normalize,
                cost_bytes=int(cost),
            )
        )
    return catalog

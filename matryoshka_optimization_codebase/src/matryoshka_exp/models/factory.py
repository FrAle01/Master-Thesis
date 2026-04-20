from __future__ import annotations

from ..config import ExperimentConfig
from .base import build_profile_catalog
from .sentence_transformer_adapter import SentenceTransformerAdapter
from .starbucks_adapter import StarbucksAdapter


def create_encoder(config: ExperimentConfig):
    if config.model.backend == "sentence_transformers":
        adapter = SentenceTransformerAdapter(config.model, config.execution.device, config.execution.dtype)
    elif config.model.backend == "transformers":
        adapter = StarbucksAdapter(config.model, config.execution.device, config.execution.dtype)
    else:
        raise ValueError(f"Unsupported model backend: {config.model.backend}")

    profiles = build_profile_catalog(config.profiles, config.execution.dtype)
    return adapter, profiles

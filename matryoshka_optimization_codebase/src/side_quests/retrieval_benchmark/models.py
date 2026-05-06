from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from matryoshka_exp.training.lora_utils import load_lora_adapter

from .config import ModelEntry


@dataclass
class LoadedModel:
    id: str
    model_entry: ModelEntry
    model: object


class ModelResolver:
    def __init__(self, *, device: str):
        self.device = device

    def load(self, entry: ModelEntry) -> LoadedModel:
        from sentence_transformers import SentenceTransformer

        if entry.type in {"hf_model", "local_model"}:
            model = SentenceTransformer(
                entry.path_or_repo,
                device=self.device,
                trust_remote_code=entry.trust_remote_code,
            )
        elif entry.type == "lora_adapter":
            if not entry.base_model:
                raise ValueError("lora_adapter model requires `base_model`.")
            model = SentenceTransformer(
                entry.base_model,
                device=self.device,
                trust_remote_code=entry.trust_remote_code,
            )
            load_lora_adapter(model, entry.path_or_repo, adapter_name=entry.adapter_name)
        else:
            raise ValueError(f"Unsupported model type: {entry.type}")

        return LoadedModel(id=entry.id, model_entry=entry, model=model)


def encode_texts(
    loaded: LoadedModel,
    texts: List[str],
    *,
    is_query: bool,
    dimension: int,
    batch_size: int,
) -> torch.Tensor:
    entry = loaded.model_entry
    if is_query and entry.query_prompt:
        texts = [entry.query_prompt + t for t in texts]
    if not is_query and entry.document_prompt:
        texts = [entry.document_prompt + t for t in texts]

    return loaded.model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_tensor=True,
        normalize_embeddings=entry.normalize,
        truncate_dim=dimension,
    )

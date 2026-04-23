from __future__ import annotations

from typing import List

import torch

from .base import EncoderAdapter, RepresentationProfile


class StarbucksAdapter(EncoderAdapter):
    """Layer-aware adapter for 2D Matryoshka / Starbucks-style checkpoints.

    The adapter uses hidden states from a Hugging Face transformer model and applies:
    - layer selection
    - pooling
    - dimension truncation

    This is intentionally generic: it can be used with `ielabgroup/Starbucks-msmarco`
    and with other HF models that expose hidden states.
    """

    def __init__(self, model_cfg, device: str, dtype: str):
        super().__init__(model_cfg, device, dtype)
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.tokenizer_name_or_path or model_cfg.model_name_or_path,
            trust_remote_code=model_cfg.trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(
            model_cfg.model_name_or_path,
            trust_remote_code=model_cfg.trust_remote_code,
            output_hidden_states=True,
        ).to(device)
        self.model.eval()

    def embed_texts(
        self,
        texts: List[str],
        profile: RepresentationProfile,
        *,
        prompt_name: str,
        batch_size: int,
    ) -> torch.Tensor:
        outputs: List[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            if prompt_name == "query" and self.model_cfg.query_prompt:
                batch = [self.model_cfg.query_prompt + t for t in batch]
            elif prompt_name == "document" and self.model_cfg.document_prompt:
                batch = [self.model_cfg.document_prompt + t for t in batch]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**encoded)
                hidden_states = out.hidden_states
                layer_idx = profile.layer if profile.layer is not None else len(hidden_states) - 1
                selected = hidden_states[layer_idx]
                pooled = self._pool(selected, encoded["attention_mask"])
                pooled = pooled[:, : profile.dimension]
                if profile.normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
                outputs.append(pooled.detach().cpu())
        if not outputs:
            return torch.empty((0, profile.dimension), dtype=self.target_torch_dtype)
        return self.cast_embeddings(torch.cat(outputs, dim=0))

    def _pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.model_cfg.pooling == "cls":
            return token_embeddings[:, 0]
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any, Dict, Optional

import torch

from ..results.persistence import ensure_dir
from ..retrieval.checkpointing import (
    collect_valid_chunks,
    load_manifest,
    load_tensor_from_chunks,
    manifest_matches_context,
)
from ..retrieval.embedding_store import load_full_embeddings_from_hf, save_full_embeddings_to_hf
from ..retrieval.materialization import encode_full_corpus
from .device_policy import resolve_torch_dtype


class EmbeddingPipeline:
    def __init__(self, config, output_dir: Path, logger):
        self.config = config
        self.output_dir = output_dir
        self.logger = logger

    def save_query_embeddings_checkpoint(self, query_ids, query_embeddings: torch.Tensor, usage: str = "opt") -> None:
        from ..results.persistence import save_json

        torch.save(query_embeddings.detach().cpu(), self.output_dir / f"query_embeddings{f'_{usage}' if usage != 'opt' else ''}.pt")
        save_json({"query_ids": [str(qid) for qid in query_ids], "usage": usage}, self.output_dir / f"query_ids{f'_{usage}' if usage != 'opt' else ''}.json")

    def load_or_compute_full_doc_embeddings(self, *, adapter, full_profile, docnos, corpus_records):
        source = self.config.data.full_embeddings_source
        target_repo_id = self.resolve_model_specific_embeddings_repo_id()
        expected_docnos = [str(docno) for docno in docnos]

        for local_stage in ("compute", "hf_load"):
            local_embeddings = self.try_load_full_doc_embeddings_from_local_checkpoint(
                stage=local_stage,
                full_profile=full_profile,
                expected_docnos=expected_docnos,
                repo_id=target_repo_id,
                split=self.config.data.hf_embeddings_split,
            )
            self.logger.info(
                "Checked for local `%s` checkpoints for full document embeddings. Found valid checkpoint: %s",
                local_stage,
                "yes" if local_embeddings is not None else "no",
            )
            if local_embeddings is not None:
                self.logger.info(
                    "Recovered %s full document embeddings from local `%s` checkpoints.",
                    len(expected_docnos),
                    local_stage,
                )
                return local_embeddings

        if source in {"auto", "hf_dataset"}:
            try:
                hf_checkpoint_cfg = self.embedding_checkpoint_cfg(
                    stage="hf_load",
                    full_profile=full_profile,
                    source="hf_dataset",
                    repo_id=target_repo_id,
                    split=self.config.data.hf_embeddings_split,
                )
                loaded_docnos, loaded_embeddings = load_full_embeddings_from_hf(
                    repo_id=target_repo_id,
                    split=self.config.data.hf_embeddings_split,
                    docno_column=self.config.data.hf_embeddings_docno_column,
                    vector_column=self.config.data.hf_embeddings_vector_column,
                    expected_docnos=docnos,
                    expected_dimension=full_profile.dimension,
                    target_dtype=self.target_dtype_from_execution(),
                    verbose=self.config.execution.verbose,
                    checkpoint_cfg=hf_checkpoint_cfg,
                )
                if loaded_docnos != expected_docnos:
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

        compute_checkpoint_cfg = self.embedding_checkpoint_cfg(
            stage="compute",
            full_profile=full_profile,
            source="compute",
            repo_id=target_repo_id,
            split=self.config.data.hf_embeddings_split,
        )
        _docnos, computed_embeddings, _ = encode_full_corpus(
            iter(corpus_records),
            adapter,
            full_profile,
            batch_size=self.config.execution.doc_batch_size,
            prompt_name="document",
            verbose=self.config.execution.verbose,
            checkpoint_cfg=compute_checkpoint_cfg,
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

    def try_load_full_doc_embeddings_from_local_checkpoint(
        self,
        *,
        stage: str,
        full_profile,
        expected_docnos,
        repo_id: str | None,
        split: str,
    ) -> torch.Tensor | None:
        checkpoint_cfg = self.embedding_checkpoint_cfg(
            stage=stage,
            full_profile=full_profile,
            source="hf_dataset" if stage == "hf_load" else "compute",
            repo_id=repo_id,
            split=split,
        )
        if not checkpoint_cfg:
            self.logger.info("Local embedding checkpointing is disabled for stage `%s`.", stage)
            return None

        chunk_dir = Path(checkpoint_cfg["chunk_dir"])
        manifest_path = Path(checkpoint_cfg["manifest_path"])
        context = dict(checkpoint_cfg.get("context", {}))
        expected_dtype = checkpoint_cfg["dtype"]
        expected_dim = int(full_profile.dimension)

        manifest = load_manifest(manifest_path)
        if not manifest or not manifest_matches_context(manifest, context):
            if not manifest:
                self.logger.info("No manifest found at %s for local `%s` checkpoints.", manifest_path, stage)
            else:
                self.logger.warning(
                    "Manifest context mismatch for local `%s` checkpoints at %s. Expected context: %s, manifest context: %s. Ignoring checkpoint.",
                    stage,
                    manifest_path,
                    context,
                    manifest.get("context", {}),
                )
            return None

        valid_chunks = collect_valid_chunks(chunk_dir, expected_dim, expected_dtype)
        contiguous = []
        for idx, chunk in enumerate(valid_chunks):
            if chunk[0] != idx:
                break
            contiguous.append(chunk)
        if not contiguous:
            self.logger.info("No valid contiguous chunk sequence found in %s for local `%s` checkpoints.", chunk_dir, stage)
            return None

        loaded_docnos, loaded_tensor = load_tensor_from_chunks(chunk_dir, contiguous, expected_dim, expected_dtype)
        if len(loaded_docnos) != len(expected_docnos):
            self.logger.warning(
                "Loaded docnos length mismatch for local `%s` checkpoints at %s. Expected %s docnos, got %s docnos. Ignoring checkpoint.",
                stage,
                manifest_path,
                len(expected_docnos),
                len(loaded_docnos),
            )
            return None
        if [str(d) for d in loaded_docnos] != [str(d) for d in expected_docnos]:
            self.logger.warning(
                "Loaded docnos content mismatch for local `%s` checkpoints at %s. Ignoring checkpoint.",
                stage,
                manifest_path,
            )
            return None
        if int(loaded_tensor.shape[0]) != len(expected_docnos):
            self.logger.warning(
                "Loaded tensor shape mismatch for local `%s` checkpoints at %s. Ignoring checkpoint.",
                stage,
                manifest_path,
            )
            return None

        if stage == "hf_load":
            num_rows_in_checkpoints = int(manifest.get("num_rows_loaded", 0))
        else:
            num_rows_in_checkpoints = int(manifest.get("num_docs_done", 0))

        if num_rows_in_checkpoints < len(expected_docnos):
            self.logger.warning(
                "Loaded tensor incomplete for local `%s` checkpoints at %s. Manifest indicates only %s rows loaded, but expected %s rows. Ignoring checkpoint.",
                stage,
                manifest_path,
                num_rows_in_checkpoints,
                len(expected_docnos),
            )
            return None

        return loaded_tensor.to(expected_dtype)

    def embedding_checkpoint_cfg(
        self,
        *,
        stage: str,
        full_profile,
        source: str,
        repo_id: str | None,
        split: str,
    ) -> dict | None:
        if not self.config.execution.embedding_checkpoint_enabled:
            return None
        base_dir = (
            Path(self.config.execution.embedding_checkpoint_dir)
            if self.config.execution.embedding_checkpoint_dir
            else (self.output_dir / "embedding_checkpoints")
        )
        stage_dir = ensure_dir(base_dir / stage)
        return {
            "every_docs": int(self.config.execution.embedding_checkpoint_every_docs),
            "resume_mode": str(self.config.execution.embedding_checkpoint_resume),
            "chunk_dir": str(stage_dir / "chunks"),
            "manifest_path": str(stage_dir / "manifest.json"),
            "dtype": self.target_dtype_from_execution(),
            "context": {
                "stage": stage,
                "source": source,
                "repo_id": str(repo_id or ""),
                "split": str(split),
                "dimension": int(full_profile.dimension),
                "model_name_or_path": str(self.config.model.model_name_or_path),
                "adapter_type": str(self.config.model.adapter_type or ""),
                "adapter_name": str(self.config.model.adapter_name),
            },
        }

    def resolve_model_specific_embeddings_repo_id(self) -> str | None:
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
            return f"{namespace}/{repo_name}_{model_slug}"
        return f"{base_repo_id}_{model_slug}"

    def target_dtype_from_execution(self) -> torch.dtype:
        return resolve_torch_dtype(self.config.execution.dtype)

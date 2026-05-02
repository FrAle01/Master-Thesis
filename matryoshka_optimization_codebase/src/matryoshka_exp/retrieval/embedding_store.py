from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm
from .checkpointing import (
    atomic_write_docnos,
    atomic_write_json,
    collect_valid_chunks,
    load_manifest,
    load_tensor_from_chunks,
    manifest_matches_context,
)


def _resolve_hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def load_full_embeddings_from_hf(
    *,
    repo_id: str,
    split: str,
    docno_column: str,
    vector_column: str,
    expected_docnos: Sequence[str],
    expected_dimension: int,
    target_dtype: torch.dtype,
    verbose: bool = True,
    checkpoint_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], torch.Tensor]:
    from datasets import load_dataset

    token = _resolve_hf_token()
    if not token:
        raise ValueError(
            "Missing Hugging Face token. Private embedding datasets require `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`."
        )

    try:
        ds = load_dataset(repo_id, split=split, token=token)
    except Exception as exc:
        message = str(exc).lower()
        if any(keyword in message for keyword in ("401", "403", "unauthorized", "forbidden", "not found", "404")):
            raise ValueError(
                f"Cannot access Hugging Face dataset `{repo_id}` (split `{split}`). "
                "Check that the token has read access and the repo is visible to your account."
            ) from exc
        raise
    columns = set(ds.column_names)

    missing_columns = [col for col in (docno_column, vector_column) if col not in columns]
    if missing_columns:
        raise ValueError(
            f"Hugging Face embedding dataset `{repo_id}` split `{split}` is missing columns: {missing_columns}"
        )

    chunk_doc_target = int((checkpoint_cfg or {}).get("every_docs", 0))
    chunk_dir = Path((checkpoint_cfg or {}).get("chunk_dir", "")) if checkpoint_cfg else None
    manifest_path = Path((checkpoint_cfg or {}).get("manifest_path", "")) if checkpoint_cfg else None
    resume_mode = str((checkpoint_cfg or {}).get("resume_mode", "auto"))
    context = dict((checkpoint_cfg or {}).get("context", {}))
    num_rows_loaded = 0
    vector_by_docno = {}

    if checkpoint_cfg and chunk_doc_target > 0 and chunk_dir is not None and manifest_path is not None:
        chunk_dir.mkdir(parents=True, exist_ok=True)
        manifest = load_manifest(manifest_path)
        if manifest and not manifest_matches_context(manifest, context):
            if resume_mode == "fail":
                raise ValueError("HF embedding checkpoint manifest does not match current context.")
            if resume_mode == "restart":
                manifest = None
        if manifest:
            valid_chunks = collect_valid_chunks(chunk_dir, expected_dimension, target_dtype)
            contiguous = []
            expected_idx = 0
            for chunk in valid_chunks:
                if chunk[0] != expected_idx:
                    break
                contiguous.append(chunk)
                expected_idx += 1
            loaded_docnos, loaded_tensor = load_tensor_from_chunks(
                chunk_dir, contiguous, expected_dimension, target_dtype
            )
            for d, emb in zip(loaded_docnos, loaded_tensor):
                vector_by_docno[str(d)] = emb.to(torch.float32).tolist()
            num_rows_loaded = int(manifest.get("num_rows_loaded", len(loaded_docnos)))
            if num_rows_loaded < len(loaded_docnos):
                num_rows_loaded = len(loaded_docnos)

    chunk_buffer_docnos: List[str] = []
    chunk_buffer_vectors: List[torch.Tensor] = []
    next_chunk_idx = len(collect_valid_chunks(chunk_dir, expected_dimension, target_dtype)) if checkpoint_cfg and chunk_dir else 0

    def flush_checkpoint_chunk() -> None:
        nonlocal chunk_buffer_docnos, chunk_buffer_vectors, next_chunk_idx
        if not checkpoint_cfg or not chunk_buffer_docnos or chunk_dir is None or manifest_path is None:
            return
        emb_tensor = torch.stack(chunk_buffer_vectors, dim=0).to(target_dtype)
        emb_path = chunk_dir / f"chunk_{next_chunk_idx:06d}.pt"
        docno_path = chunk_dir / f"chunk_{next_chunk_idx:06d}.docnos.txt"
        torch.save(emb_tensor, emb_path)
        atomic_write_docnos(chunk_buffer_docnos, docno_path)
        atomic_write_json(
            {
                **context,
                "num_rows_loaded": num_rows_loaded,
                "next_chunk_idx": next_chunk_idx + 1,
                "dimension": int(expected_dimension),
                "dtype": str(target_dtype),
            },
            manifest_path,
        )
        next_chunk_idx += 1
        chunk_buffer_docnos = []
        chunk_buffer_vectors = []

    for idx, row in enumerate(tqdm(ds, desc="Loading HF full embeddings", disable=not verbose)):
        if idx < num_rows_loaded:
            continue
        docno = str(row[docno_column])
        if docno in vector_by_docno:
            raise ValueError(f"Duplicate docno `{docno}` found in Hugging Face embedding dataset `{repo_id}`.")
        vector = row[vector_column]
        if not isinstance(vector, (list, tuple)):
            raise ValueError(
                f"Embedding value for docno `{docno}` must be a list/tuple, got `{type(vector).__name__}`."
            )
        if len(vector) != expected_dimension:
            raise ValueError(
                f"Embedding size mismatch for docno `{docno}`: expected {expected_dimension}, got {len(vector)}."
            )
        vector_f = [float(v) for v in vector]
        vector_by_docno[docno] = vector_f
        num_rows_loaded += 1
        if checkpoint_cfg and chunk_doc_target > 0:
            chunk_buffer_docnos.append(docno)
            chunk_buffer_vectors.append(torch.tensor(vector_f, dtype=torch.float32))
            if len(chunk_buffer_docnos) >= chunk_doc_target:
                flush_checkpoint_chunk()
    flush_checkpoint_chunk()

    aligned_vectors = []
    missing_docnos = []
    for docno in expected_docnos:
        vector = vector_by_docno.get(str(docno))
        if vector is None:
            missing_docnos.append(str(docno))
            continue
        aligned_vectors.append(vector)

    if missing_docnos:
        preview = missing_docnos[:5]
        raise ValueError(
            f"Embedding dataset `{repo_id}` split `{split}` is incomplete. Missing {len(missing_docnos)} docnos "
            f"(first few: {preview})."
        )

    tensor = torch.tensor(aligned_vectors, dtype=torch.float32)
    if tensor.dtype != target_dtype:
        tensor = tensor.to(target_dtype)
    return [str(docno) for docno in expected_docnos], tensor


def save_full_embeddings_to_hf(
    *,
    repo_id: str,
    split: str,
    docnos: Sequence[str],
    embeddings: torch.Tensor,
    docno_column: str,
    vector_column: str,
) -> None:
    from datasets import Dataset, Features, Sequence, Value

    if len(docnos) != int(embeddings.shape[0]):
        raise ValueError(
            f"Cannot save embeddings: docnos count ({len(docnos)}) does not match embeddings rows ({embeddings.shape[0]})."
        )

    # Keep memory usage bounded: avoid materializing a giant Python list via `.tolist()` on the full tensor.
    embeddings_cpu = embeddings.detach().cpu().to(torch.float32)
    docnos_str = [str(docno) for docno in docnos]
    features = Features(
        {
            docno_column: Value("string"),
            vector_column: Sequence(Value("float32"), length=int(embeddings_cpu.shape[1])),
        }
    )

    def _row_generator():
        for i, docno in enumerate(docnos_str):
            yield {
                docno_column: docno,
                vector_column: embeddings_cpu[i].tolist(),
            }

    ds = Dataset.from_generator(_row_generator, features=features)

    push_kwargs = {"repo_id": repo_id, "split": split}
    token = _resolve_hf_token()
    if token:
        push_kwargs["token"] = token
    # Smaller shards reduce peak RAM during Arrow serialization and upload.
    push_kwargs["max_shard_size"] = "500MB"
    ds.push_to_hub(**push_kwargs)

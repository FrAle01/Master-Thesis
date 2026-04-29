from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import torch


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

    vector_by_docno = {}
    for row in ds:
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
        vector_by_docno[docno] = [float(v) for v in vector]

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
    from datasets import Dataset

    if len(docnos) != int(embeddings.shape[0]):
        raise ValueError(
            f"Cannot save embeddings: docnos count ({len(docnos)}) does not match embeddings rows ({embeddings.shape[0]})."
        )

    vectors = embeddings.detach().cpu().to(torch.float32).tolist()
    ds = Dataset.from_dict(
        {
            docno_column: [str(docno) for docno in docnos],
            vector_column: vectors,
        }
    )

    push_kwargs = {"repo_id": repo_id, "split": split}
    token = _resolve_hf_token()
    if token:
        push_kwargs["token"] = token
    ds.push_to_hub(**push_kwargs)

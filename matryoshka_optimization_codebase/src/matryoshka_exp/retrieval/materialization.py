from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
import pandas as pd
import torch
from tqdm import tqdm

from ..data.pyterrier_utils import CorpusRecord
from ..models.base import EncoderAdapter, RepresentationProfile
from .checkpointing import (
    atomic_write_docnos,
    atomic_write_json,
    collect_valid_chunks,
    load_manifest,
    load_tensor_from_chunks,
    manifest_matches_context,
)
from .dense import EmbeddedCorpus


def encode_full_corpus(
    corpus_iter: Iterator[CorpusRecord],
    adapter: EncoderAdapter,
    profile: RepresentationProfile,
    *,
    batch_size: int,
    prompt_name: str = "document",
    verbose: bool = True,
    checkpoint_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], torch.Tensor]:
    # Only return doc ids + embedding matrix to avoid retaining full corpus text in memory.
    docnos: List[str] = []
    texts: List[str] = []
    embeddings_batches = []

    chunk_doc_target = int((checkpoint_cfg or {}).get("every_docs", 0))
    chunk_dir = Path((checkpoint_cfg or {}).get("chunk_dir", "")) if checkpoint_cfg else None
    manifest_path = Path((checkpoint_cfg or {}).get("manifest_path", "")) if checkpoint_cfg else None
    resume_mode = str((checkpoint_cfg or {}).get("resume_mode", "auto"))
    expected_dtype = (checkpoint_cfg or {}).get("dtype", torch.float32)
    context = dict((checkpoint_cfg or {}).get("context", {}))
    resume_docnos = []
    resume_tensor = torch.empty((0, profile.dimension), dtype=expected_dtype)
    resumed_doc_count = 0

    if checkpoint_cfg and chunk_doc_target > 0 and chunk_dir is not None and manifest_path is not None:
        chunk_dir.mkdir(parents=True, exist_ok=True)
        manifest = load_manifest(manifest_path)
        if manifest and not manifest_matches_context(manifest, context):
            if resume_mode == "fail":
                raise ValueError("Embedding checkpoint manifest does not match current context.")
            if resume_mode == "restart":
                manifest = None
        if manifest:
            valid_chunks = collect_valid_chunks(chunk_dir, profile.dimension, expected_dtype)
            contiguous = []
            expected_idx = 0
            for chunk in valid_chunks:
                if chunk[0] != expected_idx:
                    break
                contiguous.append(chunk)
                expected_idx += 1
            resume_docnos, resume_tensor = load_tensor_from_chunks(
                chunk_dir,
                contiguous,
                profile.dimension,
                expected_dtype,
            )
            resumed_doc_count = len(resume_docnos)
            if int(manifest.get("num_docs_done", resumed_doc_count)) != resumed_doc_count and resume_mode == "fail":
                raise ValueError("Embedding checkpoint manifest/doc count mismatch.")

    iterator = corpus_iter
    skipped = 0
    while skipped < resumed_doc_count:
        try:
            next(iterator)
        except StopIteration:
            break
        skipped += 1

    if resumed_doc_count:
        docnos.extend(resume_docnos)
        embeddings_batches.append(resume_tensor)

    chunk_buffer_docnos: List[str] = []
    chunk_buffer_embs: List[torch.Tensor] = []
    next_chunk_idx = len(collect_valid_chunks(chunk_dir, profile.dimension, expected_dtype)) if checkpoint_cfg and chunk_dir else 0

    def flush_checkpoint_chunk() -> None:
        nonlocal chunk_buffer_docnos, chunk_buffer_embs, next_chunk_idx
        if not checkpoint_cfg or not chunk_buffer_docnos or chunk_dir is None or manifest_path is None:
            return
        emb_tensor = torch.cat(chunk_buffer_embs, dim=0).to(expected_dtype)
        emb_path = chunk_dir / f"chunk_{next_chunk_idx:06d}.pt"
        docno_path = chunk_dir / f"chunk_{next_chunk_idx:06d}.docnos.txt"
        torch.save(emb_tensor, emb_path)
        atomic_write_docnos(chunk_buffer_docnos, docno_path)
        num_done = len(docnos)
        atomic_write_json(
            {
                **context,
                "num_docs_done": num_done,
                "next_chunk_idx": next_chunk_idx + 1,
                "dimension": int(profile.dimension),
                "dtype": str(expected_dtype),
            },
            manifest_path,
        )
        next_chunk_idx += 1
        chunk_buffer_docnos = []
        chunk_buffer_embs = []

    for record in tqdm(iterator, desc=f"Encoding documents [{profile.name}]", disable=not verbose):
        docnos.append(record.docno)
        texts.append(record.text)
        if len(texts) >= batch_size:
            emb = adapter.embed_texts(texts, profile, prompt_name=prompt_name, batch_size=batch_size)
            emb_cpu = emb.cpu()
            embeddings_batches.append(emb_cpu)
            if checkpoint_cfg and chunk_doc_target > 0:
                chunk_buffer_embs.append(emb_cpu.to(expected_dtype))
                chunk_buffer_docnos.extend(docnos[-len(emb_cpu):])
                if len(chunk_buffer_docnos) >= chunk_doc_target:
                    flush_checkpoint_chunk()
            texts = []
    if texts:
        emb = adapter.embed_texts(texts, profile, prompt_name=prompt_name, batch_size=batch_size)
        emb_cpu = emb.cpu()
        embeddings_batches.append(emb_cpu)
        if checkpoint_cfg and chunk_doc_target > 0:
            chunk_buffer_embs.append(emb_cpu.to(expected_dtype))
            chunk_buffer_docnos.extend(docnos[-len(emb_cpu):])
    flush_checkpoint_chunk()
    all_embeddings = torch.cat(embeddings_batches, dim=0) if embeddings_batches else torch.empty((0, profile.dimension))
    return docnos, all_embeddings


def materialize_grouped_corpus(
    full_docnos: List[str],
    full_embeddings: torch.Tensor,
    assignments: pd.DataFrame,
    profiles: Dict[str, RepresentationProfile],
    *,
    target_device: str = "cpu",
) -> Tuple[EmbeddedCorpus, Dict[str, Tuple[str, torch.Tensor]]]:
    grouped_docnos: Dict[str, List[str]] = defaultdict(list)
    grouped_embs: Dict[str, List[torch.Tensor]] = defaultdict(list)
    lookup: Dict[str, Tuple[str, torch.Tensor]] = {}

    profile_by_docno = dict(zip(assignments["docno"], assignments["profile"]))
    for docno, full_emb in zip(full_docnos, full_embeddings):
        profile_name = profile_by_docno[docno]
        profile = profiles[profile_name]
        reduced = full_emb[: profile.dimension].clone().detach().cpu()
        grouped_docnos[profile_name].append(docno)
        grouped_embs[profile_name].append(reduced)
        lookup[docno] = (profile_name, reduced)

    embeddings_by_profile: Dict[str, torch.Tensor] = {}
    for profile_name in grouped_docnos.keys():
        rows = grouped_embs.get(profile_name, [])
        if rows:
            profile_matrix = torch.stack(rows, dim=0)
        else:
            profile_matrix = torch.empty((0, profiles[profile_name].dimension))
        if target_device != "cpu":
            profile_matrix = profile_matrix.to(target_device)
        embeddings_by_profile[profile_name] = profile_matrix

    corpus = EmbeddedCorpus(
        docnos_by_profile={k: v for k, v in grouped_docnos.items()},
        embeddings_by_profile=embeddings_by_profile,
    )
    return corpus, lookup

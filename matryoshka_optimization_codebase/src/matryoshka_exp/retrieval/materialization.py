from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from ..data.pyterrier_utils import CorpusRecord
from ..models.base import EncoderAdapter, RepresentationProfile
from .dense import EmbeddedCorpus


def encode_full_corpus(
    corpus_iter: Iterator[CorpusRecord],
    adapter: EncoderAdapter,
    profile: RepresentationProfile,
    *,
    batch_size: int,
    prompt_name: str = "document",
    verbose: bool = True,
) -> Tuple[List[str], torch.Tensor, pd.DataFrame]:
    docnos: List[str] = []
    texts: List[str] = []
    metadata_rows = []
    embeddings_batches = []

    iterator = corpus_iter
    for record in tqdm(iterator, desc=f"Encoding documents [{profile.name}]", disable=not verbose):
        docnos.append(record.docno)
        texts.append(record.text)
        metadata_rows.append({"docno": record.docno, "text": record.text})
        if len(texts) >= batch_size:
            emb = adapter.embed_texts(texts, profile, prompt_name=prompt_name, batch_size=batch_size)
            embeddings_batches.append(emb.cpu())
            texts = []
    if texts:
        emb = adapter.embed_texts(texts, profile, prompt_name=prompt_name, batch_size=batch_size)
        embeddings_batches.append(emb.cpu())
    all_embeddings = torch.cat(embeddings_batches, dim=0) if embeddings_batches else torch.empty((0, profile.dimension))
    metadata = pd.DataFrame(metadata_rows)
    return docnos, all_embeddings, metadata


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
        reduced = full_emb[: profile.dimension].clone().detach()
        if target_device != "cpu":
            reduced = reduced.to(target_device)
        else:
            reduced = reduced.cpu()
        grouped_docnos[profile_name].append(docno)
        grouped_embs[profile_name].append(reduced)
        lookup[docno] = (profile_name, reduced)

    corpus = EmbeddedCorpus(
        docnos_by_profile={k: v for k, v in grouped_docnos.items()},
        embeddings_by_profile={
            k: (
                torch.stack(v, dim=0) if len(v) else torch.empty((0, profiles[k].dimension), device=target_device)
            )
            for k, v in grouped_embs.items()
        },
    )
    return corpus, lookup

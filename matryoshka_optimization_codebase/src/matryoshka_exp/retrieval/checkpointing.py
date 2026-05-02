from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


def atomic_write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def atomic_write_docnos(docnos: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for docno in docnos:
            f.write(f"{str(docno)}\n")
    os.replace(tmp_path, path)


def read_docnos(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def load_manifest(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def manifest_matches_context(manifest: Dict[str, Any], context: Dict[str, Any]) -> bool:
    for key, value in context.items():
        if manifest.get(key) != value:
            return False
    return True


def collect_valid_chunks(
    chunk_dir: Path,
    expected_dimension: int,
    expected_dtype: torch.dtype,
) -> List[Tuple[int, Path, Path]]:
    chunks: List[Tuple[int, Path, Path]] = []
    for emb_path in chunk_dir.glob("chunk_*.pt"):
        stem = emb_path.stem
        if not stem.startswith("chunk_"):
            continue
        idx_str = stem.replace("chunk_", "", 1)
        if not idx_str.isdigit():
            continue
        chunk_idx = int(idx_str)
        docno_path = chunk_dir / f"chunk_{idx_str}.docnos.txt"
        if not docno_path.exists():
            continue
        try:
            emb = torch.load(emb_path, map_location="cpu")
            if not isinstance(emb, torch.Tensor):
                continue
            if emb.ndim != 2 or int(emb.shape[1]) != int(expected_dimension):
                continue
            if emb.dtype != expected_dtype:
                emb = emb.to(expected_dtype)
                torch.save(emb, emb_path)
            docnos = read_docnos(docno_path)
            if len(docnos) != int(emb.shape[0]):
                continue
        except Exception:
            continue
        chunks.append((chunk_idx, emb_path, docno_path))
    chunks.sort(key=lambda x: x[0])
    return chunks


def load_tensor_from_chunks(
    chunk_dir: Path,
    chunks: Sequence[Tuple[int, Path, Path]],
    expected_dimension: int,
    expected_dtype: torch.dtype,
) -> Tuple[List[str], torch.Tensor]:
    docnos: List[str] = []
    tensors: List[torch.Tensor] = []
    for _, emb_path, docno_path in chunks:
        emb = torch.load(emb_path, map_location="cpu")
        if emb.dtype != expected_dtype:
            emb = emb.to(expected_dtype)
        tensors.append(emb)
        docnos.extend(read_docnos(docno_path))
    if tensors:
        return docnos, torch.cat(tensors, dim=0)
    return [], torch.empty((0, expected_dimension), dtype=expected_dtype)


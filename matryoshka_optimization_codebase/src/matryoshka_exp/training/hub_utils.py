from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional


def resolve_hf_token(token_env: str = "HF_TOKEN") -> Optional[str]:
    token = os.getenv(token_env or "")
    if token:
        return token
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "artifact"


def resolve_hub_repo_id(*, hub_cfg, experiment_name: str, finetune_strategy: str) -> str:
    if not hub_cfg.auto_repo_from_experiment:
        raise ValueError("Only `training.hub.auto_repo_from_experiment=true` is currently supported.")
    owner = (hub_cfg.repo_prefix or "").strip()
    if not owner:
        raise ValueError("`training.hub.repo_prefix` must be set when Hub upload is enabled.")
    suffix = hub_cfg.repo_suffix_lora if finetune_strategy == "lora" else hub_cfg.repo_suffix_full
    repo_name = f"{_slugify(experiment_name)}-{_slugify(suffix)}"
    return f"{owner}/{repo_name}"


def upload_folder_to_hub(
    *,
    folder_path: str | Path,
    repo_id: str,
    private: bool,
    token: Optional[str],
    commit_message: str,
) -> dict:
    from huggingface_hub import HfApi

    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise ValueError(f"Cannot upload missing folder: {folder_path}")
    if not folder_path.is_dir():
        raise ValueError(f"Cannot upload non-directory path: {folder_path}")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    commit_info = api.upload_folder(
        repo_id=repo_id,
        folder_path=str(folder_path),
        repo_type="model",
        commit_message=commit_message,
        ignore_patterns=["checkpoint-*", "checkpoints/*", "checkpoints/**", "checkpoint/*", "checkpoint/**"],
    )
    revision = getattr(commit_info, "oid", None) or getattr(commit_info, "commit_hash", None) or ""
    return {"repo_id": repo_id, "revision": str(revision)}

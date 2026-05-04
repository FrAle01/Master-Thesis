from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Optional

import torch


def _get_sentence_transformer_backbone(sentence_model):
    first_module = sentence_model._first_module()
    if hasattr(first_module, "auto_model"):
        return first_module, first_module.auto_model
    raise ValueError("The SentenceTransformer model does not expose a transformer backbone via `auto_model`.")


def discover_lora_target_modules(backbone, exclude_modules: Optional[Iterable[str]] = None) -> List[str]:
    excluded = {item.lower() for item in (exclude_modules or [])}
    linear_suffixes = set()

    for name, module in backbone.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        suffix = name.split(".")[-1]
        name_l = name.lower()
        suffix_l = suffix.lower()
        if any(token in name_l or token == suffix_l for token in excluded):
            continue
        linear_suffixes.add(suffix)

    if not linear_suffixes:
        raise ValueError("Could not auto-discover LoRA target modules: no linear layers were found.")

    preferred_suffixes = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "query",
        "key",
        "value",
        "dense",
        "fc1",
        "fc2",
        "proj",
    ]
    preferred = [name for name in preferred_suffixes if name in linear_suffixes]
    return preferred or sorted(linear_suffixes)


def apply_lora_to_sentence_transformer(sentence_model, lora_cfg, logger=None) -> List[str]:
    from peft import LoraConfig, TaskType, get_peft_model

    first_module, backbone = _get_sentence_transformer_backbone(sentence_model)
    target_modules = lora_cfg.target_modules or discover_lora_target_modules(
        backbone, exclude_modules=lora_cfg.exclude_modules
    )

    task_type_name = (lora_cfg.task_type or "FEATURE_EXTRACTION").upper()
    try:
        task_type = TaskType[task_type_name]
    except KeyError as exc:
        valid = ", ".join([name for name in TaskType.__members__])
        raise ValueError(f"Unsupported LoRA task_type `{task_type_name}`. Valid options: {valid}") from exc

    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        bias=lora_cfg.bias,
        target_modules=target_modules,
        modules_to_save=lora_cfg.modules_to_save or None,
        task_type=task_type,
        inference_mode=False,
    )
    first_module.auto_model = get_peft_model(backbone, peft_config, adapter_name=lora_cfg.adapter_name)

    if logger is not None:
        logger.info("Applied LoRA adapter `%s` to target modules: %s", lora_cfg.adapter_name, target_modules)
    return target_modules


def save_lora_adapter(sentence_model, output_adapter_dir: str | Path, metadata: Optional[dict] = None) -> Path:
    output_path = Path(output_adapter_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _, backbone = _get_sentence_transformer_backbone(sentence_model)
    backbone.save_pretrained(str(output_path))

    if metadata:
        metadata_path = output_path / "lora_adapter_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    return output_path


def load_lora_metadata(adapter_path: str | Path) -> Optional[dict]:
    adapter_ref = str(adapter_path)
    if not adapter_ref.strip():
        return None

    metadata_filename = "lora_adapter_metadata.json"
    local_candidate = Path(adapter_ref) / metadata_filename
    if local_candidate.exists():
        with open(local_candidate, "r", encoding="utf-8") as f:
            return json.load(f)

    # Support Hub repo ids directly.
    if "/" in adapter_ref and not Path(adapter_ref).exists():
        from huggingface_hub import hf_hub_download

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        try:
            downloaded = hf_hub_download(
                repo_id=adapter_ref,
                filename=metadata_filename,
                repo_type="model",
                token=token,
            )
        except Exception:
            return None
        with open(downloaded, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_lora_adapter(sentence_model, adapter_path: str | Path, adapter_name: str = "default", logger=None) -> None:
    from peft import PeftModel

    adapter_ref = str(adapter_path).strip()
    if not adapter_ref:
        raise ValueError("LoRA adapter path/repo id cannot be empty.")
    is_local_path = Path(adapter_ref).exists()
    if ("/" not in adapter_ref) and not is_local_path:
        raise ValueError(f"LoRA adapter path does not exist and is not a valid Hub repo id: {adapter_ref}")

    first_module, backbone = _get_sentence_transformer_backbone(sentence_model)
    metadata = load_lora_metadata(adapter_ref)
    if metadata:
        expected_base = str(metadata.get("base_model_name_or_path", "")).strip()
        if expected_base and str(getattr(backbone.config, "_name_or_path", "")).strip() != expected_base and logger is not None:
            logger.warning(
                "LoRA metadata base model `%s` does not match runtime backbone `%s`.",
                expected_base,
                str(getattr(backbone.config, "_name_or_path", "")).strip(),
            )
    elif ("/" in adapter_ref) and not is_local_path and logger is not None:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if not token:
            logger.warning(
                "No HF token detected while loading adapter repo `%s`. Loading private adapters may fail.",
                adapter_ref,
            )

    peft_model = PeftModel.from_pretrained(
        backbone,
        adapter_ref,
        adapter_name=adapter_name,
        is_trainable=False,
    )
    peft_model.set_adapter(adapter_name)
    first_module.auto_model = peft_model

    if logger is not None:
        logger.info("Loaded LoRA adapter `%s` from %s", adapter_name, adapter_ref)

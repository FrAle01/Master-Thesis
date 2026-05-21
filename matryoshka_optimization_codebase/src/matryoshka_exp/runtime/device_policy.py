from __future__ import annotations
from typing import Optional

import torch


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype_name, torch.float32)


def sync_profile_costs_with_observed_dtype(logger, profiles, embeddings: torch.Tensor, *, expected_dtype_name: str, stage: str) -> None:
    if embeddings.numel() == 0:
        return

    expected_dtype = resolve_torch_dtype(expected_dtype_name)
    observed_dtype = embeddings.dtype
    if observed_dtype != expected_dtype:
        logger.warning(
            "Configured execution dtype is `%s` but %s produced `%s`; profile costs will use observed dtype.",
            expected_dtype_name,
            stage,
            observed_dtype,
        )

    bytes_per_value = embeddings.element_size()
    for profile in profiles:
        if profile.cost_is_explicit:
            continue
        logger.info("Setting cost for profile `%s` based on observed dtype `%s` with %d bytes per value.", profile.name, observed_dtype, bytes_per_value)
        profile.cost_bytes = int(profile.dimension * bytes_per_value)

def resolve_retrieval_device(config) -> str:
    requested = config.execution.retrieval_device
    if not config.retrieval.use_torch_gpu_exact:
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "execution.retrieval_device='cuda' but CUDA is not available. "
            "Verify the pinned cu121 PyTorch installation and run the GPU preflight checks."
        )
    return requested


def assert_retrieval_fits_vram(config, required_bytes: int, *, retrieval_device: str, logger: Optional[None]) -> None:
    if retrieval_device != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA retrieval requested but CUDA is not available. "
            "Verify the pinned cu121 PyTorch installation and run the GPU preflight checks."
        )

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    usable_bytes = int(total_bytes * float(config.execution.retrieval_vram_utilization_limit))
    temp_overhead_factor = float(getattr(config.execution, "retrieval_vram_temp_overhead_factor", 1.2))
    peak_required_bytes = int(required_bytes * temp_overhead_factor)
    logger.info(
        "Retrieval VRAM check: required_bytes=%d, peak_required_bytes=%d, free_bytes=%d, total_bytes=%d, "
        "usable_bytes=%d (utilization_limit=%.2f, temp_overhead_factor=%.2f)",
        required_bytes,
        peak_required_bytes,
        free_bytes,
        total_bytes,
        usable_bytes,
        config.execution.retrieval_vram_utilization_limit,
        temp_overhead_factor,
    )

    if peak_required_bytes > usable_bytes or peak_required_bytes > free_bytes:
        raise RuntimeError(
            "Optimized corpus does not fit configured VRAM limits for GPU retrieval: "
            f"required_bytes={required_bytes}, peak_required_bytes={peak_required_bytes}, "
            f"free_bytes={free_bytes}, total_bytes={total_bytes}, "
            f"utilization_limit={config.execution.retrieval_vram_utilization_limit}, "
            f"temp_overhead_factor={temp_overhead_factor}"
        )

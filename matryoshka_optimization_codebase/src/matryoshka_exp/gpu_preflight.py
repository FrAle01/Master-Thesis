from __future__ import annotations

import os

import torch


def _cuda_debug_details() -> str:
    return (
        f"torch.__version__={torch.__version__}, "
        f"torch.version.cuda={torch.version.cuda}, "
        f"torch.cuda.is_available()={torch.cuda.is_available()}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
    )


def validate_cuda_runtime_or_raise(*, context: str) -> None:
    """Fail fast with actionable diagnostics if CUDA runtime is not usable."""
    expected_cuda = "12.1"

    if torch.version.cuda is None:
        raise RuntimeError(
            f"[{context}] PyTorch was installed without CUDA support. "
            "Install a CUDA wheel set compatible with this VM driver (recommended: cu121). "
            f"Diagnostics: {_cuda_debug_details()}"
        )

    if str(torch.version.cuda) != expected_cuda:
        raise RuntimeError(
            f"[{context}] Incompatible PyTorch CUDA runtime detected: torch.version.cuda={torch.version.cuda}. "
            f"Expected {expected_cuda} (cu121) for this VM setup. "
            "Reinstall PyTorch from the cu121 index. "
            f"Diagnostics: {_cuda_debug_details()}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"[{context}] CUDA is not available at runtime. "
            "This usually means a driver/runtime mismatch in the environment. "
            "Use the pinned cu121 dependency setup described in README. "
            f"Diagnostics: {_cuda_debug_details()}"
        )

    try:
        _ = torch.empty(1, device="cuda")
        _ = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - depends on runtime
        raise RuntimeError(
            f"[{context}] CUDA failed during initialization/allocation. "
            "This is commonly caused by an incompatible torch CUDA build for the installed driver. "
            "Reinstall the pinned cu121 stack and verify with nvidia-smi + python checks. "
            f"Diagnostics: {_cuda_debug_details()}. Original error: {exc}"
        ) from exc

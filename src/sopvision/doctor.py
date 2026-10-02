"""GPU and environment checks shared by `doctor`, `serve` and `eval`."""

import os
from dataclasses import dataclass

NO_GPU_MESSAGE = (
    "No GPU visible in the container. Docker Desktop needs the WSL2 backend and the Windows NVIDIA driver; "
    "check: docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi"
)


class GpuUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class GpuReport:
    cuda_available: bool
    device_name: str | None
    torch_version: str


def gpu_report() -> GpuReport:
    import torch

    available = torch.cuda.is_available()
    return GpuReport(
        cuda_available=available,
        device_name=torch.cuda.get_device_name(0) if available else None,
        torch_version=torch.__version__,
    )


def in_container() -> bool:
    return os.environ.get("SOPVISION_IN_CONTAINER", "").strip().lower() in {"1", "true", "yes"}


def require_gpu(in_container: bool, report: GpuReport) -> None:
    """Fail in the container when no GPU is visible; natively a CPU run is allowed (just slow)."""
    if in_container and not report.cuda_available:
        raise GpuUnavailableError(NO_GPU_MESSAGE)

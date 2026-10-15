"""GPU and environment checks shared by `doctor`, `serve` and `eval`."""

from dataclasses import dataclass

NO_GPU_MESSAGE = (
    "No GPU visible in the container. Enable NVIDIA Container Toolkit on Linux, or the WSL2 backend and "
    "NVIDIA driver for Docker Desktop; "
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


def require_gpu(report: GpuReport) -> None:
    """Fail when the application container cannot access a GPU."""
    if not report.cuda_available:
        raise GpuUnavailableError(NO_GPU_MESSAGE)

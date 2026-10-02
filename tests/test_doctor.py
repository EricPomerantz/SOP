import pytest

from sopvision.doctor import GpuReport, GpuUnavailableError, require_gpu

NO_GPU = GpuReport(cuda_available=False, device_name=None, torch_version="x")
GPU = GpuReport(cuda_available=True, device_name="NVIDIA GeForce RTX 4080", torch_version="x")


def test_container_without_gpu_fails_with_setup_hint():
    with pytest.raises(GpuUnavailableError, match="No GPU visible in the container"):
        require_gpu(True, NO_GPU)


def test_native_without_gpu_does_not_raise():
    require_gpu(False, NO_GPU)


def test_container_with_gpu_passes():
    require_gpu(True, GPU)

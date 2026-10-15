import pytest

from sopvision.doctor import GpuReport, GpuUnavailableError, require_gpu

NO_GPU = GpuReport(cuda_available=False, device_name=None, torch_version="x")
GPU = GpuReport(cuda_available=True, device_name="NVIDIA GeForce RTX 4080", torch_version="x")


def test_container_without_gpu_fails_with_setup_hint():
    with pytest.raises(GpuUnavailableError, match="No GPU visible in the container"):
        require_gpu(NO_GPU)


def test_legacy_native_flag_cannot_bypass_gpu_check(monkeypatch):
    monkeypatch.setenv("SOPVISION_IN_CONTAINER", "0")
    with pytest.raises(GpuUnavailableError, match="No GPU visible in the container"):
        require_gpu(NO_GPU)


def test_container_with_gpu_passes():
    require_gpu(GPU)

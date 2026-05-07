import subprocess

from app.model import gpu_status
from app.model.gpu_status import GpuMemoryStats, get_gpu_status, parse_nvidia_smi_memory_csv


def test_parse_nvidia_smi_memory_csv_represents_vram_stats():
    stats = parse_nvidia_smi_memory_csv("24576, 1024, 23552\n")

    assert stats == [GpuMemoryStats(total_mb=24576, used_mb=1024, free_mb=23552)]


def test_get_gpu_status_uses_nvidia_smi_when_available(monkeypatch):
    def fake_run(*args, **kwargs):
        assert args[0][0] == "nvidia-smi"
        return subprocess.CompletedProcess(args[0], 0, stdout="24576, 2048, 22528\n", stderr="")

    monkeypatch.setattr(gpu_status.subprocess, "run", fake_run)

    status = get_gpu_status()

    assert status.available is True
    assert status.gpus == [GpuMemoryStats(total_mb=24576, used_mb=2048, free_mb=22528)]
    assert status.detail is None


def test_get_gpu_status_has_no_gpu_dependency(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(gpu_status.subprocess, "run", fake_run)

    status = get_gpu_status()

    assert status.available is False
    assert status.gpus == []
    assert "nvidia-smi not found" in status.detail

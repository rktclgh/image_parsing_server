from dataclasses import dataclass, field
import subprocess


@dataclass(frozen=True)
class GpuMemoryStats:
    total_mb: int
    used_mb: int
    free_mb: int


@dataclass(frozen=True)
class GpuStatus:
    available: bool
    gpus: list[GpuMemoryStats] = field(default_factory=list)
    detail: str | None = None


def parse_nvidia_smi_memory_csv(output: str) -> list[GpuMemoryStats]:
    stats: list[GpuMemoryStats] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise ValueError(f"unexpected nvidia-smi memory row: {line!r}")
        total_mb, used_mb, free_mb = (_parse_memory_mb(part) for part in parts)
        stats.append(
            GpuMemoryStats(total_mb=total_mb, used_mb=used_mb, free_mb=free_mb)
        )
    return stats


def get_gpu_status() -> GpuStatus:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return GpuStatus(available=False, detail="nvidia-smi not found")
    except subprocess.TimeoutExpired:
        return GpuStatus(available=False, detail="nvidia-smi timed out")
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "nvidia-smi failed"
        return GpuStatus(available=False, detail=detail)

    try:
        return GpuStatus(
            available=True,
            gpus=parse_nvidia_smi_memory_csv(result.stdout),
        )
    except ValueError as exc:
        return GpuStatus(available=False, detail=f"failed to parse GPU stats: {exc}")


def _parse_memory_mb(value: str) -> int:
    number = value.lower().replace("mib", "").replace("mb", "").strip()
    return int(number)

"""Live hardware telemetry for the dashboard.

Uses NVML (``nvidia-ml-py``) for real GPU numbers.  When no NVIDIA driver is
present it falls back to host CPU/RAM via ``psutil`` so the dashboard still has
something truthful to show (rows are tagged kind="cpu").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DeviceStat:
    index: int
    name: str
    kind: str                  # "gpu" | "cpu"
    util: float                # core / SM utilisation, %
    mem_util: float            # memory-controller utilisation, % (GPU) or n/a
    mem_used: int              # bytes
    mem_total: int             # bytes
    temp: Optional[float] = None
    power_w: Optional[float] = None


def gpu_count() -> int:
    """Number of NVIDIA GPUs, without creating a CUDA context (0 if none)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            return int(pynvml.nvmlDeviceGetCount())
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return 0


class Telemetry:
    def __init__(self):
        self._nvml = None
        self._handles = []
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                             for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception:
            self._nvml = None
        try:
            import psutil
            self._psutil = psutil
            psutil.cpu_percent(interval=None)   # prime the counter
        except Exception:
            self._psutil = None

    @property
    def has_gpu(self) -> bool:
        return bool(self._handles)

    def sample(self) -> List[DeviceStat]:
        if self._handles:
            return [self._sample_gpu(i, h) for i, h in enumerate(self._handles)]
        return [self._sample_cpu()]

    def _sample_gpu(self, i, h) -> DeviceStat:
        n = self._nvml
        name = n.nvmlDeviceGetName(h)
        name = name.decode() if isinstance(name, bytes) else str(name)
        mem = n.nvmlDeviceGetMemoryInfo(h)
        try:
            u = n.nvmlDeviceGetUtilizationRates(h)
            util, mutil = float(u.gpu), float(u.memory)
        except Exception:
            util = mutil = 0.0
        try:
            temp = float(n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU))
        except Exception:
            temp = None
        try:
            power = n.nvmlDeviceGetPowerUsage(h) / 1000.0
        except Exception:
            power = None
        return DeviceStat(i, name, "gpu", util, mutil, int(mem.used), int(mem.total), temp, power)

    def _sample_cpu(self) -> DeviceStat:
        if self._psutil is None:
            return DeviceStat(0, "CPU (psutil not installed)", "cpu", 0.0, 0.0, 0, 0)
        vm = self._psutil.virtual_memory()
        return DeviceStat(0, "Host CPU (no NVIDIA GPU)", "cpu",
                          float(self._psutil.cpu_percent(interval=None)), float(vm.percent),
                          int(vm.total - vm.available), int(vm.total))

    def close(self):
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml = None

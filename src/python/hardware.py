"""wormsim2 hardware detection layer (v0.11.0).

Call detect_hardware() once at startup to get a HardwareProfile describing
the compute resources available on the current host.  The profile is used by
backends.select_backend() to pick the fastest skeleton / simulation kernel.

Detection order (preferred → fallback):
  CUDA (via cupy/torch.cuda)  →  Apple Metal MPS (torch.mps)
  →  OpenCL (pyopencl)  →  JAX-CPU (XLA)  →  joblib multiprocessing
  →  NumPy serial

All probes are non-destructive and free of side effects.
"""
from __future__ import annotations

import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class GPUDevice:
    name: str
    backend: str          # 'cuda' | 'mps' | 'opencl'
    vram_mb: int = 0
    compute_capability: str = ""


@dataclass
class HardwareProfile:
    # CPU
    cpu_name: str
    cpu_logical_cores: int
    cpu_physical_cores: int
    ram_gb: float

    # GPU / accelerator
    gpus: list[GPUDevice] = field(default_factory=list)

    # Available backends (ordered best→fallback)
    has_cuda: bool = False
    has_mps: bool = False
    has_opencl: bool = False
    has_jax: bool = False
    jax_devices: list[str] = field(default_factory=list)
    has_openmp_cpp: bool = False   # detected via compiler probe
    joblib_nproc: int = 1

    # Recommended Python backend token
    recommended_backend: str = "numpy_serial"  # 'cuda'|'mps'|'opencl'|'jax_cpu'|'numpy_mp'|'numpy_serial'

    # HPC estimate helpers
    hpc_node_gpu_model: str = ""   # e.g. 'Tesla V100S'  if detectable
    hpc_note: str = ""

    def summary(self) -> str:
        lines = [
            f"CPU:        {self.cpu_name}  ({self.cpu_logical_cores} logical / {self.cpu_physical_cores} physical cores)",
            f"RAM:        {self.ram_gb:.1f} GB",
        ]
        if self.gpus:
            for g in self.gpus:
                vram = f" {g.vram_mb} MB VRAM" if g.vram_mb else ""
                lines.append(f"GPU [{g.backend.upper()}]: {g.name}{vram}")
        else:
            lines.append("GPU:        none detected")
        lines.append(f"CUDA:       {'yes' if self.has_cuda else 'no'}")
        lines.append(f"MPS:        {'yes' if self.has_mps else 'no'}")
        lines.append(f"OpenCL:     {'yes' if self.has_opencl else 'no'}")
        lines.append(f"JAX:        {'yes — ' + ', '.join(self.jax_devices) if self.has_jax else 'no'}")
        lines.append(f"OpenMP(C++):{'yes' if self.has_openmp_cpp else 'no'}")
        lines.append(f"joblib jobs: {self.joblib_nproc}")
        lines.append(f"Recommended: {self.recommended_backend}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal probes
# ---------------------------------------------------------------------------


def _probe_cpu() -> tuple[str, int, int, float]:
    """Return (cpu_name, logical, physical, ram_gb)."""
    cpu_name = platform.processor() or platform.machine()

    # macOS: use sysctl for human-readable brand string
    if sys.platform == "darwin":
        try:
            r = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=3,
            )
            if r.returncode == 0 and r.stdout.strip():
                cpu_name = r.stdout.strip()
        except Exception:
            pass

    # Linux: read /proc/cpuinfo
    elif sys.platform.startswith("linux"):
        try:
            for line in open("/proc/cpuinfo").readlines():
                if "model name" in line:
                    cpu_name = line.split(":", 1)[1].strip()
                    break
        except Exception:
            pass

    logical = multiprocessing.cpu_count()

    physical = logical
    try:
        import psutil  # type: ignore
        physical = psutil.cpu_count(logical=False) or logical
    except ImportError:
        if sys.platform == "darwin":
            try:
                r = subprocess.run(
                    ["sysctl", "-n", "hw.physicalcpu"],
                    capture_output=True, text=True, timeout=3,
                )
                physical = int(r.stdout.strip()) if r.returncode == 0 else logical
            except Exception:
                pass
        elif sys.platform.startswith("linux"):
            try:
                seen: set[str] = set()
                for line in open("/proc/cpuinfo"):
                    if "physical id" in line:
                        seen.add(line.split(":", 1)[1].strip())
                physical = len(seen) or logical
            except Exception:
                pass

    ram_gb = 0.0
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1024**3
    except ImportError:
        if sys.platform == "darwin":
            try:
                r = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=3,
                )
                ram_gb = int(r.stdout.strip()) / 1024**3
            except Exception:
                pass
        elif sys.platform.startswith("linux"):
            try:
                for line in open("/proc/meminfo"):
                    if line.startswith("MemTotal"):
                        ram_gb = int(line.split()[1]) / 1024**2
                        break
            except Exception:
                pass

    return cpu_name, logical, physical, ram_gb


def _probe_cuda() -> list[GPUDevice]:
    """Return list of CUDA GPUs (cupy > torch.cuda > nvidia-smi fallback)."""
    devices: list[GPUDevice] = []

    # cupy (fastest)
    try:
        import cupy as cp  # type: ignore
        n = cp.cuda.runtime.getDeviceCount()
        for i in range(n):
            props = cp.cuda.runtime.getDeviceProperties(i)
            vram = props["totalGlobalMem"] // 1024**2
            cc = f"{props['major']}.{props['minor']}"
            devices.append(GPUDevice(props["name"].decode(), "cuda", vram, cc))
        if devices:
            return devices
    except Exception:
        pass

    # torch.cuda
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append(GPUDevice(
                    props.name, "cuda",
                    props.total_memory // 1024**2,
                    f"{props.major}.{props.minor}",
                ))
            return devices
    except Exception:
        pass

    # nvidia-smi fallback (no Python bindings needed)
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    devices.append(GPUDevice(parts[0], "cuda", int(parts[1]), parts[2]))
    except Exception:
        pass

    return devices


def _probe_mps() -> bool:
    """True if Apple Metal Performance Shaders GPU is available."""
    try:
        import torch  # type: ignore
        return bool(torch.backends.mps.is_available())
    except Exception:
        pass

    # macOS: check for Metal support via system_profiler
    if sys.platform == "darwin":
        try:
            r = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=5,
            )
            if "Metal" in r.stdout:
                return True
        except Exception:
            pass
    return False


def _probe_opencl() -> list[str]:
    """Return list of OpenCL platform names."""
    try:
        import pyopencl as cl  # type: ignore
        return [p.name for p in cl.get_platforms()]
    except Exception:
        pass
    # clinfo fallback
    if shutil.which("clinfo"):
        try:
            r = subprocess.run(["clinfo", "--list"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                return [r.stdout.strip().splitlines()[0]]
        except Exception:
            pass
    return []


def _probe_jax() -> tuple[bool, list[str]]:
    """Return (available, device_strings)."""
    try:
        import jax  # type: ignore
        devs = [str(d) for d in jax.devices()]
        return True, devs
    except Exception:
        return False, []


def _probe_openmp_cpp() -> bool:
    """Check if the compiled wormsim2 C++ was built with OpenMP."""
    # Look for the build dir relative to this file's package root
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.join(here, "..", "..")
    cmake_cache = os.path.join(repo_root, "build", "CMakeCache.txt")
    if os.path.isfile(cmake_cache):
        with open(cmake_cache) as f:
            content = f.read()
        if "OpenMP_FOUND:INTERNAL=1" in content or "WORMSIM2_OPENMP=ON" in content:
            return True
    # Secondary: try omp_get_num_threads via ctypes
    import ctypes
    for libname in ("libgomp.so.1", "libgomp.dylib", "libomp.dylib", "libomp.so"):
        try:
            lib = ctypes.cdll.LoadLibrary(libname)
            lib.omp_get_num_threads.restype = ctypes.c_int
            _ = lib.omp_get_num_threads()
            return True
        except Exception:
            continue
    return False


def _pick_backend(profile: "HardwareProfile") -> str:
    if profile.has_cuda:
        return "cuda"
    if profile.has_mps:
        return "mps"
    if profile.has_opencl:
        return "opencl"
    if profile.has_jax and any("gpu" in d.lower() or "tpu" in d.lower() for d in profile.jax_devices):
        return "jax_gpu"
    if profile.has_jax:
        return "jax_cpu"
    if profile.joblib_nproc > 1:
        return "numpy_mp"
    return "numpy_serial"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_hardware() -> HardwareProfile:
    """Probe all hardware and return a HardwareProfile.

    This function is designed to never raise: every probe is wrapped so that
    a missing optional dependency gracefully degrades.
    """
    cpu_name, logical, physical, ram_gb = _probe_cpu()

    gpus = _probe_cuda()
    has_cuda = bool(gpus)

    has_mps = _probe_mps()
    if has_mps and not gpus:
        # Apple Silicon / AMD Metal — add a synthetic GPU entry
        gpus.append(GPUDevice("Apple GPU (Metal)", "mps"))

    opencl_platforms = _probe_opencl()
    has_opencl = bool(opencl_platforms)

    has_jax, jax_devs = _probe_jax()
    has_openmp = _probe_openmp_cpp()

    # joblib workers: physical cores (avoid hyperthreading overhead for compute)
    joblib_nproc = max(1, physical)

    profile = HardwareProfile(
        cpu_name=cpu_name,
        cpu_logical_cores=logical,
        cpu_physical_cores=physical,
        ram_gb=ram_gb,
        gpus=gpus,
        has_cuda=has_cuda,
        has_mps=has_mps,
        has_opencl=has_opencl,
        has_jax=has_jax,
        jax_devices=jax_devs,
        has_openmp_cpp=has_openmp,
        joblib_nproc=joblib_nproc,
    )
    profile.recommended_backend = _pick_backend(profile)
    return profile

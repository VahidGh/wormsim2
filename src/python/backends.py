"""wormsim2 parallel skeleton / kinematics backends (v0.11.0).

Angle convention
----------------
All backends accept **theta_body** angles — the wormsim2 standard defined in
``skeleton.py``:

    theta_body[j] = arctan2(y[j+1] − y[j], x[j+1] − x[j]) + π/2

This is an **absolute tangent angle** (not a turning angle / relative bend).
Do NOT apply cumsum to theta_body before passing it in.

The canonical forward kinematics is implemented in ``skeleton.tangent_to_xy_batch``
and is the single source of truth for all NumPy backends.  JAX has its own
JIT-compiled equivalent (documented to match).

Interface
---------
    compute_skeletons(
        theta_body: np.ndarray,  # (n_frames, n_seg) — theta_body convention
        ds: float,               # arc-length per segment (BL); = 1 / n_seg
        mid_idx: int,            # index to centre the skeleton at
    ) -> tuple[np.ndarray, np.ndarray]   # xs (n_fr, n_pts), ys (n_fr, n_pts)

Usage
-----
    from backends import select_backend
    bk = select_backend()
    xs, ys = bk.compute_skeletons(tuner.theta_rec, ds=1/48, mid_idx=24)

    # bench all backends:
    from backends import benchmark_backends
    results = benchmark_backends(tuner.theta_rec, ds=1/48, mid_idx=24)
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from skeleton import tangent_to_xy_batch   # canonical FK — single source of truth

if TYPE_CHECKING:
    from hardware import HardwareProfile

# ---------------------------------------------------------------------------
# Serial NumPy helper (used only for the benchmark baseline)
# ---------------------------------------------------------------------------


def _forward_kinematics_serial(theta_body: np.ndarray, ds: float, mid_idx: int
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Frame-by-frame forward kinematics — reference baseline for benchmark.

    Delegates per-frame computation to skeleton.tangent_to_xy so the formula
    is always identical to the canonical implementation.
    """
    from skeleton import tangent_to_xy
    n_fr  = theta_body.shape[0]
    n_pts = theta_body.shape[1] + 1
    xb = np.zeros((n_fr, n_pts), dtype=np.float32)
    yb = np.zeros((n_fr, n_pts), dtype=np.float32)
    for i in range(n_fr):
        x, y = tangent_to_xy(theta_body[i], ds, mid_idx)
        xb[i] = x.astype(np.float32)
        yb[i] = y.astype(np.float32)
    return xb, yb


# ---------------------------------------------------------------------------
# Backend classes
# ---------------------------------------------------------------------------


class NumPySerialBackend:
    """Single-threaded NumPy, frame-by-frame.  Always available; benchmark baseline."""

    name = "numpy_serial"

    def compute_skeletons(self, theta_body: np.ndarray, ds: float, mid_idx: int
                          ) -> tuple[np.ndarray, np.ndarray]:
        return _forward_kinematics_serial(theta_body, ds, mid_idx)


class NumPyBatchBackend:
    """Vectorised NumPy — no Python loop over frames.

    Delegates to ``skeleton.tangent_to_xy_batch`` which vectorises all trig
    and prefix-sums in a single C-level NumPy pass.  Typically 50–80× faster
    than NumPySerial on large batches.
    """

    name = "numpy_batch"

    def compute_skeletons(self, theta_body: np.ndarray, ds: float, mid_idx: int
                          ) -> tuple[np.ndarray, np.ndarray]:
        return tangent_to_xy_batch(theta_body, ds, mid_idx)


class NumPyMultiprocessBackend:
    """joblib Parallel over frame chunks.

    Effective when n_frames > 500 and the host has ≥4 physical cores.
    For smaller batches, NumPyBatch is faster (no process-spawn overhead).
    """

    name = "numpy_mp"

    def __init__(self, n_jobs: int = -1, chunk_size: int | None = None) -> None:
        import multiprocessing
        self.n_jobs    = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        self.chunk_size = chunk_size

    def compute_skeletons(self, theta_body: np.ndarray, ds: float, mid_idx: int
                          ) -> tuple[np.ndarray, np.ndarray]:
        from joblib import Parallel, delayed  # type: ignore
        n_fr   = theta_body.shape[0]
        chunk  = self.chunk_size or max(1, n_fr // (self.n_jobs * 4))
        slices = [slice(i, min(i + chunk, n_fr)) for i in range(0, n_fr, chunk)]
        results = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(tangent_to_xy_batch)(theta_body[sl], ds, mid_idx)
            for sl in slices
        )
        return (np.concatenate([r[0] for r in results], axis=0),
                np.concatenate([r[1] for r in results], axis=0))


class OpenCLBackend:
    """PyOpenCL backend — one GPU work-item per frame, parallel across frames.

    The OpenCL kernel assigns one work-item per animation frame.  Within each
    work-item the 49-point cumsum is serial (49 iterations), which is trivially
    fast.  The parallelism is across *frames*: on a V100S (6912 CUDA cores) or
    Apple GPU (thousands of Metal ALUs via the Apple OpenCL ICD) all frames
    run concurrently.

    Device selection:  uses the first available GPU device.  If none found,
    falls back to the first CPU device (still JIT-compiled via OpenCL driver).

    Requires:  pip install pyopencl
    Platforms:
      NVIDIA CUDA  — NVIDIA OpenCL ICD (bundled with CUDA toolkit)
      AMD          — ROCm OpenCL  or  AMDGPU-Pro
      Intel        — Intel OpenCL Runtime  or  Intel GPU OpenCL
      macOS        — Apple OpenCL (deprecated in macOS 12 but still functional)
    """

    name = "opencl_gpu"

    _KERNEL = r"""
    #define HALF_PI 1.5707963267948966f

    __kernel void tangent_to_xy_kernel(
        __global const float* theta_body,  /* (n_frames * n_seg)  row-major  */
        __global       float* xs,          /* (n_frames * n_pts) out         */
        __global       float* ys,
        const int n_seg,
        const int n_pts,
        const int mid_idx,
        const float ds
    ) {
        const int frame    = get_global_id(0);
        const int base_th  = frame * n_seg;
        const int base_xy  = frame * n_pts;

        float x = 0.0f, y = 0.0f;
        xs[base_xy] = 0.0f;
        ys[base_xy] = 0.0f;

        for (int j = 0; j < n_seg; j++) {
            float th = theta_body[base_th + j] - HALF_PI;
            x += ds * cos(th);
            y += ds * sin(th);
            xs[base_xy + j + 1] = x;
            ys[base_xy + j + 1] = y;
        }

        /* Centre skeleton at mid_idx */
        float x_mid = xs[base_xy + mid_idx];
        float y_mid = ys[base_xy + mid_idx];
        for (int j = 0; j < n_pts; j++) {
            xs[base_xy + j] -= x_mid;
            ys[base_xy + j] -= y_mid;
        }
    }
    """

    def __init__(self, platform_idx: int = 0, device_idx: int | None = None) -> None:
        try:
            import pyopencl as cl  # type: ignore
        except ImportError as exc:
            raise ImportError("OpenCLBackend requires pyopencl: pip install pyopencl") from exc

        self._cl = cl
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms found")

        platform = platforms[platform_idx]
        all_devices = platform.get_devices()
        if not all_devices:
            raise RuntimeError(f"No OpenCL devices on platform '{platform.name}'")

        # Prefer GPU; fall back to first available device
        if device_idx is not None:
            dev = all_devices[device_idx]
        else:
            gpu_devs = [d for d in all_devices if d.type == cl.device_type.GPU]
            dev = gpu_devs[0] if gpu_devs else all_devices[0]

        dev_type = cl.device_type.to_string(dev.type)
        self.name = "opencl_gpu" if dev.type == cl.device_type.GPU else "opencl_cpu"
        self._device_name = f"{dev.name.strip()} ({dev_type}, {platform.name.strip()})"

        self._ctx   = cl.Context([dev])
        self._queue = cl.CommandQueue(self._ctx)
        self._prog  = cl.Program(self._ctx, self._KERNEL).build()

    @property
    def device_name(self) -> str:
        return self._device_name

    def compute_skeletons(self, theta_body: np.ndarray, ds: float, mid_idx: int
                          ) -> tuple[np.ndarray, np.ndarray]:
        cl  = self._cl
        mf  = cl.mem_flags
        th32 = np.ascontiguousarray(theta_body, dtype=np.float32)
        n_fr, n_seg = th32.shape
        n_pts = n_seg + 1
        nbytes = th32.itemsize * n_fr * n_pts

        th_buf = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=th32)
        xs_buf = cl.Buffer(self._ctx, mf.WRITE_ONLY, nbytes)
        ys_buf = cl.Buffer(self._ctx, mf.WRITE_ONLY, nbytes)

        self._prog.tangent_to_xy_kernel(
            self._queue, (n_fr,), None,
            th_buf, xs_buf, ys_buf,
            np.int32(n_seg), np.int32(n_pts), np.int32(mid_idx), np.float32(ds),
        )
        self._queue.finish()

        xs_out = np.empty((n_fr, n_pts), dtype=np.float32)
        ys_out = np.empty((n_fr, n_pts), dtype=np.float32)
        cl.enqueue_copy(self._queue, xs_out, xs_buf)
        cl.enqueue_copy(self._queue, ys_out, ys_buf)
        self._queue.finish()
        return xs_out, ys_out


class JAXBackend:
    """JAX XLA-compiled forward kinematics.

    Implements the same formula as ``skeleton.tangent_to_xy_batch``:

        th = theta_body − π/2
        xs = cumsum([0, cos(th)*ds, ...], axis=1)
        ys = cumsum([0, sin(th)*ds, ...], axis=1)
        xs −= xs[:, mid_idx]

    Uses ``jax.lax.dynamic_slice`` for the midpoint subtraction so that
    ``mid_idx`` can be a Python int (not a JAX traced value) without
    triggering recompilation.

    First call includes JIT compilation (~0.5 s); subsequent calls are fast.
    """

    name = "jax_cpu"

    def __init__(self) -> None:
        import jax            # type: ignore
        import jax.numpy as jnp  # type: ignore

        self._jax = jax
        self._jnp = jnp

        devs     = jax.devices()
        dev_strs = [str(d).lower() for d in devs]
        if any("cuda" in d or "gpu" in d for d in dev_strs):
            self.name = "jax_cuda"
        elif any("tpu" in d for d in dev_strs):
            self.name = "jax_tpu"
        elif any("metal" in d or "mps" in d for d in dev_strs):
            self.name = "jax_mps"

        @jax.jit
        def _fk_ds(theta_body: "jax.Array", ds: float, mid: int
                   ) -> "tuple[jax.Array, jax.Array]":
            # theta_body[i,j] — absolute tangent angles (theta_body convention)
            # th = theta_body − π/2  is the lab heading for each segment.
            # No cumsum on angles — cumsum is only on position steps.
            n_fr = theta_body.shape[0]
            th     = theta_body - jnp.pi / 2
            cos_th = jnp.cos(th)
            sin_th = jnp.sin(th)
            zeros  = jnp.zeros((n_fr, 1), dtype=jnp.float32)
            xs = jnp.cumsum(jnp.concatenate([zeros, cos_th * ds], axis=1), axis=1)
            ys = jnp.cumsum(jnp.concatenate([zeros, sin_th * ds], axis=1), axis=1)
            x_mid = jax.lax.dynamic_slice(xs, (0, mid), (n_fr, 1))
            y_mid = jax.lax.dynamic_slice(ys, (0, mid), (n_fr, 1))
            return xs - x_mid, ys - y_mid

        self._fk = _fk_ds

    def compute_skeletons(self, theta_body: np.ndarray, ds: float, mid_idx: int
                          ) -> tuple[np.ndarray, np.ndarray]:
        jnp    = self._jnp
        t_jax  = jnp.array(theta_body.astype(np.float32))
        xs_j, ys_j = self._fk(t_jax, ds, mid_idx)
        self._jax.block_until_ready(xs_j)
        return np.asarray(xs_j), np.asarray(ys_j)


# ---------------------------------------------------------------------------
# Automatic backend selection
# ---------------------------------------------------------------------------


def select_backend(
    profile: "HardwareProfile | None" = None,
    force: str | None = None,
) -> "NumPySerialBackend | NumPyBatchBackend | NumPyMultiprocessBackend | OpenCLBackend | JAXBackend":
    """Return the best available backend for the current host.

    Priority (highest to lowest):
        jax_cuda / jax_gpu → jax_mps → opencl → jax_cpu → numpy_mp → numpy_batch

    Args:
        profile: HardwareProfile from detect_hardware().  Detected lazily if None.
        force:   One of 'numpy_serial', 'numpy_batch', 'numpy_mp',
                 'opencl', 'jax_cpu', 'jax_cuda', 'jax_mps'.
    """
    if force is not None:
        return _create_by_name(force, profile)
    if profile is None:
        from hardware import detect_hardware
        profile = detect_hardware()
    return _create_by_name(profile.recommended_backend, profile)


def _create_by_name(name: str, profile: "HardwareProfile | None") -> object:
    if name in ("jax_cpu", "jax_cuda", "jax_gpu", "jax_mps", "jax_tpu"):
        try:
            return JAXBackend()
        except ImportError:
            pass
        name = "opencl"   # JAX unavailable → try OpenCL next

    if name == "opencl":
        try:
            return OpenCLBackend()
        except Exception:
            pass
        name = "numpy_mp"  # OpenCL unavailable → CPU multiprocess

    if name == "numpy_mp":
        import multiprocessing
        ncpu = (profile.cpu_physical_cores if profile else multiprocessing.cpu_count())
        if ncpu <= 1:
            return NumPyBatchBackend()
        return NumPyMultiprocessBackend(n_jobs=ncpu)

    if name in ("numpy_batch", "cuda", "mps"):
        return NumPyBatchBackend()

    return NumPySerialBackend()


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------


def benchmark_backends(
    theta_body: np.ndarray,
    ds: float,
    mid_idx: int,
    backends: "list[str] | None" = None,
    n_warmup: int = 1,
    n_repeat: int = 3,
) -> list[dict]:
    """Time each backend on theta_body and return a list of result dicts.

    Returns list of dicts: backend, n_frames, time_s, fps, speedup_vs_serial.

    Deduplication: if a requested backend (e.g. jax_cpu) is unavailable and
    silently falls back to one already measured (e.g. opencl), the duplicate
    is skipped rather than reported twice.
    """
    if backends is None:
        backends = ["numpy_serial", "numpy_batch", "numpy_mp", "opencl", "jax_cpu"]

    n_frames = theta_body.shape[0]
    results: list[dict] = []
    serial_fps: float | None = None
    seen_classes: set[type] = set()

    for name in backends:
        try:
            b = _create_by_name(name, None)
        except Exception as e:
            results.append({"backend": name, "n_frames": n_frames,
                            "time_s": None, "fps": None,
                            "speedup_vs_serial": None, "error": str(e)})
            continue

        # Skip if this backend class already ran under a different requested name
        if type(b) in seen_classes:
            continue
        seen_classes.add(type(b))

        try:
            for _ in range(n_warmup):
                b.compute_skeletons(theta_body, ds, mid_idx)
        except Exception as e:
            results.append({"backend": name, "n_frames": n_frames,
                            "time_s": None, "fps": None,
                            "speedup_vs_serial": None, "error": str(e)})
            continue

        times: list[float] = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            b.compute_skeletons(theta_body, ds, mid_idx)
            times.append(time.perf_counter() - t0)

        best_s  = min(times)
        fps     = n_frames / best_s
        if serial_fps is None and "serial" in name:
            serial_fps = fps
        speedup = fps / serial_fps if serial_fps else None

        results.append({
            "backend": b.name,
            "n_frames": n_frames,
            "time_s": round(best_s, 4),
            "fps": round(fps, 1),
            "speedup_vs_serial": round(speedup, 2) if speedup else None,
        })

    return results

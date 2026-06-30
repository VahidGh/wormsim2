# wormsim2 v0.11 — Local GPU Installation Guide (Docker + CUDA)

> **Platform:** Linux with NVIDIA GPU (CUDA 12.4 recommended).
> macOS users: no NVIDIA GPU support; use the local CPU guide or HPC SLURM.

---

## Prerequisites

| Tool | Notes |
|---|---|
| Docker ≥ 24.0 | `sudo apt install docker.io` |
| NVIDIA GPU driver ≥ 525 | `nvidia-smi` should work |
| NVIDIA Container Toolkit | enables `--gpus all` flag |
| Git | clone repo |

### Install NVIDIA Container Toolkit (Ubuntu)

```bash
# Add NVIDIA GPG key and repo
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor \
    -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
# → Should show your GPU (e.g. "NVIDIA GeForce RTX 4090")
```

---

## 1. Clone and build the GPU image

```bash
git clone https://github.com/VahidGh/wormsim2.git
cd wormsim2

docker build -f docker/Dockerfile.wormsim2-gpu -t wormsim2-gpu .
# ≈ 5–10 minutes first run (downloads CUDA base + JAX[cuda12])
```

---

## 2. Verify GPU is visible inside the container

```bash
docker run --rm --gpus all wormsim2-gpu bash -c "
  nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader &&
  python3 -c \"import jax; print('JAX devices:', jax.devices())\"
"
# → NVIDIA GeForce RTX ...  |  24576 MiB  |  8.9
#    JAX devices: [cuda(id=0)]
```

---

## 3. Run hardware detection inside container

```bash
docker run --rm --gpus all -v "$(pwd)":/wormsim2 wormsim2-gpu bash -c "
  cd /wormsim2
  python3 -c \"
import sys; sys.path.insert(0, 'src/python')
from hardware import detect_hardware
print(detect_hardware().summary())
\"
"
```

Expected (RTX 4090 example):
```
CPU:         Intel(R) Core(TM) i9-13900K (32 logical / 24 physical cores)
GPU [CUDA]:  NVIDIA GeForce RTX 4090  24564 MB VRAM
CUDA:        yes
MPS:         no
JAX:         yes — cuda(id=0)
OpenMP(C++): yes
Recommended: cuda
```

---

## 4. Run CV-11 benchmark

```bash
docker run --rm --gpus all -v "$(pwd)":/wormsim2 wormsim2-gpu bash -c "
  cd /wormsim2 &&
  cmake -B build_gpu -GNinja -DCMAKE_BUILD_TYPE=Release -DWORMSIM2_OPENMP=ON &&
  cmake --build build_gpu -j\$(nproc) &&
  python3 -c \"
import sys, numpy as np; sys.path.insert(0,'src/python')
from backends import benchmark_backends
theta = np.random.randn(10000, 48).astype('float32') * 0.3
backends = ['numpy_serial','numpy_batch','opencl','jax_cpu','jax_cuda']
results = benchmark_backends(theta, 1/48, 24, backends=backends)
for r in results: print(r['backend'], r.get('fps','ERR'), 'fps', r.get('error',''))
\"
"
```

---

## 5. OpenCL backend (NVIDIA / AMD / Intel / Apple)

From **v0.11.0** wormsim2 ships a native `OpenCLBackend` in `src/python/backends.py`.
The kernel (`tangent_to_xy_kernel`) assigns one OpenCL work-item per animation frame,
so all frames are computed in parallel on the GPU's ALUs.

### Install OpenCL runtime

| GPU vendor | Runtime to install |
|---|---|
| AMD RX/RDNA | `sudo apt install rocm-opencl-runtime` |
| Intel iGPU (Gen9+) | `sudo apt install beignet-opencl-icd` (or Intel Compute Runtime) |
| NVIDIA (CUDA) | NVIDIA OpenCL ICD bundled with CUDA toolkit — already in the image |
| macOS (Apple GPU) | Apple OpenCL ICD built into macOS (deprecated in 12, still functional) |

```bash
pip install pyopencl

# Verify platforms and devices:
python3 -c "
import pyopencl as cl
for p in cl.get_platforms():
    for d in p.get_devices():
        print(p.name, '|', d.name, '|', cl.device_type.to_string(d.type))
"
```

### Run with OpenCL inside the Docker container

The GPU image (`wormsim2-gpu`) already includes `pyopencl`.

```bash
# NVIDIA — CUDA + OpenCL ICD:
docker run --rm --gpus all -v "$(pwd)":/wormsim2 wormsim2-gpu bash -c "
  cd /wormsim2
  python3 -c \"
import sys; sys.path.insert(0,'src/python')
from backends import OpenCLBackend
b = OpenCLBackend()
print('OpenCL device:', b.device_name)
import numpy as np
theta = np.random.randn(1000, 48).astype('float32')
xs, ys = b.compute_skeletons(theta, ds=1/48, mid_idx=24)
print('xs shape:', xs.shape)   # → (1000, 49)
\"
"

# AMD GPU (ROCm OpenCL) — use /dev/dri device pass-through:
docker run --rm --device=/dev/dri -v "$(pwd)":/wormsim2 wormsim2-gpu bash -c "
  cd /wormsim2
  python3 -c 'import sys; sys.path.insert(0,\"src/python\"); from backends import OpenCLBackend; b=OpenCLBackend(); print(b.device_name)'
"
```

When `hardware.py` detects OpenCL, `recommended_backend` is set to `"opencl"` and
`select_backend()` will automatically return an `OpenCLBackend` instance.

---

## 6. No GPU passthrough via Docker? Run natively on the host

This whole guide assumes `--gpus all` (NVIDIA Container Toolkit) or `--device=/dev/dri`
(AMD/Intel) GPU passthrough into the container. **Neither exists on macOS** — Docker
Desktop on Mac runs containers inside a Linux VM with no path to the host's GPU at all
(no CUDA on macOS since 10.2; no Metal inside a Linux container). If you're on a Mac, or
any host where Docker GPU passthrough isn't available, skip the container entirely and
run the Python backend natively against the host's own OpenCL/Metal driver.

This is exactly what we did to get a real GPU benchmark on a non-NVIDIA dev laptop
(Intel Core i5-8257U, **Intel Iris Plus 645** integrated GPU, no Docker involved):

**1. Install `pyopencl` on the host** (not in any container):

```bash
pip3 install pyopencl
```

**2. Confirm the host driver exposes the GPU** — on macOS this is Apple's built-in
OpenCL ICD (deprecated since macOS 12 but still functional); on Linux it's whatever
vendor runtime is installed (`beignet`/Intel Compute Runtime, ROCm, etc.):

```python
import pyopencl as cl
for p in cl.get_platforms():
    for d in p.get_devices():
        print(p.name, '|', d.name, '|', cl.device_type.to_string(d.type),
              '| CUs:', d.max_compute_units)
```

```
Apple | Intel(R) Core(TM) i5-8257U CPU @ 1.40GHz | CPU | CUs: 8
Apple | Intel(R) Iris(TM) Plus Graphics 645      | GPU | CUs: 48
```

**3. Run the benchmark natively** — `backends.py` needs no container, no CUDA, no
special build step; `OpenCLBackend` auto-selects the GPU device:

```python
import sys; sys.path.insert(0, 'src/python')
from backends import benchmark_backends
import numpy as np

# real 600 s N2 WCON recording, 17,226 valid frames (see notebooks/project_tour.ipynb CV-11.1)
theta_body = ...  # loaded from data/raw/wcon_raw/N2_*.wcon
results = benchmark_backends(theta_body, ds=1/48, mid_idx=24,
                              backends=['numpy_serial','numpy_batch','numpy_mp','opencl','jax_cpu'])
```

**Actual result of this run** (captured on the dev laptop, no Docker):

```
numpy_serial                  24,658 fps   1.0x
numpy_batch                1,322,804 fps  53.6x
numpy_mp                     420,529 fps  17.1x
opencl_gpu (Intel Iris 645)   994,013 fps  40.3x
```

The Intel Iris 645 — a weak integrated GPU with no dedicated VRAM — still gets 40×
over serial once the workload is large enough (17,226 frames) to amortize OpenCL's
kernel-launch overhead; at 200 frames the same GPU only scored 2.7× (see
`README.md` benchmark table). The takeaway: **GPU passthrough through Docker is a
convenience, not a requirement** — the same `backends.py` code path runs identically
inside or outside a container, on any OpenCL-capable device, NVIDIA or not.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `docker: unknown flag: --gpus` | Install nvidia-container-toolkit (Step 0) |
| `jax: no CUDA devices found` | Pass `--gpus all` to `docker run`; check `nvidia-smi` works |
| JAX warns about CUDA version mismatch | Ensure driver ≥ 525 and image base matches driver |
| macOS with NVIDIA (older Mac Pro) | No CUDA support on macOS since CUDA 10.2; use HPC guide |

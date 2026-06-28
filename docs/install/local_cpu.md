# wormsim2 v0.11 — Local CPU Installation Guide

> **Platform:** macOS (Intel or Apple Silicon) · Linux (Ubuntu 22.04+) · Windows WSL2
> **No GPU required.**  All CV-11 benchmarks run on CPU only.

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | ≥ 3.11 | `brew install python` / `sudo apt install python3` |
| CMake  | ≥ 3.22 | `brew install cmake` / `sudo apt install cmake` |
| GCC or Clang | ≥ 11 (C++20) | system or `sudo apt install build-essential` |
| Git | any | system |

Optional (for parallel backends):

| Tool | Notes |
|---|---|
| OpenMP (`libgomp`) | `sudo apt install libgomp1 libomp-dev` — enables `#pragma omp` in C++ gate loop |
| joblib | installed via Python below — multiprocessing backend |

---

## 1. Clone the repository

```bash
git clone https://github.com/VahidGh/wormsim2.git
cd wormsim2
```

---

## 2. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r docs/requirements/requirements_cpu.txt
```

**`requirements_cpu.txt`** (create if not present):
```text
numpy>=1.26
scipy>=1.12
matplotlib>=3.8
pandas>=2.2
plotly>=5.20
kaleido>=1.3
pillow>=10.0
joblib>=1.3
jax[cpu]>=0.4.30
```

Verify:
```bash
python3 -c "from hardware import detect_hardware; print(detect_hardware().summary())"
# → CPU:  Intel(R) Core(TM) i5-8257U …  joblib jobs: 4  Recommended: jax_cpu
```

---

## 3. Build C++ with OpenMP

```bash
cmake -B build -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DWORMSIM2_OPENMP=ON \
      -DWORMSIM2_BUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

Expected output: all CTest tests PASS.

---

## 4. Run CV-11 benchmark notebook

```bash
cd notebooks
jupyter lab project_tour.ipynb
```

Navigate to **CV-11.1** and run all cells.

---

## 5. Verify hardware detection output

Expected on a 4-core Intel Mac (dev machine):

```
CPU:         Intel(R) Core(TM) i5-8257U CPU @ 1.40GHz  (8 logical / 4 physical cores)
RAM:         8.0 GB
GPU:         none detected          ← or Apple GPU (Metal) on M-chip Macs
CUDA:        no
MPS:         no (Intel) / yes (Apple Silicon)
OpenCL:      no
JAX:         yes — TFRT_CPU_0
OpenMP(C++): yes                    ← if OpenMP build succeeded
joblib jobs: 4
Recommended: jax_cpu
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `cmake: OpenMP not found` | macOS: `brew install libomp && cmake ... -DOpenMP_CXX_FLAGS="-Xclang -fopenmp" -DOpenMP_CXX_LIB_NAMES="omp" -DOpenMP_omp_LIBRARY=$(brew --prefix libomp)/lib/libomp.dylib` |
| `import jax` fails | `pip install --upgrade "jax[cpu]"` |
| `joblib` multiprocessing slower than serial | Normal for n < 500 frames; use `numpy_batch` backend |

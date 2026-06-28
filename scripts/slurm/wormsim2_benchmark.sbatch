#!/bin/bash
# wormsim2 v0.11.0 — benchmark SLURM job (CINECA G100)
# ─────────────────────────────────────────────────────
# Runs the CV-11 benchmark (hardware detection + skeleton kinematics speedup
# on CPU-only, OpenMP, and JAX-CUDA) on the G100 Tesla V100S cluster.
#
# Submit:
#   sbatch scripts/slurm/wormsim2_benchmark.sbatch
#
# Monitor:
#   squeue -u $USER
#   tail -f wormsim2_bench_<jobid>.out
#
# Fetch results back to laptop:
#   scp <user>@login.g100.cineca.it:$WORK/wormsim2/wormsim2_bench_*.out ./results_slurm/
#
# References:
#   docs/install/hpc_slurm.md — full G100 onboarding guide
#   CINECA G100 specs: 2 × IBM POWER9, 4 × V100S (32 GB each), 384 GB RAM
#   AMSC Lec 10: Amdahl / Gustafson, embarrassingly parallel decomposition
#   AMSC Lec 12: OpenMP — OMP_NUM_THREADS, schedule(static)

#SBATCH --job-name=wormsim2_bench
#SBATCH --account=<your_account>           # replace: e.g. train_amsc2025
#SBATCH --partition=g100_usr_prod
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32                 # 2×IBM POWER9 = 32 cores/node
#SBATCH --gres=gpu:1                       # 1 Tesla V100S (32 GB)
#SBATCH --mem=64G
#SBATCH --output=wormsim2_bench_%j.out
#SBATCH --error=wormsim2_bench_%j.err

# ── Environment ──────────────────────────────────────────────────────────────
module load profile/deeplrn
module load cuda/12.4
module load gnu/11.3.0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export JAX_PLATFORMS=cuda,cpu
export PYTHONPATH=$WORK/wormsim2/src/python:$PYTHONPATH

echo "=== wormsim2 v0.11.0 Benchmark ==="
echo "Host:       $(hostname)"
echo "Date:       $(date)"
echo "Job ID:     $SLURM_JOB_ID"
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo ""

cd $WORK/wormsim2

# ── Step 1: Build C++ with OpenMP ────────────────────────────────────────────
echo "--- Building C++ (OpenMP ON) ---"
cmake -B build_hpc -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DWORMSIM2_OPENMP=ON \
      -DWORMSIM2_BUILD_TESTS=OFF
cmake --build build_hpc -j${SLURM_CPUS_PER_TASK}
echo "C++ build: OK"

# ── Step 2: Python hardware detection ────────────────────────────────────────
echo ""
echo "--- Hardware Detection ---"
python3 -c "
import sys; sys.path.insert(0, 'src/python')
from hardware import detect_hardware
hw = detect_hardware()
print(hw.summary())
"

# ── Step 3: Python backend benchmark ─────────────────────────────────────────
echo ""
echo "--- Backend Benchmark (n=10000 frames = 345 seconds at 29fps) ---"
python3 - << 'PYEOF'
import sys, numpy as np
sys.path.insert(0, 'src/python')
from backends import benchmark_backends

np.random.seed(42)
# 10,000 frames ~ 5.7 minutes at 29fps — typical long recording
theta = np.random.randn(10000, 48).astype(np.float32) * 0.3

backends = ["numpy_serial", "numpy_batch", "numpy_mp"]
try:
    from backends import OpenCLBackend
    OpenCLBackend()           # probe: raises if OpenCL ICD unavailable
    backends.append("opencl")
except Exception:
    print("OpenCL: not available on this node")
try:
    import jax
    backends.append("jax_cpu")
    jax.devices("cuda")
    backends.append("jax_cuda")
except Exception:
    pass

results = benchmark_backends(theta, ds=1/48, mid_idx=24, n_repeat=3, backends=backends)
serial_fps = next(r["fps"] for r in results if r.get("fps") and "serial" in r["backend"])

print(f"\n{'Backend':<22} {'fps':>12}  {'time_s':>8}  {'speedup':>10}")
print("-" * 60)
for r in results:
    if r.get("error"):
        print(f"{r['backend']:<22} ERROR: {r['error'][:40]}")
        continue
    sp_str = f"{r['fps']/serial_fps:.1f}×" if r["fps"] and serial_fps else "—"
    print(f"{r['backend']:<22} {r['fps']:>12.0f}  {r['time_s']:>8.4f}  {sp_str:>10}")
print()

# Amdahl's law estimate for V100S (6912 CUDA cores vs 1 serial Python thread)
import math
f_parallel = 0.97   # ~97% of FK loop is parallel (serial: axis-1 cumsum init)
n_cuda     = 6912
speedup_theoretical = 1.0 / ((1 - f_parallel) + f_parallel / n_cuda)
print(f"Amdahl estimate (f_par={f_parallel:.0%}, N_cuda={n_cuda}):")
print(f"  Theoretical max speedup:  {speedup_theoretical:.0f}×")
print(f"  Practical JAX-CUDA est.:  ~200–500× (bandwidth-bound at n=10000)")
PYEOF

echo ""
echo "=== Benchmark complete ==="
echo "Job $SLURM_JOB_ID finished at $(date)"

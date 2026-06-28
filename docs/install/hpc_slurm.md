# wormsim2 v0.11 — HPC / SLURM Deployment Guide

> **Target cluster:** CINECA G100  (2 × IBM POWER9 + 4 × Tesla V100S per node)
> **Container:** Apptainer 1.1+  (formerly Singularity)
>

---

## 1. Prerequisites (laptop side)

| Tool                   | Notes                                                 |
| ---------------------- | ----------------------------------------------------- |
| `gh` CLI             | `brew install gh && gh auth login`                  |
| `ssh` access to G100 | CINECA account required; step-login via HPC SSH proxy |
| `scp`                | for transferring results back                         |

CINECA G100 SSH:

```bash
# Replace with your CINECA username
ssh <user>@login.g100.cineca.it
```

---

## 2. Filesystem on G100

| Path         | Description                                              |
| ------------ | -------------------------------------------------------- |
| `$HOME`    | small quota (50 GB); for scripts and config only         |
| `$WORK`    | large quota (1 TB); put simulation data and results here |
| `$SCRATCH` | ephemeral; fast I/O, auto-deleted after 30 days          |

Working directory:

```bash
mkdir -p $WORK/wormsim2
cd $WORK/wormsim2
```

---

## 3. Transfer repository to G100

**Option A: git clone (if internet access is available on login node)**

```bash
git clone https://github.com/VahidGh/wormsim2.git $WORK/wormsim2
```

**Option B: rsync from laptop**

```bash
rsync -avz --exclude='.git' --exclude='build*' --exclude='__pycache__' \
    /Users/vghayoomie/git/wormsim2/ \
    <user>@login.g100.cineca.it:$WORK/wormsim2/
```

---

## 4. Build the Apptainer image

**Option A: CI-built image (recommended)**

The GitHub Actions workflow (`.github/workflows/slurm-deploy.yml`) builds `wormsim2.sif`
automatically on every version tag and uploads it as a GitHub Release asset.

```bash
# On the G100 login node:
cd $WORK/wormsim2
gh release download v0.11.0 --pattern '*.sif' --dir .
# → wormsim2.sif  (~2 GB)
```

**Option B: Build locally on the cluster (requires sudo/root on the node)**

```bash
# G100 login node has Apptainer available:
module load apptainer/1.3.4
apptainer build wormsim2.sif singularity/wormsim2.def
```

---

## 5. Submit the benchmark job

```bash
# On G100 login node:
cd $WORK/wormsim2
sbatch scripts/slurm/wormsim2_benchmark.sbatch
```

Edit `#SBATCH --account=<your_account>` with your project code before submitting.

Monitor:

```bash
squeue -u $USER
tail -f wormsim2_bench_<jobid>.out
```

---

## 6. Retrieve results

```bash
# From your laptop:
scp "<user>@login.g100.cineca.it:$WORK/wormsim2/wormsim2_bench_*.out" ./results_slurm/
```

---

## 7. G100 hardware specs (for Amdahl's law estimates)

| Component         | Spec                                                |
| ----------------- | --------------------------------------------------- |
| CPU               | 2 × IBM POWER9 @ 3.1 GHz (48 total cores/node)     |
| RAM               | 384 GB DDR4                                         |
| GPU               | 4 × Tesla V100S (32 GB VRAM, 6912 CUDA cores each) |
| Interconnect      | NVLink 2.0 (NVSwitch), InfiniBand HDR100            |
| CUDA              | 12.4                                                |
| OpenMP            | gcc 11.3 / OpenMP 5.0                               |
| Storage (SCRATCH) | IBM Spectrum Scale (GPFS), ~100 GB/s aggregate      |

**Parallelism budget for wormsim2:**

| Mode                                      | Units                | Parallelism                                  |
| ----------------------------------------- | -------------------- | -------------------------------------------- |
| C++ OpenMP (gate loop, 302 neurons)       | 48 CPU threads/job   | 302 neurons → ~20× speedup (Amdahl f=0.95) |
| Python JAX-CUDA (skeleton FK, 10K frames) | 1 V100S (6912 cores) | ~200-500× vs NumPy serial                   |
| MPI multi-worm (future v0.12)             | N nodes × 48 CPUs   | ~N× for independent worm ensemble           |

**Amdahl's law speedup example (V100S vs serial Python):**

```
f_parallel = 0.97  (FK loop — 97% parallelizable; serial: data-load + centring init)
N_cuda = 6912

speedup_max = 1 / ((1 - f_par) + f_par / N_cuda)
           ≈ 1 / (0.03 + 0.97/6912)
           ≈ 33×   (theoretical)

Practical (bandwidth-limited at n=10000 frames, 48 segments × float32):
   data transfer: 10000 × 49 × 2 × 4B ≈ 3.9 MB  → negligible on NVLink
   expected speedup: ~200–500×  (JAX XLA vs NumPy serial)
```

---

## 8. Run via Apptainer (binding the repo)

```bash
apptainer run --nv \
    --bind $WORK/wormsim2:/opt/wormsim2 \
    wormsim2.sif \
    python3 /opt/wormsim2/scripts/benchmark.py --n-frames 10000
```

The `--nv` flag exposes the host NVIDIA driver to the container.

---

## 9. SLURM GPU partition list (G100)

| Partition            | Max time | GPUs          | CPUs/node |
| -------------------- | -------- | ------------- | --------- |
| `g100_usr_prod`    | 24h      | up to 4 V100S | 48        |
| `g100_metabo_prod` | 4h       | up to 4 V100S | 48        |
| `g100_all_serial`  | 4h       | 0             | 48        |

Use `g100_usr_prod` for CV-11 benchmark runs.

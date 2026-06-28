# wormsim2 v0.11 — GitHub Actions CI Guide

> Covers all three CI workflows active in this repo.
> Reference: Lec 11 (11.CI_CD.pdf) — triggering events, evidence artifacts.

---

## Workflow overview

| Workflow file             | Trigger                 | What it does                                           |
| ------------------------- | ----------------------- | ------------------------------------------------------ |
| `ci-baseline.yml`       | push / PR / manual      | C++ build + tests + static analysis + Python benchmark |
| `slurm-deploy.yml`      | tag push`v*` / manual | Apptainer build → sign → attach to GitHub Release    |
| `auto-release.yml`      | tag push`v*`          | Extract CHANGELOG section → GitHub Release body       |
| `backfill-releases.yml` | manual                  | Backfill missing release entries for past tags         |

---

## 1. C++ + benchmark CI (`ci-baseline.yml`)

Runs on every push and PR.  Jobs:

### Job `build-test`

1. **C++ build with OpenMP** — `cmake -DWORMSIM2_OPENMP=ON`
2. **CTest** — all unit/integration tests
3. **cppcheck** — static analysis
4. **clang-tidy** — style + correctness
5. **Python environment** — installs `jax[cpu]`, numpy, scipy, joblib
6. **Hardware detection** — runs `detect_hardware()`, records output
7. **Backend benchmark** — 5 backends × 1000 frames; asserts numpy_batch speedup ≥ 30×
8. **Evidence upload** — all output files uploaded as `ci-evidence` artifact

### Job `benchmark-gpu` (self-hosted, optional)

Enabled only when a self-hosted runner with label `gpu` is registered:

```yaml
runs-on: [self-hosted, gpu]
```

This job runs the full JAX-CUDA benchmark on 10,000 frames and asserts
`jax_cuda` fps ≥ 100 × `numpy_serial` fps.

To register a self-hosted GPU runner:

```bash
# On the GPU machine:
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.316.1.tar.gz -L \
    https://github.com/actions/runner/releases/download/v2.316.1/actions-runner-linux-x64-2.316.1.tar.gz
tar xzf *.tar.gz
./config.sh --url https://github.com/VahidGh/wormsim2 --token <token>
./run.sh &
```

---

## 2. Apptainer build + release CI (`slurm-deploy.yml`)

Triggered on tag push `v*`.  Steps:

1. Install Apptainer 1.3.4 via `eWaterCycle/setup-apptainer@v2`
2. `apptainer build wormsim2.sif singularity/wormsim2.def`  (≈ 15–20 min)
3. GPG sign the `.sif`:
   - If `GPG_PRIVATE_KEY` secret is set: real GPG signing
   - Otherwise: sha256 checksum only (CI-without-key fallback)
4. `apptainer inspect` + smoke `apptainer run --dry-run`
5. Upload `wormsim2.sif` + `wormsim2-sif-info.txt` as release asset

### Required GitHub secrets

| Secret              | Purpose                                    |
| ------------------- | ------------------------------------------ |
| `GPG_PRIVATE_KEY` | optional — enables GPG signing of`.sif` |
| `GPG_PASSPHRASE`  | required if GPG key is set                 |

Set via: GitHub → repo → Settings → Secrets and variables → Actions.

---

## 3. How to manually trigger a workflow

```bash
# Trigger ci-baseline on current branch:
gh workflow run ci-baseline.yml

# Trigger slurm-deploy on tag v0.11.0:
gh workflow run slurm-deploy.yml --ref v0.11.0
```

---

## 4. Evidence artifacts

After each CI run, download artifacts:

```bash
gh run download --name ci-evidence -D ci-evidence/
ls ci-evidence/
# benchmark-output.txt  ctest-output.txt  hardware-detect-output.txt
# cppcheck-output.txt   clang-tidy-output.txt  ci-metadata.txt
```

The `benchmark-output.txt` file is the primary V&V evidence for CV-11:

- `numpy_batch_fps`, `jax_cpu_fps`, `numpy_batch_speedup`
- `hw_cpu_name`, `hw_cpu_cores`, `hw_ram_gb`
- `hw_recommended_backend`

---

## 5. Triggering a full release (tag + Apptainer)

```bash
# 1. Bump version, update CHANGELOG, commit:
echo "0.11.0" > VERSION
git add VERSION CHANGELOG.md README.md ...
git commit -m "feat: v0.11.0 ..."

# 2. Tag:
git tag v0.11.0

# 3. Push tag → triggers slurm-deploy.yml + auto-release.yml:
git push origin main
git push origin v0.11.0
```

The GitHub Release will automatically contain:

- CHANGELOG section for v0.11.0
- `wormsim2.sif` binary attachment
- SHA256 checksum / GPG signature file

---

## 6. Matrix strategy for multi-backend CI

The `ci-baseline.yml` uses a job matrix to test multiple Python backends:

```yaml
strategy:
  matrix:
    backend: [numpy_serial, numpy_batch, numpy_mp, jax_cpu]
```

Each matrix job verifies that the backend produces bit-identical results to
`numpy_serial` and meets a minimum-fps floor appropriate for that backend.

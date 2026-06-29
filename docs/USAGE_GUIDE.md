# wormsim2 — Installation & Usage Guide

Complete guide for installing wormsim2, running the tuner pipeline, and using the interactive browser GUI.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Command-line usage](#command-line-usage)
4. [Interactive GUI (browser)](#interactive-gui-browser)
5. [Adding a strain](#adding-a-strain)
6. [Running the GUI locally with a Python server](#running-the-gui-locally-with-a-python-server)
7. [GitHub Pages deployment](#github-pages-deployment)
8. [GitHub Actions tuner (server-side, persistent)](#github-actions-tuner-server-side-persistent)

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | ≥ 3.10 |
| numpy, scipy, pandas | any recent |
| matplotlib, plotly, kaleido | any recent |
| C++ compiler (optional) | GCC ≥ 11 or Clang ≥ 14 |
| CMake + Ninja (optional) | ≥ 3.22 |

Install Python dependencies:

```bash
pip install numpy scipy pandas matplotlib plotly kaleido pillow joblib "jax[cpu]"
```

---

## Installation

Choose the guide that matches your environment:

| Environment | Guide | Notes |
|---|---|---|
| **Local CPU** (macOS / Linux) | [`docs/install/local_cpu.md`](docs/install/local_cpu.md) | NumPy + JAX[cpu] + OpenMP C++; no GPU required |
| **Local GPU — CUDA / OpenCL** (Docker) | [`docs/install/local_gpu_docker.md`](docs/install/local_gpu_docker.md) | NVIDIA CUDA + PyOpenCL on Linux; Docker + `nvidia-container-toolkit` |
| **HPC / SLURM** (CINECA G100) | [`docs/install/hpc_slurm.md`](docs/install/hpc_slurm.md) | Singularity + SLURM; targets Tesla V100S 32 GB |
| **CI / GitHub Actions** | [`docs/install/github_actions.md`](docs/install/github_actions.md) | Ubuntu runner; `jax[cpu]` + `pyopencl` (Intel ICD fallback) |

Then clone and (optionally) build the C++ backend:

```bash
git clone https://github.com/VahidGh/wormsim2.git
cd wormsim2

# (Optional) Build C++ FEM backend
cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DWORMSIM2_OPENMP=ON
cmake --build build -j$(nproc)
```

---

## Command-line usage

### List available pre-built scenarios

```bash
python3 -m src.python.pipeline --list
```

Output:
```
n2_wt                N2 Wild-Type — 600 s Agar Plate
nca_knockout         nca-1(gk9); nca-2(gk5) — NALCN Double Knockout
egl19_rof            egl-19(n2368) — CaV1.2 Partial LOF
```

### Run a pre-built scenario (generates scenario JSON for the GUI)

```bash
python3 -m src.python.pipeline --scenario n2_wt
```

This writes `docs/gui/scenarios/n2_wt.json` with metrics, tuner parameters, hardware info, and download links.

### Run all scenarios

```bash
python3 -m src.python.pipeline
```

### Run the tuner on your own WCON file

```bash
python3 -m src.python.pipeline \
    --wcon-file  path/to/my_strain.wcon \
    --strain-id  my_strain_gk1 \
    --strain-name "my-gene(gk1)" \
    --strain-type mutant
```

This will:
1. Parse the WCON file and convert to the skeleton CSV format
2. Run the 4-mode PCA NeuromuscularTuner
3. Export the tuner reconstruction as a WCON file
4. Write `docs/gui/strains/my_strain_gk1/meta.json` + `tuner.wcon`
5. Update `docs/gui/strains/index.json` so the GUI discovers the new strain

**Note:** The Plotly animation HTML is not generated automatically. Generate it with:

```bash
python3 -c "
import sys; sys.path.insert(0, 'src/python')
from tuner import TunerConfig, NeuromuscularTuner
cfg = TunerConfig(skeleton_csv='/tmp/my_strain.csv', bl_mm=0.70)
t = NeuromuscularTuner(cfg)
t.tune()
t.export_html('docs/gui/images/my_strain_gk1_tuner.html', t_max=600)
"
```

Then update `docs/gui/strains/my_strain_gk1/meta.json` → `outputs.animation_html` to point to the HTML file.

---

## Interactive GUI (browser)

The live GUI is deployed at:

> **[vahidgh.github.io/wormsim2](https://vahidgh.github.io/wormsim2/)**

It shows:
- **✓ Pre-tuned strains** — N2, nca-1(gk9), egl-19(n2368) — animation loads instantly
- **○ Catalog strains** — 30 OWMD strains from `zenodo_catalog.json` — run the tuner via Pyodide (client-side Python/WASM)
- **○ OWMD live** — additional results fetched live from the Zenodo API

### Search bar

Type a gene name, allele, phenotype keyword, or tag (e.g. `egl`, `calcium`, `ion_channel`, `dumpy`) to filter. The search bar distinguishes:

| Badge | Meaning |
|-------|---------|
| `✓ tuned vX.Y.Z` | Pre-tuned locally — animation is instant |
| `○ catalog` | In the built-in catalog — can be tuned via Pyodide in-browser |
| `○ WT` / `○ KO/LOF` | From live OWMD Zenodo API search |

When you select a catalog or OWMD strain, the GUI fetches the WCON from Zenodo and runs the 4-mode PCA tuner via **Pyodide** (Python/WASM, client-side). This takes ~30–90 seconds depending on WCON size. After tuning, the animation renders in the same player as the pre-tuned strains, and **💾 Sim WCON** + **📄 Metrics JSON** download buttons appear. If the Zenodo CDN is CORS-blocked, use the GitHub Actions tuner (see [Adding a strain](#adding-a-strain)) to process it server-side.

---

## Adding a strain

### Scenario A — CLI + local GUI

1. Place your WCON file anywhere, e.g. `data/raw/wcon_raw/my_strain.wcon`
2. Run the pipeline:
   ```bash
   python3 -m src.python.pipeline \
       --wcon-file data/raw/wcon_raw/my_strain.wcon \
       --strain-id my_strain \
       --strain-name "my-gene(allele)"
   ```
3. The GUI will auto-discover it when you open `docs/gui/index.html` via a local server

### Scenario B — GitHub Actions (server-side, result persists for all users)

1. Go to **Actions → "Tuner: add strain from Zenodo"** in the repository
2. Click **"Run workflow"** and fill in `zenodo_id`, `strain_id`, `strain_name`, `strain_type`
3. The workflow downloads the WCON from Zenodo (no CORS), runs the pipeline, and commits the result to `main`
4. The new strain appears in the GUI's **✓ Pre-tuned** section after the `gh-pages.yml` redeploy

Via CLI:

```bash
gh workflow run tuner.yml \
  -f zenodo_id=1011979 \
  -f strain_id=egl2_n693 \
  -f strain_name="egl-2(n693)" \
  -f strain_type=mutant
```

> **Tip:** Selecting a catalog or OWMD strain in the live GUI will attempt to fetch the WCON from Zenodo and run the tuner in-browser via Pyodide. If the Zenodo CDN is CORS-blocked, use Scenario B instead to process it server-side.

---

## Running the GUI locally with a Python server

The GUI uses `fetch()` calls to load JSON files (`strains/index.json`, `strains/<id>/meta.json`, `zenodo_catalog.json`). These calls are blocked by the browser's CORS policy when opening `index.html` as a `file://` URL.

**You must serve the GUI via a local HTTP server:**

```bash
# From the repo root — serves the full GUI on http://localhost:8080
python3 -m http.server 8080 --directory docs/gui/

# Then open in your browser:
open http://localhost:8080          # macOS
xdg-open http://localhost:8080      # Linux
```

The Pyodide tuner (in-browser Python/WASM) also requires HTTP (not `file://`) to load its runtime.

> **Tip:** If you are on a network where `localhost` is not accessible, use `127.0.0.1:8080` explicitly.

---

## GitHub Pages deployment

The live site at [vahidgh.github.io/wormsim2](https://vahidgh.github.io/wormsim2/) is deployed automatically via the `gh-pages.yml` GitHub Actions workflow whenever `main` is updated.

What gets deployed:
- All files from `docs/gui/` (HTML, CSS, JS, JSON catalogs, scenario JSONs)
- `docs/gui/strains/` — per-strain meta.json + tuner WCON files
- `docs/gui/downloads/` — VTU ZIP and WCON files
- `docs/images/` — Plotly animation HTML files

**Can the Pyodide tuner work on GH Pages?** Yes. Pyodide is a pure client-side WASM runtime — it runs entirely in the browser. The GH Pages deployment enables full in-browser tuning for catalog and OWMD strains. The only limitation is that results are not persisted server-side; they live in the browser session only.

---

## GitHub Actions tuner (server-side, persistent)

For persistent results (so tuned strains appear in the `✓ Pre-tuned` section for all users), use the **`tuner.yml`** GitHub Actions workflow:

```
.github/workflows/tuner.yml
```

### Trigger manually via GitHub UI

1. Go to **Actions** → **"Tuner: add strain from Zenodo"**
2. Click **"Run workflow"**
3. Fill in:
   - `zenodo_id` — Zenodo record ID (e.g. `1031837`)
   - `strain_id` — unique slug (e.g. `n2_wt`)
   - `strain_name` — display name (e.g. `N2 (wild type)`)
   - `strain_type` — `wild_type` or `mutant`

### What the workflow does

1. Downloads the WCON from Zenodo
2. Runs `python3 -m src.python.pipeline --wcon-file ... --strain-id ...`
3. Commits `docs/gui/strains/<strain_id>/` (meta.json + tuner.wcon) to `main`
4. Triggers the `gh-pages.yml` workflow to redeploy

After the workflow completes, the new strain appears in the GUI's **✓ Pre-tuned** section for all users.

### Trigger via GitHub CLI

```bash
gh workflow run tuner.yml \
  -f zenodo_id=1031837 \
  -f strain_id=n2_wt \
  -f strain_name="N2 (wild type)" \
  -f strain_type=wild_type
```

---

## Quick reference

| Task | Command |
|------|---------|
| Install deps | `pip install numpy scipy pandas matplotlib plotly kaleido pillow joblib "jax[cpu]"` |
| Run N2 scenario | `python3 -m src.python.pipeline --scenario n2_wt` |
| Run all scenarios | `python3 -m src.python.pipeline` |
| Run tuner on WCON | `python3 -m src.python.pipeline --wcon-file my.wcon --strain-id my_id` |
| Serve GUI locally | `python3 -m http.server 8080 --directory docs/gui/` |
| Live GUI | [vahidgh.github.io/wormsim2](https://vahidgh.github.io/wormsim2/) |
| Add strain via GH Actions | `gh workflow run tuner.yml -f zenodo_id=... -f strain_id=...` |

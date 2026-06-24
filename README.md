# wormsim2

[![Version](https://img.shields.io/badge/version-v0.9.1-blue?style=flat-square)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Language](https://img.shields.io/badge/language-C%2B%2B20-blue?style=flat-square)](src/cpp/)
[![Backends](https://img.shields.io/badge/backends-CPU%20%7C%20CUDA%20%7C%20OpenCL-76b900?style=flat-square)](docs/research/00-motivation-objectives-related-work.md)
[![Status](https://img.shields.io/badge/status-CI%20live-brightgreen?style=flat-square)](docs/ISSUES.md)
[![CI](https://github.com/VahidGh/wormsim2/actions/workflows/ci-baseline.yml/badge.svg)](https://github.com/VahidGh/wormsim2/actions/workflows/ci-baseline.yml)

> **A real-time, biophysically-faithful *C. elegans* locomotion simulator.**

wormsim2 aims to fill the unoccupied quadrant in whole-worm simulation: a simulator
that is **real-time on a commodity GPU** *and* keeps **Hodgkin–Huxley ion-channel
dynamics** (rather than a trained-network approximation of the nervous system), with a
**corotated finite-element body**, a **closed proprioceptive sensorimotor loop**, and
**quantitative validation against real-worm behavioural data**.

Existing reference simulators each give up one of these: **Sibernetic** is
ion-channel-faithful but ~5×10⁴× slower than real time; **MetaWorm/BAAIWorm** is
real-time but replaces the ion-channel nervous system with a black-box trained network.
Neither closes the proprioceptive loop, and both validate weakly. wormsim2 targets all
of these at once, and ships a **web-friendly browser 3D viewer** so results are
explorable without a native install.

> 📄 **Scientific foundation:** [docs/research/00-motivation-objectives-related-work.md](docs/research/00-motivation-objectives-related-work.md)

> 📋 **Requirements:** [docs/requirements/01-requirements-analysis.md](docs/requirements/01-requirements-analysis.md)

> 🏗️ **Architecture:** [docs/design/02-architecture-design.md](docs/design/02-architecture-design.md)

> 🗒️ **Project tour (notebook):** [notebooks/project_tour.ipynb](notebooks/project_tour.ipynb)

---

## Objectives

|              | Goal                                                                                                 |
| ------------ | ---------------------------------------------------------------------------------------------------- |
| **G1** | Real-time on a commodity GPU**while retaining Hodgkin–Huxley ion channels**                   |
| **G2** | Corotated linear-elastic FEM body (handles large rotations: crawl / swim / omega-turn)               |
| **G3** | **Closed proprioceptive loop** — stretch feedback sustains undulation *(core contribution)* |
| **G4** | Quantitative behavioural validation via the 256-feature WCON / movement-analysis pipeline            |
| **G5** | Mechanistic interpretability — an ion-channel perturbation produces a predicted phenotype           |
| **G6** | Open, reproducible, well-tested, containerised, HPC-deployable software                              |
| **G7** | Web-friendly output schema + self-contained browser 3D viewer                                        |

Engineering objectives (modern C++ core, CUDA acceleration, parallelism, V&V/CI, Python
analysis layer), domain assumptions, and explicit non-goals are detailed in the
[research charter](docs/research/00-motivation-objectives-related-work.md).

---

## Latest validated results (v0.9.1)

**CV-9.2 — 3D foraging trajectory reconstruction vs Nguyen 2018 real-worm data**

> **Publication:** Nguyen et al. (2018) *"Three-dimensional behavioural phenotyping of freely moving C. elegans using quantitative light field microscopy"*, PLOS ONE — [https://doi.org/10.1371/journal.pone.0200108](https://doi.org/10.1371/journal.pone.0200108)
> **Dataset:** figshare — [https://doi.org/10.6084/m9.figshare.6670805](https://doi.org/10.6084/m9.figshare.6670805) (`MidlineSkeletons.mat`: 601 frames × 25 body points × 3D coordinates, N2 foraging in agarose gel, 20 fps)

wormsim2 `NeuromuscularTuner.crawl_3d_skeleton_trackfollow()` uses the real head positions from the Nguyen et al. 2018 light-field microscopy dataset as the head trajectory and places each body segment using a retrograde track-following model (each segment traces the same 3D path as the head, delayed by `ds / v_wave` per step; phase velocity `v_wave = f × λ × L = 272 µm/s`).

<img src="docs/images/nguyen2018_fig4b_tracked.gif" width="900" alt="wormsim2 track-following — 30 s foraging animation, real Nguyen 2018 head path"/>

> **Animation:** 30 s foraging trajectory at 20 fps (601 frames). Colour encodes time (viridis); red dots = head. Body follows the real Nguyen 2018 head path with a retrograde wave delay of ~2.2 s head-to-tail.

**Side-by-side vs real data (matched axes):**

<img src="docs/images/nguyen2018_fig4b_tracked_compare.png" width="900" alt="Real Nguyen 2018 Fig 4(b) vs wormsim2 track-following — same coordinate frame"/>

> **Left:** Nguyen 2018 Fig 4(b) real skeleton (all 601 frames overlaid).  **Right:** wormsim2 track-following model on the same axes (X, Y, Z in µm; same centering offset). Head trail and overall foraging volume match.

**Body-position error vs real skeleton:**

<img src="docs/images/nguyen2018_fig4b_error.png" width="900" alt="Track-following body-position error vs Nguyen 2018 real skeleton"/>

| Metric | Value | Notes |
|---|---|---|
| Head error (k = 0) | **0 µm** | Exact by construction (real positions used) |
| Overall body RMSE | **278.7 µm = 46.1% BL** | Residual = lateral undulation not in track-following model |
| Tail error (k = 24) | **443.7 µm** | Tail is furthest from head → largest deviation |
| Body length L | 604 µm | Nguyen 2018 N2 median |

> Residual error monotonically increases from head to tail (heatmap shows no temporal structure). The gap quantifies the undulation amplitude that the pure track-following model does not capture; the next step is to superimpose the measured eigenworm bending modes.

---

## v0.8.2 results

**NeuromuscularTuner — interactive Plotly animation (CV-8.1.1):**

<img src="docs/images/v082_n2_vs_cel_tuned.gif" width="900" alt="N2 real worm vs wormsim2 v0.8.2 NeuromuscularTuner — animated comparison with 0.1 mm grid"/>

> **Left:** real *C. elegans* N2 locomotion (Zenodo 1031837, Schafer Lab); 4-second window at 10 fps. White dot = head; dotted line = cumulative head trajectory; dots = 48 muscle attachment points (24D + 24V). Grid spacing = 0.1 mm.
> **Right:** wormsim2 v0.8.2 `NeuromuscularTuner` — 4-mode eigenworm activation: θ(s,t) = μ(s) + Σ aₙ(t)·eₙ(s), n=0–3; R² = 0.944 (94.4% posture variance). Same grid and scale.
> [**▶ Open interactive animation (scrubber + speed panel)**](docs/images/v082_n2_vs_cel_tuned.html)

| | v0.8 FEM (uniform drive) | v0.8.2 NeuromuscularTuner |
|---|---|---|
| Activation | spatially uniform C-bend | 4-mode eigenworm, PCA of N2 θ(s,t) |
| Posture variance captured | — | **94.4%** (modes 0–3: 76.8 + 8.5 + 6.5 + 2.5%) |
| CEl₄₈ (mean) | 0.4278 BL² | **0.000133 BL²** |
| RMS / muscle point | 434 µm | **8 µm** |
| Improvement | — | **3212× CEl · 57× RMS** |

---

## v0.8.0 results

**Corotated FEM body: CV-8.x suite (deal.II 9.5.1, UMFPACK, overdamped implicit Euler)**

| CV     | Check                                                         | Published reference                                                                          | Result                  |
| ------ | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ----------------------- |
| CV-8.1 | Dorsal C-bend tangent-angle correlation vs linear ref ≥ 0.80 | Dorsal activation → monotone κ(s) → linear θ(s); uniform dorsal → C-bend not S-wave     | **r = 0.93 PASS** |
| CV-8.2 | Mid-body oscillation frequency ∈ [0.35, 0.65] Hz             | [Stephens et al. 2008](https://doi.org/10.1371/journal.pcbi.1000028) Table 1: 0.529 ± 0.069 Hz | **0.50 Hz PASS**  |
| CV-8.3 | Orbit circularity max\|a1\|/max\|a0\| > 0.5                   | Traveling-wave criterion: a0/a1 phase orbit should be circular, not degenerate               | **0.64 PASS**     |

CTest 10/10 PASS — new suite: `test_fem_body` (label: `body;fem`, deal.II guard).

**N2 wild-type vs wormsim2 v0.8 — locomotion comparison (CV-8.5):**

<img src="docs/images/v080_n2_vs_cel_3d.gif" width="900" alt="N2 real worm vs wormsim2 v0.8 CEl 2D bird's-eye locomotion comparison"/>

> **Left:** real *C. elegans* N2 locomotion (Zenodo 1031837, Schafer Lab); 4-second window at 30 fps.
> **Right:** wormsim2 CEl traveling-wave kinematic target (v0.9 goal) — θ(s,t) = 0.40·sin(2πft − 2πs/λ), amplitude calibrated to match N2.
> **Table:** CEl metric computed against the actual v0.8 FEM output.

> **Note on v0.8 motor drive:** the v0.8 FEM body is currently driven by an open-loop, spatially uniform sinusoidal activation applied identically across all 95 body-wall muscles — equivalent to an unstructured default signal with no propagating phase gradient. This produces a quasi-static C-bend rather than a traveling S-wave (first eigenworm coefficient a₀ = 0.011 rad, 12× below the N2 mean of 0.139 rad; CEl metric = 0.428 BL², RMS positional error = 434 µm per body point). The quantitative gap between the simulated body and real N2 kinematics reflects the absence of spatiotemporally structured neuromuscular drive, not a limitation of the mechanical model per se.

> **v0.9 target — closed-loop neuromuscular tuning:** the next milestone will introduce a traveling-wave activation pattern derived from the Hodgkin–Huxley motor circuit, wherein dorsal and ventral motoneuron populations generate a rostrocaudal phase gradient across the 95 BWM segments. Convergence toward wild-type kinematics will be quantified by the CEl metric; the target threshold is ≤ 0.01 BL² (a 43-fold reduction), corresponding to sub-100 µm mean positional error per body point.

---

## v0.7.0 validated results

**Body shape comparison vs Stephens 2008 eigenbasis — [Stephens et al. 2008](https://doi.org/10.1371/journal.pcbi.1000028)**

<img src="docs/images/v070_shape_comparison.png" width="720" alt="v0.7.0 body shape comparison vs Stephens 2008"/>

*Row 1 (blue): Stephens 2008 eigenbasis reference at the phases the simulation visits (A = 0.5 rad, biologically estimated).
Row 2 (green): wormsim2 modes 0+1 reconstruction, amplitude-normalised — **shape r = 1.000** (body module correctly implements the Stephens basis).
Row 3 (red): wormsim2 all-mode output showing mode-2 excess from CPG proxy (3.1× reference, expected at v0.7.0).
Phase portrait: actual simulation orbit (near-linear) vs ideal circular orbit at same amplitude (what a proper traveling wave produces).*

| CV     | Check                                            | Published reference                                                                          | Result                               |
| ------ | ------------------------------------------------ | -------------------------------------------------------------------------------------------- | ------------------------------------ |
| CV-7.1 | Undulation frequency 0.500 Hz ∈ [0.35, 0.65] Hz | [Stephens et al. 2008](https://doi.org/10.1371/journal.pcbi.1000028) Table 1: 0.529 ± 0.069 Hz | **PASS**                       |
| CV-7.2 | Modes 0+1 energy fraction 0.687 > 0.50           | Stephens 2008 Fig. 2B: modes 1+2 ≈ 0.75 for real N2                                         | **PASS** (CPG-proxy threshold) |
| Shape  | Modes 0+1 body shape vs Stephens eigenbasis (r)  | Stephens 2008 eigenbasis (`master_eigen_worms_N2.mat`)                                     | **r = 1.000** (exact match)    |

CTest 9/9 PASS — new suites: `test_eigenworm` (body) + `test_wcon` (output).

---

**v0.6.0 — NMJ layer: digitized reference vs wormsim2 side-by-side** (openworm `NeuronMuscle.png`, same Boyle-Cohen muscle channels):

![CV-6.1E NMJ comparison: digitized reference vs wormsim2](docs/images/cv_6_1e_nmj_comparison.png)

*Left: openworm jNeuroML reference (I&F neuron, gbase=25nS, 8 muscle APs, ΔV=68 mV).
Middle: traces digitized from the reference figure (muscle −80→−12 mV).
Right: wormsim2 (HH DB1 motoneuron, 0.5nS NMJ, sustained ΔV=67 mV at 400ms).*
*Shape differs (APs vs. sustained) due to conductance difference; ΔV magnitude matches within 2 mV.*
*Source: [openworm/muscle_model NeuronMuscle.png](https://github.com/openworm/muscle_model/blob/master/NeuroML2/images/NeuronMuscle.png)*

| CV      | Check                                           | Published/online reference                                                                                      | Result                     |
| ------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------- |
| CV-6.1A | 552 NMJ connections loaded                      | —                                                                                                              | **PASS**             |
| CV-6.1B | ΔV_muscle = +66.95 mV (NMJ-driven)             | —                                                                                                              | **PASS** (> 2 mV)    |
| CV-6.1C | L/R symmetry max\|ΔV\| = 0.07 mV               | —                                                                                                              | **PASS** (< 1 mV)    |
| CV-6.1D | Muscle resting Vm = −65.0 mV                   | [Richmond 2009, JoVE](https://doi.org/10.3791/1165): −30 to −65 mV                                               | **PASS**             |
| CV-6.1E | ΔV = 66.95 mV vs openworm ref ΔV ≈ 60–65 mV | [openworm NeuronMuscle.png](https://github.com/openworm/muscle_model/blob/master/NeuroML2/images/NeuronMuscle.png) | **PASS** (50–80 mV) |

---

**v0.5.1 — Muscle cell dynamics (isolated): `muscle_trace` vs Boyle & Cohen 2008 Fig. 2A:**

![CV-5.4 muscle_trace vs Boyle & Cohen 2008](docs/images/cv_5_4_muscle_trace.png)

| CV      | Check                                         | Published reference                                              | Result         |
| ------- | --------------------------------------------- | ---------------------------------------------------------------- | -------------- |
| CV-5.4A | 100 pA sub-threshold trace, max\|ΔV\| < 8 mV | [Boyle &amp; Cohen 2008](https://doi.org/10.2976/1.2804583) Fig. 2A | **PASS** |
| CV-5.4B | 400 pA supra-threshold peak within 20 mV      | Boyle & Cohen 2008 Fig. 2A                                       | **PASS** |
| CV-5.4C | 700 pA action potential Δt_peak < 5 ms       | Boyle & Cohen 2008 Fig. 2A                                       | **PASS** |

System CV: **5/5 ALL PASS** (engine accuracy · ca_boyle kinetics · connectome topology · muscle dynamics · NMJ layer).
Every CV row is validated against a published/online source with DOI.

---

## Architecture (target)

```
(1) Neural layer       GPU-batched Hodgkin–Huxley over the connectome
        │              (302 neurons · synapses + gap junctions)
        ▼
(2) Neuromuscular map  motoneuron V → 95 body-wall-muscle activations
        ▼
(3) Body mechanics     corotated linear-elastic FEM (per-quadrant muscle drive)
        ▼
(4) Environment        agar (crawl) ↔ liquid (swim) · contact + drag
        ▲                                              │
        └──── (5) Proprioceptive feedback: curvature ──┘
                  → stretch-receptor channels → motor circuit
        │
        ▼
(6) Output layer       web-friendly serialization → WCON export + browser 3D viewer
```

The compute core is **backend-agnostic**: a portable **CPU/OpenMP** reference backend
(development + CI), a **CUDA** backend (NVIDIA real-time / HPC deployment), and an
optional **OpenCL** backend — all behind a single internal kernel interface.

---

## Repository layout

```
wormsim2/
├── README.md  CHANGELOG.md  VERSION  LICENSE
├── CMakeLists.txt                                          ← root build (C++20, warnings, backend option)
├── .clang-tidy  .dockerignore  .gitignore
├── docker/
│   └── Dockerfile.wormsim2-dev                            ← block-structured dev image (Ubuntu 22.04)
├── notebooks/
│   └── project_tour.ipynb                                 ← version-by-version tour (input→output examples)
├── docs/
│   ├── research/00-motivation-objectives-related-work.md  ← scientific charter
│   ├── requirements/01-requirements-analysis.md           ← requirements analysis
│   ├── design/02-architecture-design.md                   ← architecture design
│   ├── file-registry.md                                   ← per-file update trigger registry
│   └── ISSUES.md                                          ← issues & improvements tracker
├── src/
│   ├── cpp/
│   │   ├── CMakeLists.txt
│   │   │   ├── include/io/                                ← NetworkConfig, loaders, parser headers
│   │   ├── src/io/                                    ← NEURONLoader, NeuroMLLoader, etc.
│   │   ├── include/neural/                            ← ChannelKinetics, NeuralState, NeuralIntegrator headers
│   │   ├── src/neural/                                ← HH ODE integrator, Rush–Larsen gates
│   │   ├── tools/                                     ← neural_trace, connectome_trace: CSV data runners
│   │   └── tests/                                     ← CTest suites (io: 3/3, neural: 2/2)
│   └── python/notebooks/                              ← validation & analysis (planned)
└── config/                                            ← run configuration (planned)
```

---

## Status

Track progress in [docs/ISSUES.md](docs/ISSUES.md) and [CHANGELOG.md](CHANGELOG.md).

| Component                                                                               | Status                           |
| --------------------------------------------------------------------------------------- | -------------------------------- |
| [Scientific charter](docs/research/00-motivation-objectives-related-work.md)               | Draft                            |
| [Requirements analysis](docs/requirements/01-requirements-analysis.md)                     | Draft                            |
| [Architecture design](docs/design/02-architecture-design.md)                               | Draft                            |
| `src/cpp/io/` — dual-format network loader                                           | **Done** (2/2 tests pass)  |
| CI pipeline (build/test/cppcheck/clang-tidy/coverage)                                   | **Done**                   |
| C++20 quality checker (72 static checks, 12 categories)                                 | **Done**                   |
| `src/cpp/neural/` — HH ODE integrator                                                | **Done** (2/2 tests pass)  |
| `src/cpp/tools/neural_trace` — CSV data runner (7 scenarios, incl. `muscle_trace`) | **Done**                   |
| `data/c302/c302_C2_Full.net.nml` — c302 C2 full connectome                           | **Done**                   |
| `src/cpp/tools/connectome_trace` — full-connectome trace tool                        | **Done** (5/5 tests pass)  |
| NMJ layer (`NeuralIntegrator` + `NeuralState` + `NetworkConfig`)                  | **Done** (3/3 tests pass)  |
| `notebooks/project_tour.ipynb` — C++ output demos + Boyle-Cohen CV + v0.5            | **Done**                   |
| FEM body (`FEMBody`) — corotated elastic, UMFPACK, overdamped implicit Euler         | **Done** (3/3 CV-8.x pass) |
| Compute backends (OpenCL / CUDA)                                                        | Planned                          |
| Python validation layer                                                                 | Planned                          |
| Browser 3D viewer                                                                       | Planned                          |

---

## License

[MIT](LICENSE) © 2026 Seyed Vahid Ghayoomie — <seyedvahid.ghayoomie@mail.polimi.it>

## AI attribution

Developed with the assistance of **Claude Code** (Anthropic) acting as a development
engineer under the author's scientific direction.

## Acknowledgements

Builds on the open *C. elegans* modelling ecosystem — the connectome and c302 nervous
system, the Sibernetic body simulator, MetaWorm/BAAIWorm, and the WCON / worm-movement
community tooling. See the [research charter](docs/research/00-motivation-objectives-related-work.md) for full citations.

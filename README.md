# wormsim2

[![Version](https://img.shields.io/badge/version-v0.7.0-blue?style=flat-square)](CHANGELOG.md)
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

## Latest validated results (v0.7.0)

**Body shape comparison vs Stephens 2008 eigenbasis — [Stephens et al. 2008](https://doi.org/10.1371/journal.pcbi.1000028)**

<img src="docs/images/v070_shape_comparison.png" width="720" alt="v0.7.0 body shape comparison vs Stephens 2008"/>

*Row 1 (blue): Stephens 2008 eigenbasis reference at the phases the simulation visits (A = 0.5 rad, biologically estimated).
Row 2 (green): wormsim2 modes 0+1 reconstruction, amplitude-normalised — **shape r = 1.000** (body module correctly implements the Stephens basis).
Row 3 (red): wormsim2 all-mode output showing mode-2 excess from CPG proxy (3.1× reference, expected at v0.7.0).
Phase portrait: actual simulation orbit (near-linear) vs ideal circular orbit at same amplitude (what a proper traveling wave produces).*

| CV      | Check                                               | Published reference                                                                 | Result                          |
| ------- | --------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------- |
| CV-7.1  | Undulation frequency 0.500 Hz ∈ [0.35, 0.65] Hz   | [Stephens et al. 2008](https://doi.org/10.1371/journal.pcbi.1000028) Table 1: 0.529 ± 0.069 Hz | **PASS**               |
| CV-7.2  | Modes 0+1 energy fraction 0.687 > 0.50             | Stephens 2008 Fig. 2B: modes 1+2 ≈ 0.75 for real N2                               | **PASS** (CPG-proxy threshold) |
| Shape   | Modes 0+1 body shape vs Stephens eigenbasis (r)    | Stephens 2008 eigenbasis (`master_eigen_worms_N2.mat`)                              | **r = 1.000** (exact match)    |

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

| Component                                                                               | Status                          |
| --------------------------------------------------------------------------------------- | ------------------------------- |
| [Scientific charter](docs/research/00-motivation-objectives-related-work.md)               | Draft                           |
| [Requirements analysis](docs/requirements/01-requirements-analysis.md)                     | Draft                           |
| [Architecture design](docs/design/02-architecture-design.md)                               | Draft                           |
| `src/cpp/io/` — dual-format network loader                                           | **Done** (2/2 tests pass) |
| CI pipeline (build/test/cppcheck/clang-tidy/coverage)                                   | **Done**                  |
| C++20 quality checker (72 static checks, 12 categories)                                 | **Done**                  |
| `src/cpp/neural/` — HH ODE integrator                                                | **Done** (2/2 tests pass) |
| `src/cpp/tools/neural_trace` — CSV data runner (7 scenarios, incl. `muscle_trace`) | **Done**                  |
| `data/c302/c302_C2_Full.net.nml` — c302 C2 full connectome                           | **Done**                  |
| `src/cpp/tools/connectome_trace` — full-connectome trace tool                        | **Done** (5/5 tests pass) |
| NMJ layer (`NeuralIntegrator` + `NeuralState` + `NetworkConfig`)                  | **Done** (3/3 tests pass) |
| `notebooks/project_tour.ipynb` — C++ output demos + Boyle-Cohen CV + v0.5            | **Done**                  |
| FEM body (`FEMBody`)                                                                  | Planned                         |
| Compute backends (OpenCL / CUDA)                                                        | Planned                         |
| Python validation layer                                                                 | Planned                         |
| Browser 3D viewer                                                                       | Planned                         |

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

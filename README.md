# wormsim2

[![Version](https://img.shields.io/badge/version-v0.4.1-blue?style=flat-square)](CHANGELOG.md)
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
│   │   ├── tools/                                     ← neural_trace: CSV data runner for notebook demos
│   │   └── tests/                                     ← CTest suites (io: 2/2, neural: 2/2)
│   └── python/notebooks/                              ← validation & analysis (planned)
└── config/                                            ← run configuration (planned)
```

---

## Status

Track progress in [docs/ISSUES.md](docs/ISSUES.md) and [CHANGELOG.md](CHANGELOG.md).

| Component                                                                 | Status                          |
| ------------------------------------------------------------------------- | ------------------------------- |
| [Scientific charter](docs/research/00-motivation-objectives-related-work.md) | Draft                           |
| [Requirements analysis](docs/requirements/01-requirements-analysis.md)       | Draft                           |
| [Architecture design](docs/design/02-architecture-design.md)                 | Draft                           |
| `src/cpp/io/` — dual-format network loader                             | **Done** (2/2 tests pass) |
| CI pipeline (build/test/cppcheck/clang-tidy/coverage)                     | **Done**                  |
| C++20 quality checker (72 static checks, 12 categories)                   | **Done**                  |
| `src/cpp/neural/` — HH ODE integrator                                  | **Done** (2/2 tests pass) |
| `src/cpp/tools/neural_trace` — CSV data runner (6 scenarios)           | **Done**                  |
| `notebooks/project_tour.ipynb` — C++ output demos + Boyle-Cohen CV     | **Done**                  |
| FEM body (`FEMBody`)                                                    | Planned                         |
| Compute backends (OpenCL / CUDA)                                          | Planned                         |
| Python validation layer                                                   | Planned                         |
| Browser 3D viewer                                                         | Planned                         |

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

# wormsim2 — File Registry

> **Purpose**: before every commit, scan this table. Any file marked as affected by the
> change being committed must be updated in the same commit. "Affected" means the change
> touches the trigger column for that file.
>
> Add a new row whenever a new file is created. Delete or mark `[removed]` when a file
> is deleted.

---

## Root

| File | Description | Update triggers |
|---|---|---|
| `README.md` | Public-facing overview, badges, architecture diagram, repo layout, status table | New module added/removed; version bump; badge values change (C++ std, status, backends); layout changes; status table rows change |
| `CHANGELOG.md` | Keep-a-Changelog history of every notable change | **Every commit that changes user-visible behaviour or project structure** — new files, API changes, build changes, doc additions |
| `VERSION` | Single-line version string (`X.Y.Z[-suffix]`) | Every formal version tag |
| `LICENSE` | MIT licence text | Change of author or licence |
| `CMakeLists.txt` | Root CMake: C++ standard, warnings, subdirectories, backend option | C++ standard bump; new top-level subdirectory; new compile options; new CMake option |
| `.clang-tidy` | clang-tidy check configuration | Adding/suppressing a check category; adopting a new check |
| `.dockerignore` | Docker build context exclusions | New top-level directory or large asset that should be excluded |
| `.gitignore` | Git ignore rules | New generated artifact type; new tool producing output under the repo root |

---

## `docs/images/`

| File | Description | Update triggers |
|---|---|---|
| `docs/images/v051_muscle_traces.png` | C++ muscle_trace demo output at 100/400/700 pA — extracted from `notebooks/project_tour.ipynb` v0.5.1 section | New `muscle_trace` run with changed parameters; notebook re-executed |
| `docs/images/cv_5_4_muscle_trace.png` | CV-5.4 comparison plot: C++ traces vs Boyle & Cohen 2008 Fig. 2A digitized data — extracted from `notebooks/project_tour.ipynb` CV-5.4 cell | CV-5.4 cell re-run; new reference data; pass threshold change |
| `docs/images/v060_nmj_traces.png` | v0.6.0 full motor circuit demo: AVBL→DB1→MDL07/MDR07 500 ms traces — extracted from `notebooks/project_tour.ipynb` v0.6.0 section | v0.6.0 demo cell re-run; new NMJ parameters |
| `docs/images/cv_6_1_nmj_coupling.png` | CV-6.1 NMJ coupling plot: DB1 motoneuron vs MDL07/MDR07 muscle voltages (first 100 ms) — extracted from CV-6.1 cell | CV-6.1 cell re-run; NMJ parameter change |
| `docs/images/ref_neuronmuscle_openworm.png` | CV-6.1E reference: openworm/muscle_model NeuronMuscle.png — jNeuroML simulation of I&F motoneuron → Boyle-Cohen muscle via expTwoSynapse (gbase=25nS). Used as published reference for NMJ ΔV validation. | Source figure updated upstream; comparison threshold changed |
| `docs/images/cv_6_1e_nmj_comparison.png` | CV-6.1E 3-panel comparison: (1) openworm NeuronMuscle.png screenshot, (2) digitized reference traces (8 muscle APs, ΔV=68mV), (3) wormsim2 DB1→MDL07 sustained depolarisation (ΔV=67mV at 400ms) — extracted from CV-6.1E notebook cell | CV-6.1E cell re-run; digitized reference data changed |

---

## `data/`

| File | Description | Update triggers |
|---|---|---|
| `data/c302/c302_C2_Full.net.nml` | OpenWorm c302 C2 reference connectome in NeuroML2 format — 302 GenericNeuronCell + 95 GenericMuscleCell, 1084 gap junctions, 2279 chemical synapses, 552 NMJs; canonical source for all connectome-level tests and notebook demos | New upstream c302 release; format change in NeuroML2 output; partial files for faster dev iteration added |

---

## `notebooks/`

| File | Description | Update triggers |
|---|---|---|
| `notebooks/project_tour.ipynb` | Version-by-version project tour: one section per MINOR/MAJOR release. C++ binaries run via Docker emit CSV; Python reads and plots. No algorithm reimplemented in Python. | New MINOR or MAJOR version released; new `neural_trace` scenario added; channel catalog parameter changed |

---

## `docker/`

| File | Description | Update triggers |
|---|---|---|
| `docker/Dockerfile.wormsim2-dev` | Block-structured dev image (Ubuntu 22.04; cmake/GCC; OpenCL; cppcheck/clang-tidy; commented CUDA/dealii/Python blocks) | New system dependency; enabling a commented block; base image version bump; new tool added to Block 3+ |

---

## `docs/`

| File | Description | Update triggers |
|---|---|---|
| `docs/file-registry.md` | **This file** — inventory of all tracked files and their update triggers | New file added or removed anywhere in the project |
| `docs/ISSUES.md` | ISSUE-NNN tracker for bugs, improvements, decisions, research questions | New issue opened; issue resolved/closed; issue scope changed |
| ~~`docs/cpp-guidelines.md`~~ | **[removed v0.4.0]** — superseded by `src/cpp/tests/static/cpp_quality_checker.py` | — |
| `docs/research/00-motivation-objectives-related-work.md` | Scientific charter — motivation, related work, gap analysis, 4-layer architecture, validation strategy, G/E/DA/N objectives | New competitor added; architecture layer added/changed; objective added/changed; GPU estimate updated |
| `docs/requirements/01-requirements-analysis.md` | Requirements analysis — stakeholders, world-machine, DA, FR, NFR, UC, traceability | New functional/non-functional requirement; new use case; new stakeholder; traceability matrix row added |
| `docs/design/02-architecture-design.md` | Architecture design — C&C/Module/Deployment views, component interfaces, VR gates, design principles | New component; interface changed; VR gate added; design principle updated; deployment target added |

---

## `.github/workflows/`

| File | Description | Update triggers |
|---|---|---|
| `.github/workflows/ci-baseline.yml` | CI: cmake build, CTest (all labels), cppcheck, clang-tidy, coverage, artifact upload | New test added; new source file added to build; new tool in the check chain; OpenCL/CUDA block enabled |
| `.github/workflows/auto-release.yml` | Auto-creates GitHub Release on `v*` tag push; extracts CHANGELOG section | Repo URL changes; release body template changes; softprops version bump |
| `.github/workflows/backfill-releases.yml` | Manual one-shot workflow to backfill GitHub Releases for all existing `v*` tags | Release body template changes |

---

## `src/cpp/`

| File | Description | Update triggers |
|---|---|---|
| `src/cpp/CMakeLists.txt` | Library and test target definitions for the C++ core | New source file added to `wormsim2_io` or other library; new library target; new include directory |
| `src/cpp/tests/CMakeLists.txt` | Top-level test subdirectory enabler | New test subdirectory added |
| `src/cpp/tests/io/CMakeLists.txt` | CTest targets for io/ tests | New test executable in `tests/io/`; link dependency change |

### `src/cpp/include/neural/`

| File | Description | Update triggers |
|---|---|---|
| `ChannelKinetics.h` | `GateKinetics` / `ChannelSpec` structs; pure `x_inf`, `tau_x`, `rush_larsen` functions; `channel_spec()` catalog lookup | New gate kinetics form; new pure function; catalog interface change |
| `NeuralState.h` | SoA state (`v[]`, `gate[]`, `s_syn[]`) + `allocate()` factory | New state field (new ion species, synaptic model change); stride layout change |
| `NeuralIntegrator.h` | `NeuralIntegrator` class: `step()`, `voltages()`, `state()`, `reset()` | New public method; interface change; new numerical scheme; unit change |

### `src/cpp/src/neural/`

| File | Description | Update triggers |
|---|---|---|
| `ChannelKinetics.cpp` | Built-in c302 channel catalog (NCA/KD/KA/KQS/KVS/IR + KSLOW_BC/KFAST_BC/LEAK_BC from Boyle & Cohen 2008 NMODL) | New c302 channel added; parameter re-fit from experimental data; new channel form |
| `NeuralState.cpp` | `NeuralState::allocate` implementation | State layout change |
| `NeuralIntegrator.cpp` | `build_layout`, `update_gates`, `update_synapses`, `update_voltages`, `gate_product` | Numerical scheme change; new current type; gap-junction algorithm change |

### `src/cpp/tools/`

| File | Description | Update triggers |
|---|---|---|
| `neural_trace.cpp` | CLI data runner for notebook demonstrations — 7 scenarios (`nca_decay`, `kd_gate`, `gap_junc`, `multi_ch`, `boyle2008`, `chan_kinetics`, `muscle_trace`); outputs CSV to stdout; called from `notebooks/project_tour.ipynb` via Docker | New scenario needed for notebook; new channel in catalog; new integrator feature to demonstrate |
| `test_nmj.cpp` | CTest suite for NMJ layer (3 assertions: state allocation, NMJ depolarization +2 mV, no-NMJ control ±5 mV from rest) | `NMJDef` struct change; NMJ kinetic parameter change; new assertion needed |
| `connectome_trace.cpp` | CLI tool: `connectome_trace <nml> <T_ms> <dt_ms> <neuron_names>` → CSV voltage traces (stdout) + load summary (stderr); loads the full connectome via `NetworkInputParser::load()` and runs `NeuralIntegrator` | New integrator feature; new CLI argument; notebook v0.5+ cell update needed |
| `CMakeLists.txt` | Build targets `neural_trace`, `connectome_trace` (always built, not test-gated) | New tool executable added |

---

### `src/cpp/tests/neural/`

| File | Description | Update triggers |
|---|---|---|
| `test_hh_single.cpp` | CTest suite: leak at reversal, exponential decay vs analytic, KD gate direction, I_ext depolarisation | New `NeuralIntegrator` feature; regression; new channel added to catalog |
| `test_gap_junction.cpp` | CTest suite: mean-voltage conservation, V_diff exponential decay vs analytic, long-run convergence | New gap-junction implementation; coupling scheme change |
| `CMakeLists.txt` | CTest targets for neural/ tests (label: `neural`) | New test executable; link dependency change |

---

### `src/cpp/include/io/`

| File | Description | Update triggers |
|---|---|---|
| `NetworkConfig.h` | All network data structs (`GateVar`, `ChannelDef`, `ChannelAssignment`, `NeuronDef`, `ChemSynapseDef`, `GapJunctionDef`, `NMJDef`, `NetworkConfig`) + `find_neuron` | New field in any struct; new struct; new accessor; change to any `[[nodiscard]]` annotation |
| `NEURONLoader.h` | Static-method class for NEURON `.mod` / `.hoc` loading | New public method; signature change; new private helper |
| `NeuroMLLoader.h` | Static-method class for NeuroML2 `.nml` loading | New public method; signature change; new private helper |
| `NetworkInputParser.h` | Format dispatcher (`enum class NetworkFormat`, `load()`) | New format added; `load()` signature change |
| `PerturbationConfig.h` | Static `apply()` / `applyLine()` — perturbation spec parser | New supported key type; signature change |

### `src/cpp/src/io/`

| File | Description | Update triggers |
|---|---|---|
| `NetworkConfig.cpp` | `find_neuron` implementation | Logic change to neuron lookup |
| `NEURONLoader.cpp` | NMODL block parsers (`parseNeuronBlock`, `parseParameterBlock`, `parseStateBlock`, `parseModSource`, `parseMod`, `parseModDir`, `loadHoc` stub) | New NMODL block type supported; new parameter name recognised; HOC parsing implemented |
| `NeuroMLLoader.cpp` | XML scanner helpers + population/biophysics/projection parsers | New NeuroML2 element type supported; new attribute parsed; parser refactor |
| `NetworkInputParser.cpp` | `detectFormat` + `load` dispatcher | New format; new extension; perturbation spec logic change |
| `PerturbationConfig.cpp` | `applyLine` dispatch + `apply` file reader | New key namespace supported; new subkey |

### `src/cpp/tests/static/`

| File | Description | Update triggers |
|---|---|---|
| `cpp_quality_checker.py` | 72-check Python static analyser (12 categories: TC/FP/NS/FN/RA/CL/TM/SL/LB/PS/OL/DS); runs as CTest `cpp_quality`; exit 1 on first regression | New C++ module added (add checks); check removed/relaxed; new category needed |
| `CMakeLists.txt` | Wires `cpp_quality_checker.py` into CTest via `find_package(Python3)` | Checker renamed; timeout change; new static test added |

### `src/cpp/tests/io/`

| File | Description | Update triggers |
|---|---|---|
| `test_nmodl_parser.cpp` | CTest suite for NMODL parsing (`parseMod`, `parseModDir`, multi-channel, gate variables, `find_neuron`) | New `NEURONLoader` method; new struct field; regression found |
| `test_perturbation.cpp` | CTest suite for perturbation spec (`channel.*`, `gap_junction.*`, `neuron.*`, unknown key, missing file) | New `PerturbationConfig` key; new struct field; regression found |
| `test_connectome_load.cpp` | CTest integration test for c302 C2 Full connectome: 5 assertions (population counts, connectivity, biophysics, key neurons, integrator stability); locates NML file via `WORMSIM2_SOURCE_DIR` or upward walk | New connectome file added; `NetworkConfig` struct change; new `NeuralIntegrator` feature |

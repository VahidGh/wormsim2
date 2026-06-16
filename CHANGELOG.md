# Changelog

All notable changes to **wormsim2** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Versioning policy**
> - `MAJOR` (1.x.x): first public release; breaking API changes after 1.0
> - `MINOR` (0.x.0): new module, new capability, or significant doc milestone
> - `PATCH` (0.x.y): bug fixes, test additions, minor doc corrections within a milestone

---

## [0.5.1] - 2026-06-16 *(current)*

### Added

- **`muscle_trace` scenario** (`src/cpp/tools/neural_trace.cpp`) — 7th `neural_trace`
  scenario; single-compartment body-wall muscle cell from `openworm/muscle_model`
  NeuroML2 (`SingleCompMuscle.cell.nml`): KSLOW_BC (g=31.543 nS, E=−64.35 mV),
  KFAST_BC (g=28.950 nS, E=−55.00 mV), CA_BOYLE (g=15.938 nS, E=+49.11 mV), LEAK_BC
  (g=1.399 nS, E=+10 mV); C_m=72.38 pF (π×10 µm×230.35 µm×1 µF/cm²).
  Protocol: 500 ms pre-settle at I_offset=−120 pA, then −120+I_pulse for 20 ms at t=5 ms;
  outputs CSV `t,V,n,p,q,e,f`. Ca²⁺ pool approximated as h=1 (no CaPool).
- **`notebooks/project_tour.ipynb`** — v0.5.1 section: build cell, demo plot (100/400/700 pA),
  CV-5.4 (muscle_trace vs 20 digitized points from Boyle & Cohen 2008 Fig. 2A;
  PASS: 100 pA max|ΔV|=4.5 mV, 400 pA Δpeak=12.5 mV, 700 pA AP occurs Δt_peak=0.6 ms),
  CV-5.5 cumulative regression (8/8 PASS). System CV extended to SCV-4 (4/4 ALL PASS).

---

## [0.5.0] - 2026-06-16

### Added

- **c302 C2 Full connectome data** (`data/c302/c302_C2_Full.net.nml`) — official OpenWorm
  c302 C2 reference connectome in NeuroML2 format: 302 GenericNeuronCell + 95 GenericMuscleCell
  populations (397 total), 1084 gap junctions, 2279 chemical synapses, 552 NMJs.
- **`ca_boyle` channel** (`src/cpp/src/neural/ChannelKinetics.cpp`) — voltage-gated Ca²⁺
  (Boyle & Cohen 2008); two-gate model: activation gate e² (v_half=−3.36 mV, k=6.748 mV,
  τ=0.100 ms) × inactivation gate f (v_half=25.18 mV, k=−5.032 mV, τ=150.88 ms); E_rev=10 mV.
  Also added c302 channel name aliases: `k_slow`, `k_fast`, `Leak`, `ca_boyle` and their
  `*_muscle` variants.
- **`cell_type` field** (`src/cpp/include/io/NetworkConfig.h` `NeuronDef`) — stores the
  NeuroML2 component id (e.g. `"GenericNeuronCell"`, `"GenericMuscleCell"`) so biophysics and
  channel assignments are applied per cell type.
- **Cell-type-aware biophysics parser** (`src/cpp/src/io/NeuroMLLoader.cpp`) — `extractCellBlock`
  + `applyCellBiophysics` lambdas ensure neuron and muscle channel densities + capacitances
  are assigned independently; `resolveCellRef()` resolves c302 C2 path form
  `../POP/0/GenericNeuronCell` to a `NeuronDef` id.
- **`connectome_trace` CLI tool** (`src/cpp/tools/connectome_trace.cpp`) — usage:
  `connectome_trace <nml_file> <T_ms> <dt_ms> <neuron_names>` (comma-separated);
  outputs CSV voltage traces on stdout, load summary on stderr.
- **`test_connectome_load` integration test** (`src/cpp/tests/io/test_connectome_load.cpp`) —
  5 assertions: population count (397), connectivity (≥1000 GJ, ≥1500 chem synapses, valid
  id ranges), biophysics (AVAL[0]: 4 channels, C_m≈5 pF, cell_type=GenericNeuronCell),
  key-neuron existence (AVAL[0], AVBL[0]), integrator stability (100 steps, no NaN).
- **`notebooks/project_tour.ipynb`** — v0.5.0 section: build + test output cell (all 5 tests
  PASS) and 500 ms voltage-trace plot for AVAL, AVAR, AVBL, AVBR, DB1, VB1 interneurons.

---

## [0.4.1] - 2026-06-16

### Added

- **Boyle & Cohen 2008 channel catalog** (`src/cpp/src/neural/ChannelKinetics.cpp`) —
  three new channels with parameters fetched from `openworm/CElegansNeuroML` NMODL files:
  `KSLOW_BC` (k_slow, 1 activation gate n, constant τ=25.0 ms), `KFAST_BC` (k_fast,
  activation p^4 + inactivation q, constant τ=2.26/150.0 ms), `LEAK_BC` (passive, E=−50 mV).
- **`boyle2008` scenario** (`src/cpp/tools/neural_trace.cpp`) — 6th neural_trace scenario;
  runs KSLOW_BC + KFAST_BC + LEAK_BC single-compartment neuron with current pulse; outputs
  CSV columns `t,V,n,p,q`.
- **Cross-validation cells in `notebooks/project_tour.ipynb`** (CV-1 + CV-2):
  - CV-1: fetches k_slow.mod and k_fast.mod live from openworm/CElegansNeuroML, parses
    NMODL parameters, runs C++ backward-Euler integrator (`neural_trace boyle2008`) and
    scipy RK45 (same published ODE, same parameters) side-by-side; max |ΔV| = 0.17 mV —
    consistent with first-order backward Euler at dt = 0.025 ms.
  - CV-2: computes x_inf(V) analytically from the parsed NMODL Boltzmann formula and
    compares to C++ `chan_kinetics` output for KSLOW_BC/KFAST_BC; max deviation ≤ 2×10⁻⁶
    confirming floating-point round-trip fidelity between NMODL source and C++ catalog.

---

## [0.4.0] - 2026-06-15

### Added

- **`neural/` HH ODE integrator** (`src/cpp/neural/`) — FR-SIM-01:
  - `ChannelKinetics.h/.cpp` — built-in c302 channel catalog (NCA, KD, KA, KQS, KVS, IR)
    with Boltzmann steady-state (`x_inf`) and Gaussian tau parameterisation; pure-function
    `rush_larsen()` for exact exponential gate update at constant V over dt.
  - `NeuralState.h/.cpp` — SoA state arrays (`v[]`, `gate[n×stride]`, `s_syn[]`);
    GPU-friendly layout (neuron-major; CUDA backend will transpose to gate-major).
  - `NeuralIntegrator.h/.cpp` — single-compartment HH over the full connectome:
    Rush–Larsen gate update; graded chemical synapse gating (Rush–Larsen on `s_inf(V_pre)`
    via sigmoid threshold); conductance-method voltage update (implicit diagonal, explicit
    gap-junction cross-terms); voltage clamped to [−150, 80] mV.
  - `wormsim2_neural` static library; links `PUBLIC wormsim2_io` (inherits `NetworkConfig`).
- **Neural test suite** (`src/cpp/tests/neural/`):
  - `test_hh_single` — 4 assertions: leak at reversal (V invariant), exponential decay
    vs analytic (conductance method exact for pure leak), KD gate direction, I_ext depolarisation.
  - `test_gap_junction` — 3 assertions: mean-voltage conservation (analytic), voltage-difference
    exponential decay vs analytic, convergence over 500 ms.
  - Both pass (4/4 + 3/3 assertions); registered as CTest label `neural`.

### Changed

- **`README.md`** — added direct links to `docs/requirements/01-requirements-analysis.md`,
  `docs/design/02-architecture-design.md`, and `notebooks/project_tour.ipynb` under the
  scientific charter line; added author contact email (`seyedvahid.ghayoomie@mail.polimi.it`)
  to the License section; added `notebooks/` to repo layout.
- **`notebooks/project_tour.ipynb`** *(new)* — version-by-version project tour notebook with
  one section per MINOR/MAJOR release (v0.1.0–v0.4.0); each section describes what was added
  and provides a runnable Python input→output example. The v0.4.0 section includes a full
  Python reference implementation of the channel kinetics + conductance-method integrator
  (mirrors `ChannelKinetics.h/cpp` and `NeuralIntegrator.cpp` exactly) with four runnable
  examples: NCA leak decay (analytic match), KD gate activation, gap-junction coupling
  (V_avg conservation + exponential V_diff decay), and multi-channel neuron with current pulse.

### Removed

- **`docs/cpp-guidelines.md`** — superseded by `src/cpp/tests/static/cpp_quality_checker.py`
  (66 automated checks, CI-enforced) and the internal lecture-mapping review cycle. NFR-QUAL-01
  now points to the checker.

---

## [0.3.0] - 2026-06-15

### Added

- **GitHub Actions CI** (`.github/workflows/`):
  - `ci-baseline.yml` — on every push/PR: cmake build (C++20, Ninja, CPU backend),
    CTest (all tests including `cpp_quality`), cppcheck NFR-QUAL-01, clang-tidy,
    code coverage (lcov/gcov), upload of all evidence artifacts.
  - `auto-release.yml` — fires on `v*` tag push; extracts matching CHANGELOG section
    and creates a GitHub Release automatically (+ manual `workflow_dispatch` override).
  - `backfill-releases.yml` — one-shot manual workflow to create GitHub Releases for
    all existing `v*` tags that have no release yet; dry-run mode supported.
- **C++ quality checker** (`src/cpp/tests/static/cpp_quality_checker.py`) — 66-check
  Python static analyser verifying C++20 idioms and best practices across the source
  tree; runs as CTest target `cpp_quality`; exit-1 on first regression.
  Check categories: TC (toolchain: cmake/pragma/flags), FP (float safety: bio params,
  NaN guard), NS (namespaces: wormsim2 ns, no using-std in headers, string_view),
  FN (functions: nodiscard/maybe_unused/lambdas/IIFE/SRP/defaults), RA (RAII: no bare
  new/delete, ifstream, optional), CL (classes: Rule of Zero, const noexcept,
  static-method classes, enum class), TM (templates: unordered_map, vector<T> diversity,
  sregex_iterator), SL (stdlib: transform/directory_iterator/starts_with/
  istreambuf_iterator), LB (libraries: static lib/PUBLIC/PRIVATE CMake), PS (parallel
  safety: no mutable/global state, immutable config), OL (object layout: no virtual in
  data structs, flat integer IDs), DS (data safety: const params, no shared state).
- `src/cpp/tests/static/CMakeLists.txt` — wires the Python checker into CTest with
  `find_package(Python3)` and a 30 s timeout; label `static`.
- Updated `src/cpp/tests/CMakeLists.txt` — added `add_subdirectory(static)`.

---

## [0.2.0] - 2026-06-15

### Added

- **`io/` network loading module** — dual-format loader implementing FR-NET-01/02/03:
  - `NetworkConfig` (`src/cpp/include/io/NetworkConfig.h`) — unified in-memory structs
    (`GateVar`, `ChannelDef`, `ChannelAssignment`, `NeuronDef`, `ChemSynapseDef`,
    `GapJunctionDef`, `NMJDef`, `NetworkConfig`) with Rule of Zero, `[[nodiscard]]`
    accessors, `find_neuron(string_view)` observing-pointer lookup.
  - `NEURONLoader` — NMODL `.mod` block parser (NEURON/PARAMETER/STATE blocks,
    `sregex_iterator`, anonymous-namespace helpers); `parseModDir` filesystem scan;
    `loadHoc` stub directing users to jNeuroML pre-conversion (ISSUE-009).
  - `NeuroMLLoader` — minimal hand-written XML attribute scanner for the c302 NeuroML2
    output subset; parses populations, channel densities, electrical projections (gap
    junctions), continuous projections (chemical synapses + NMJs); resolves cell
    references via `find_neuron`.
  - `NetworkInputParser` — top-level dispatcher; auto-detects format from extension
    (`.nml`→NeuroML2, `.hoc`/`.mod`→NEURON); optional perturbation spec applied
    post-load; returns `NetworkConfig` by value (RVO).
  - `PerturbationConfig` — key=value spec file parser; supports
    `channel.<ID>.conductance_scale`, `gap_junction.conductance_scale`,
    `neuron.<NAME>.v_initial_mV`, `neuron.<NAME>.capacitance_nF`; uses C++20
    `string_view::starts_with`.
  - 2 CTest suites (100 % pass): `test_nmodl_parser` (9 assertions, multi-channel,
    gate variables, `find_neuron`); `test_perturbation` (7 assertions, channel scale,
    GJ scale, neuron override, unknown key, missing file).
- **C++20 build toolchain**:
  - Root `CMakeLists.txt` — `CMAKE_CXX_STANDARD 20`, `-Wall -Wextra -Wpedantic`,
    `WORMSIM2_BACKEND` cache option (CPU/CUDA/OPENCL), `enable_testing()`.
  - `src/cpp/CMakeLists.txt` — `wormsim2_io` static library with PUBLIC C++20 feature
    requirement; `target_include_directories(PUBLIC include)`.
- **Docker dev image** (`docker/Dockerfile.wormsim2-dev`) — block-structured Ubuntu
  22.04 image: Block 1 cmake/GCC, Block 2 OpenCL (mandatory — `opencl-headers` +
  `ocl-icd`), Block 3 cppcheck/clang-tidy; Blocks 4–7 (dealii, CUDA, Python,
  OpenMPI) commented for future activation.
- **`.clang-tidy`** — project clang-tidy configuration (modernize, readability,
  performance, bugprone, cppcoreguidelines checks).
- **`docs/cpp-guidelines.md`** — 11-section C++20 best-practice checklist
  (NFR-QUAL-01): language standard, RAII, const-correctness, type safety, algorithms,
  SoA data layouts, error handling, thread safety, static analysis, file organisation,
  documentation.
- **`docs/file-registry.md`** — inventory of every tracked file with per-file update
  triggers; must be consulted before every commit.
- **NFR-QUAL-01** added to `docs/requirements/01-requirements-analysis.md`.
- **ISSUE-001 updated** (`docs/ISSUES.md`) — OpenCL is now mandatory (not optional);
  reflects that it is the only GPU path available on the dev machine (Intel Iris Plus
  645 iGPU via macOS OpenCL.framework).

---

## [0.1.0] - 2026-06-15

### Added

- **Scientific charter** (`docs/research/00-motivation-objectives-related-work.md`) —
  motivation, related-work survey (Sibernetic, MetaWorm/BAAIWorm, prior wormuse work,
  ElegansBot), gap analysis, the proposed four-layer GPU-first closed-loop architecture
  with a web-friendly output layer, three-level validation strategy, and explicit
  objectives (G1–G7), engineering objectives (E1–E6), domain assumptions (DA1–DA5),
  and non-goals (N1–N4). Includes GPU runtime estimates and the c302/NEURON GPU
  bottleneck analysis.
- **Issue tracker** (`docs/ISSUES.md`) — `ISSUE-NNN` tracker for bugs, improvements,
  research questions, and design decisions. Seeded with 9 open items and 1 resolved
  (ISSUE-001: backend-agnostic compute core — CPU/OpenMP reference + CUDA deploy +
  optional OpenCL).
- **Requirements analysis** (`docs/requirements/01-requirements-analysis.md`) —
  stakeholders/actors, world–machine model, domain assumptions (DA1–DA5), functional
  requirements (FR-SIM, FR-COMPUTE, FR-OUT, FR-VALID, FR-DEPLOY), non-functional
  requirements (performance, portability, reproducibility, maintainability, scalability,
  web usability), five use cases (UC-01–UC-05), and G1–G7 + E2/E4 traceability matrix.
- **Architecture design** (`docs/design/02-architecture-design.md`) —
  multi-view architecture (Component & Connector, Module, Deployment); 14 components
  including ChannelLoader (NMODL `.mod` parsing) and ProprioceptiveFeedback;
  backend-agnostic `ComputeBackend` kernel interface sketch; 10 architectural
  verification gates (VR-01–VR-10); six design principles; open design issues mapped.
- **Charter update** (§2.1) — "Runtime estimates" header clarified; added NEURON `.mod`
  file reuse paragraph (NCA/KA/KD/KQS/KVS/IR channels as GPU-HH parameter source;
  ISSUE-009); §4.1 updated.
- **Project scaffold** — `README.md`, `CHANGELOG.md`, `VERSION`, `LICENSE` (MIT),
  `.gitignore`, and the `src/` / `docs/` / `config/` directory skeleton.

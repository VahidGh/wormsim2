# Changelog

All notable changes to **wormsim2** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Versioning policy**
> - `MAJOR` (1.x.x): first public release; breaking API changes after 1.0
> - `MINOR` (0.x.0): new module, new capability, or significant doc milestone
> - `PATCH` (0.x.y): bug fixes, test additions, minor doc corrections within a milestone

---

## [0.2.0] - 2026-06-15 *(current)*

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

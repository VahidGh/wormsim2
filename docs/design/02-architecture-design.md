# wormsim2 — Architecture Design

> **Document type:** Architecture design (professional software engineering artifact).
> **Status:** Draft (v0.1.0-dev).
> **Audience:** Software / HPC engineers; computational neuroscience readers.
> **Scope:** Multi-view architecture (Component & Connector, Module, Deployment); design
> decisions and their rationale; interface specifications; V&V gates. Scientific motivation
> is in `docs/research/00-motivation-objectives-related-work.md`; requirements are in
> `docs/requirements/01-requirements-analysis.md`.

---

## 1. Architectural Overview

wormsim2 is structured as a **layered, event-driven data-flow pipeline** with a
closed feedback loop. A single simulation tick drives data top-to-bottom through five
simulation layers; proprioceptive feedback couples layer 5 back to layer 1.
Network configuration (topology, channels, perturbations) is loaded once at startup from
the user-supplied network model files — either NEURON (`.hoc` + `.mod`) or NeuroML2
(`.nml`). Output serialisation runs asynchronously from the main loop.

```
[Startup — network configuration]
  NEURON .hoc + .mod  ─┐
                        ├─► NetworkInputParser (FR-NET-01, FR-NET-02)
  NeuroML2 .nml       ─┘        │ PerturbationConfig (FR-NET-03)
                                 ▼
  NetworkConfig struct (topology, channel params, synapse weights, gap-junction g)

[Per-tick simulation pipeline]
┌──────────────────────────────────────────────────────────────────────┐
│ (1) Neural layer — GPU-batched Hodgkin–Huxley                          │
│     N neurons × compartments · chemical synapses + gap junctions       │
│     Channel kinetics, synapse params from NetworkConfig                │
├──────────────────────────────────────────────────────────────────────┤
│ (2) Neuromuscular mapping — motoneuron V → 95 BWM activations          │
├──────────────────────────────────────────────────────────────────────┤
│ (3) Body mechanics — corotated linear-elastic FEM                      │
│     Large-rotation body, per-quadrant muscle drive                     │
├──────────────────────────────────────────────────────────────────────┤
│ (4) Environment — agar (crawl) ↔ liquid (swim), contact + drag         │
└──────────────────────────────────────────────────────────────────────┘
        ▲                                                      │
        └──── (5) Proprioceptive feedback: body curvature ─────┘
                  → stretch-receptor channels → motor circuit
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ (6) Output layer — asynchronous serialiser                             │
│     spine + 95-muscle activation time series (JSON / binary)           │
│     → WCON exporter (validation) · → browser 3D viewer (exploration)  │
└──────────────────────────────────────────────────────────────────────┘
```

**Key architectural decisions:**

| Decision | Choice | Rationale |
|---|---|---|
| Network input | Dual-format parser (NEURON or NeuroML2) → `NetworkConfig` | User supplies either format; both produce identical `NetworkConfig`; full perturbability without a hardcoded NN; general across connectomes (FR-NET-01 – 04, ISSUE-009) |
| Body model | Corotated linear-elastic FEM | Handles large rotations (omega-turns); MetaWorm validates this approach in real time (§2.2 charter) |
| Neural model | GPU-batched HH ODEs over NetworkConfig | Biophysical fidelity (G1, G5); parameters loaded from user-supplied NEURON or NeuroML2 files at runtime |
| Backend | Backend-agnostic kernel interface | Dev/CI on CPU/OpenMP; NVIDIA deploy via CUDA; optional OpenCL (ISSUE-001) |
| Proprioceptive loop | Layer 5 → layer 1 feedback | Core contribution (G3); not present in Sibernetic or MetaWorm |
| Output decoupling | Output schema between engine and viewer | Allows viewer to evolve independently; enables WCON export without engine changes (G7) |
| Data flow style | Synchronous pipeline per tick + async output | Predictable latency per frame; output write never blocks the simulation loop |

---

## 2. Component & Connector View

### 2.1 Components

| Component | Responsibility | Primary FRs |
|---|---|---|
| **SimulationEngine** | Top-level coordinator; loads network at startup; runs the per-tick pipeline; owns simulation state | FR-NET-01 – 04, FR-SIM-01 – 06 |
| **NetworkInputParser** | Format-dispatching loader: detects whether the user supplied NEURON (`.hoc` + `.mod`) or NeuroML2 (`.nml`) input and delegates to the appropriate sub-loader; produces an identical `NetworkConfig` from either format (FR-NET-01). Format is detected by file extension or an explicit `--format` flag. | FR-NET-01, FR-NET-04, ISSUE-009, ISSUE-011 |
| **NEURONLoader** | Sub-loader for NEURON format: parses `.hoc` network description (neuron objects, synapse connectivity, gap junctions) and NMODL `.mod` files (HH gate-variable ODEs, α/β rate functions, reversal potentials, per-neuron conductance densities); populates `NetworkConfig`. | FR-NET-01, FR-NET-02 |
| **NeuroMLLoader** | Sub-loader for NeuroML2 format: parses `.nml` XML (neuron list, compartment geometry, synapse parameters, gap-junction conductances, ion-channel references); resolves channel kinetics from referenced `.mod` files or inline ChannelML definitions; populates `NetworkConfig`. | FR-NET-01, FR-NET-02 |
| **PerturbationConfig** | Reads network-level perturbation specifications from the run configuration file; applies overrides to `NetworkConfig` before simulation starts: ion-channel conductance scalings, synapse weight modifications, gap-junction conductance changes, and neuron excitability parameter overrides | FR-NET-03 |
| **NeuralIntegrator** | Hodgkin–Huxley ODE integration over the N-neuron network using the `NetworkConfig` (chemical synapses + gap junctions); batched on the selected compute backend | FR-SIM-01 |
| **NeuromuscularMapper** | Maps motoneuron membrane potentials to 95 BWM activation values (linear/sigmoid transfer, per anatomical neuromuscular connectivity in `NetworkConfig`) | FR-SIM-02 |
| **FEMBody** | Corotated linear-elastic FEM solver; per-quadrant muscle drive; implicit time integration for stability at real-time dt | FR-SIM-03 |
| **EnvironmentModel** | Agar/liquid drag and contact force models; selects environment via configuration; drives the gait-transition regime | FR-SIM-04 |
| **ProprioceptiveFeedback** | Computes local body curvature from FEM body state; updates stretch-receptor channel conductances fed into the motor circuit (closes the sensorimotor loop) | FR-SIM-05 |
| **ComputeBackend** (interface) | Thin kernel-dispatch interface; implementations: `CpuOpenMpBackend`, `CudaBackend`, `OpenClBackend` | FR-COMPUTE-01 – 05 |
| **OutputSerialiser** | Writes per-frame spine + muscle-activation data to the documented output schema (JSON / binary); runs asynchronously from the simulation loop | FR-OUT-01 |
| **WconExporter** | Converts output schema to WCON format; produces WCON files for the validation pipeline | FR-OUT-02 |
| **BrowserViewer** | Self-contained HTML/JS 3D viewer; consumes the output schema; no server, no build step required | FR-OUT-03 |
| **ValidationPipeline** | Python analysis layer; invokes open-worm-analysis-toolbox; produces per-feature comparison report against real-worm distributions | FR-VALID-01, FR-VALID-02 |

### 2.2 Connectors and data flow

```
[Startup]
NEURON .hoc + .mod ─┐
                     ├─► NetworkInputParser ──► NetworkConfig
NeuroML2 .nml      ─┘    (NEURONLoader or
                           NeuroMLLoader)
PerturbationSpec ────────► PerturbationConfig ──► (mutates NetworkConfig)

[Per tick]
NetworkConfig ──────────────────────────────────────────►┐
ProprioceptiveFeedback (I_stretch from previous tick) ──►│
                                                         │
                                              NeuralIntegrator
                                        (HH ODEs, batched on ComputeBackend)
                                                         │
                                                         ▼
                                              NeuromuscularMapper
                                              (95 BWM activations)
                                                         │
                                                         ▼
                                                     FEMBody ◄── EnvironmentModel
                                                     (corotated FEM solve)
                                                         │
                                      ┌──────────────────┘
                                      │
                                      ├──► ProprioceptiveFeedback (curvature → next tick)
                                      │
                                      └──► OutputSerialiser (async)
                                                 │
                                       ┌─────────┴────────┐
                                       │                  │
                                  WconExporter      BrowserViewer
                                  (WCON file)       (HTML viewer)
                                       │
                                ValidationPipeline
                                (Python, offline)
```

### 2.3 Provided and required interfaces

| Component | Provides | Requires |
|---|---|---|
| NetworkInputParser | `NetworkConfig load(input_path, format)` | Filesystem; dispatches to NEURONLoader or NeuroMLLoader |
| NEURONLoader | `NetworkConfig load_hoc(hoc_path, mod_dir)` | `.hoc` + `.mod` files on filesystem |
| NeuroMLLoader | `NetworkConfig load_nml(nml_path)` | `.nml` + referenced `.mod`/ChannelML files |
| PerturbationConfig | `void apply(NetworkConfig&, spec_file)` | `NetworkConfig` from NetworkInputParser |
| NeuralIntegrator | `integrate(dt, V[], I_ext[]) → V_new[]` | ComputeBackend kernel; `NetworkConfig` (channel params, synapse graph) |
| FEMBody | `solve(dt, activation[95]) → positions[]` | ComputeBackend kernel; EnvironmentModel forces |
| ProprioceptiveFeedback | `curvature(positions[]) → I_stretch[]` | FEMBody positions |
| OutputSerialiser | `write(frame, positions[], activation[])` | Filesystem; output schema spec |
| WconExporter | `export(schema_file) → .wcon` | OutputSerialiser output |
| ComputeBackend | `dispatch_kernel(fn, data)` | GPU driver (CUDA / OpenCL) or CPU threads |
| ValidationPipeline | `compare(wcon_sim, wcon_realworm) → report` | open-worm-analysis-toolbox (Python) |

---

## 3. Module View

### 3.1 Source tree layout

```
wormsim2/
├── src/
│   ├── cpp/
│   │   ├── include/
│   │   │   ├── io/             ← NetworkInputParser, NEURONLoader, NeuroMLLoader, PerturbationConfig
│   │   │   ├── neural/         ← NeuralIntegrator, NetworkConfig struct
│   │   │   ├── neuromusculcar/ ← NeuromuscularMapper
│   │   │   ├── body/           ← FEMBody, EnvironmentModel
│   │   │   ├── feedback/       ← ProprioceptiveFeedback
│   │   │   ├── backend/        ← ComputeBackend interface + CpuOpenMP, CudaBackend, OpenClBackend
│   │   │   ├── output/         ← OutputSerialiser, WconExporter
│   │   │   └── sim/            ← SimulationEngine
│   │   ├── src/                ← .cpp / .cu implementation files (mirror include/ tree)
│   │   └── tests/              ← CTest unit + integration tests (mirror include/ tree)
│   └── python/
│       └── notebooks/          ← ValidationPipeline notebooks (Jupyter)
├── viewer/                     ← BrowserViewer (HTML + JS; self-contained)
├── config/                     ← Example configuration + perturbation spec files
└── docs/                       ← research/, requirements/, design/, ISSUES.md
```

### 3.2 Module dependency rules

- `io/` (NetworkInputParser, NEURONLoader, NeuroMLLoader, PerturbationConfig) depends on
  no simulation layer; it only produces `NetworkConfig` structs and is independently
  testable with synthetic `.hoc`/`.mod` or `.nml` fixtures.
- Modules in `neural/`, `body/`, `feedback/`, and `output/` shall depend **only** on the
  `backend/` interface — never on a specific backend implementation.
- The `backend/` interface header shall include no CUDA or OpenCL headers; implementations
  are in separate `.cu` / `.cl` files selected at build time.
- Python (`src/python/`) consumes only the output schema files and WCON exports; it does
  not link against the C++ library.
- The `viewer/` directory is fully standalone (no CMake target dependency on the C++ core).

---

## 4. Deployment View

### 4.1 Developer machine (Intel Mac, no NVIDIA)

```
[macOS / Linux machine]
  CMake build (CPU/OpenMP backend only)
  CTest unit + integration tests (including NeuroMLParser + PerturbationConfig tests)
  ← No CUDA, no OpenCL required for correctness testing
```

### 4.2 NVIDIA workstation / cluster node

```
[Linux x86-64, CUDA ≥ 12.0]
  CMake build (CUDA backend selected)
  CUDA kernels compiled by nvcc
  Real-time simulation: ≤ 1 ms/frame neural + ≤ 10 ms/frame FEM (NFR-PERF-01/02)
```

### 4.3 HPC batch (Apptainer / SLURM)

```
[HPC cluster, NVIDIA GPU nodes]
  Apptainer image (built from .def file; includes NeuroML2/NMODL input files)
  SLURM job array → N independent parameter-sweep runs (each with own perturbation spec)
  Outputs → shared filesystem → transferred for offline analysis
```

### 4.4 Validation (offline Python)

```
[Any machine with Python environment]
  ValidationPipeline reads WCON files from shared filesystem
  Invokes open-worm-analysis-toolbox
  Produces per-feature comparison reports
```

### 4.5 Browser viewer (any device)

```
[Any device with Chrome / Firefox ≥ 2023]
  User opens viewer/index.html (local file or static HTTP)
  Loads output schema file (JSON/binary)
  Renders 3D animated simulation — no install, no server
```

---

## 5. Backend-Agnostic Kernel Interface

The `ComputeBackend` interface (ISSUE-001) is the central portability seam. The design
rule (NFR-MAINT-02) is that all simulation logic (neural ODE integration, FEM linear
solve) is expressed against this interface; backend implementations are selected at build
time via CMake options.

### 5.1 Interface sketch

```cpp
// include/backend/ComputeBackend.h
class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;

    // Integrate one HH ODE step for `n_compartments` state variables.
    // state: [V, m, h, n, ...] per compartment, updated in-place.
    // params: channel + synapse parameters from NetworkConfig.
    virtual void integrate_hh(float* state, const float* params,
                               int n_compartments, float dt) = 0;

    // Sparse symmetric linear solve: A x = b (FEM system).
    // Returns x in-place.
    virtual void sparse_solve(const int* row_ptr, const int* col_idx,
                               const float* vals, float* x, const float* b,
                               int n, int nnz) = 0;
};
```

### 5.2 Backend implementations

| Class | Backend | Build flag | Notes |
|---|---|---|---|
| `CpuOpenMpBackend` | CPU threads (OpenMP) | default / `-DBACKEND=CPU` | Always available; primary dev + CI target |
| `CudaBackend` | NVIDIA CUDA | `-DBACKEND=CUDA` | Requires CUDA ≥ 12.0; nvcc; HH kernels are embarrassingly parallel over compartments |
| `OpenClBackend` | OpenCL 1.2+ | `-DBACKEND=OPENCL` | Optional; enables Mac iGPU and AMD/Intel GPU execution |

The sub-decision of whether to replace this hand-rolled interface with a portability
layer (Kokkos / SYCL) is deferred to ISSUE-010.

---

## 6. Key Design Principles

1. **Separation of concerns:** network I/O (NeuroMLParser, PerturbationConfig), simulation
   logic, compute kernels, output serialisation, and visualisation are in separate components
   with narrow interfaces.
2. **Abstraction / information hiding:** the backend interface hides GPU/threading details
   from the rest of the system (rule: no CUDA header in non-backend includes); the
   NeuroMLParser hides file format details from the simulation engine.
3. **Single responsibility:** each component has one reason to change (e.g. NeuroMLParser
   changes only if the input format changes; FEMBody changes only if the body mechanics
   model changes).
4. **Dependency inversion:** high-level components (NeuralIntegrator, FEMBody) depend on
   the ComputeBackend *abstraction*, not on concrete CUDA / OpenCL implementations.
5. **Explicit data flow:** all inter-component communication is via function arguments,
   `NetworkConfig` structs, or documented schema files — no hidden global state except the
   main simulation state struct owned by SimulationEngine.
6. **Testability:** every component can be instantiated with the CPU backend in a CTest
   unit test on any machine; NEURONLoader and NeuroMLLoader tests use minimal synthetic
   `.hoc`/`.mod` and `.nml` fixtures respectively; there are no GPU-only code paths that
   bypass testing.

---

## 7. Architectural Verification Gates

These are the concrete checks that verify the design is met before a version is released:

| Gate | Check | Artefact |
|---|---|---|
| **VR-01** | CPU/OpenMP backend builds and all unit tests pass on Linux + macOS without a GPU | CI: `cmake --build` + `ctest -L unit` log |
| **VR-02** | CUDA backend builds and the neural integration test produces the same result as the CPU backend (within float tolerance) | CI: `ctest -L backend-parity` log |
| **VR-03** | Real-time budget: neural step < 1 ms/frame and FEM step < 10 ms/frame on the NVIDIA reference GPU | Benchmark report in `docs/perf/` |
| **VR-04** | Proprioceptive-loop positive control: closed-loop run sustains undulation for ≥ 5 s; open-loop run with same initial condition decays within 1 s | Test report: `tests/integration/test_proprioception` |
| **VR-05** | Biomechanical metrics (velocity, wavelength, frequency) fall within Fang-Yen et al. 2010 experimental ranges | Validation report in `docs/validation/` |
| **VR-06** | 256-feature battery: ≥ 80% of features fall within wild-type N2 distribution bounds (KS p > 0.05) | `docs/validation/256feature_report.html` |
| **VR-07** | Browser viewer loads a 30 s simulation and begins rendering in < 5 s (Chrome / Firefox) | Manual verification note in release notes |
| **VR-08** | WCON export round-trips: simulated WCON parses without error in the open-worm-analysis-toolbox | CI: `pytest tests/python/test_wcon_export.py` |
| **VR-09** | Network-level perturbation (e.g. zero KD conductance, 50% synapse weight scaling) produces a statistically distinct phenotype from wild-type in the 256-feature report | Experiment report in `docs/validation/` |
| **VR-10** | Container (Apptainer .def) builds and runs the simulation to completion on a Linux x86-64 node | CI + manual HPC deployment note |
| **VR-11** | Parser round-trip (both formats): (a) parse c302 `.hoc`/`.mod` via NEURONLoader; (b) parse c302 `.nml` via NeuroMLLoader; both resulting `NetworkConfig` objects must agree on neuron count, synapse count, gap-junction count, and representative channel parameters | CI: `ctest -L io-parser` log |

---

## 8. Open Design Issues

| ISSUE | Impact on design |
|---|---|
| ISSUE-002 | Body mesh choice determines the FEMBody mesh-input interface and the muscle-attachment geometry in `NetworkConfig` |
| ISSUE-003 | Neural granularity (single vs multi-compartment) determines the `integrate_hh` state vector size and per-neuron compartment graph in `NetworkConfig` |
| ISSUE-004 | Proprioceptive coupling topology determines the ProprioceptiveFeedback → NeuralIntegrator interface (which neurons receive I_stretch, what gain) |
| ISSUE-005 | Output schema format (JSON vs binary threshold) and viewer rendering stack (Canvas2D vs WebGL) determine OutputSerialiser and BrowserViewer implementations |
| ISSUE-009 | Parser implementation strategy for both formats (NEURONLoader + NeuroMLLoader): native C++ vs Python jNeuroML/libNeuroML as a pre-processing step producing serialised `NetworkConfig`, or hybrid. Affects `io/` module, build dependencies, and startup latency. |
| ISSUE-010 | Backend interface (hand-rolled vs Kokkos/SYCL) determines the `ComputeBackend` implementation strategy and CMake build structure |
| ISSUE-011 | Conformance scope of NEURONLoader and NeuroMLLoader (FR-NET-04): which NEURON/NeuroML2 features are required beyond c302? Determines test surface and what networks the parsers accept. |

See [docs/ISSUES.md](../ISSUES.md) for full details and current status.

---

*This document drives the implementation work in `src/cpp/` and the test plan in `src/cpp/tests/`.*

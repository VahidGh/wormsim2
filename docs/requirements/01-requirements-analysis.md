# wormsim2 — Requirements Analysis

> **Document type:** Requirements analysis (professional software engineering artifact).
> **Status:** Draft (v0.1.0-dev).
> **Audience:** Software / HPC engineers; computational neuroscience readers.
> **Scope:** Stakeholders, world phenomena, machine requirements, functional requirements,
> non-functional requirements, domain assumptions, use cases, and traceability to project
> objectives (G1–G7). Scientific motivation, related work, and objectives definitions are in
> `docs/research/00-motivation-objectives-related-work.md`.

---

## 1. Scope and Context

wormsim2 is a real-time, biophysically-faithful *C. elegans* locomotion simulator. The
system occupies the previously unoccupied intersection of **Hodgkin–Huxley ion-channel
nervous system** and **real-time GPU execution**: an intersection that existing simulators
(Sibernetic: faithful but ~5×10⁴× slower than real time; MetaWorm: real-time but trained-NN
nervous system) do not reach. The system closes the **proprioceptive sensorimotor loop**,
validates quantitatively against real-worm behavioural data, and exposes results through a
**web-friendly browser 3D viewer**.

A key design principle distinguishes wormsim2 from both reference simulators: the nervous
system model is **loaded at runtime from standard NeuroML2 / NMODL files** — not a
trained neural network (MetaWorm) and not bound to the NEURON runtime (Sibernetic / c302).
This gives full network-level perturbability (ion channels, synaptic weights, gap-junction
conductances) and makes the simulator general beyond *C. elegans*.

The full scientific context, gap analysis, and rationale for every objective are in the
[scientific charter](../research/00-motivation-objectives-related-work.md).

---

## 2. Stakeholders and Actors

### 2.1 Stakeholder categories

| Stakeholder | Interest |
|---|---|
| **Computational neuroscience researcher** | Run closed-loop *C. elegans* simulations, study proprioceptive feedback, compare results to real-worm data |
| **Biophysicist / pharmacologist** | Apply ion-channel knock-outs, synaptic weight changes, and other network perturbations; interpret predicted phenotypes |
| **Software / HPC engineer** | Build, test, and deploy the system on commodity workstations and HPC clusters |
| **Validator / data scientist** | Run the 256-feature behavioural phenotyping pipeline on simulated output and compare to real-worm distributions |
| **Web user / collaborator** | Explore simulated locomotion interactively in a browser 3D viewer without installing native software |

### 2.2 Active actors (interact with the system at runtime)

| Actor | Interaction |
|---|---|
| **Researcher** | Configures and launches simulation runs; specifies network perturbations; interprets output |
| **Validator** | Consumes WCON export files; runs open-worm-analysis-toolbox; compares feature distributions |
| **Web viewer user** | Loads the browser 3D viewer; explores spine and muscle-activation time series interactively |
| **HPC scheduler** | Submits batch simulation jobs via SLURM; manages resource allocation |

### 2.3 External systems

| System | Role |
|---|---|
| **NEURON format input (`.hoc` + `.mod`)** | Native c302 network description: `.hoc` file (neuron objects, synapse connectivity, gap junctions) + NMODL `.mod` files (HH gate-variable ODEs, α/β rate functions, reversal potentials, per-neuron conductance densities for NCA, KA, KD, KQS, KVS, IR, …). First-class input format for wormsim2. |
| **NeuroML2 format input (`.nml`)** | XML network description: neuron list, compartment geometry, chemical synapse parameters (max conductance, reversal potential, rise/decay time constants), gap-junction conductances, ion-channel model references (resolved from `.mod` or inline ChannelML). c302 can export to this format via jNeuroML. First-class input format for wormsim2. |
| **MetaWorm / BAAIWorm FEM mesh** | Reference tetrahedral mesh geometry (984 vertices, 3,341 tetrahedra) — candidate body geometry input (ISSUE-002) |
| **Open Worm Movement Database** | Real-worm wild-type N2 spine tracks in WCON format (ground-truth validation data, DA4) |
| **open-worm-analysis-toolbox** | 256-feature behavioural phenotyping pipeline consuming WCON files |
| **GPU / HPC platform** | NVIDIA GPU (CUDA, real-time/deploy), CPU/OpenMP (development + CI), optional OpenCL |
| **Container runtime** | Apptainer / Docker for reproducible builds and cluster deployment (G6) |
| **CI platform (GitHub Actions)** | Automated build, unit/integration tests, static analysis |

---

## 3. World–Machine Model

Following the Zave–Jackson world–machine framework, we distinguish world phenomena
(observable facts about *C. elegans* biology and the computational environment) from
machine requirements (properties the software must satisfy):

### 3.1 World phenomena (shared with the machine)

- **WP1 — Connectome:** 302 neurons, ~7,000 chemical synapses, ~890 gap junctions,
  95 body-wall-muscle (BWM) map — known and fixed (DA1).
- **WP2 — Network model files:** c302 exposes the nervous system in two peer-reviewed,
  community-validated formats: **NEURON** (`.hoc` network + `.mod` channel kinetics) and
  **NeuroML2** (`.nml` — the standard XML exchange format). Both encode the same
  biophysical ground truth: synapse topology, gap junctions, and ion-channel kinetics (DA1).
- **WP3 — Undulatory gait:** forward crawling on agar is quasi-periodic (0.3–0.5 Hz);
  swimming is higher-frequency, lower-amplitude; omega-turns are large rotations (DA2).
- **WP4 — Proprioceptive hypothesis:** body curvature feeds back into the motor circuit
  through stretch-sensitive channels; open-loop c302 drive alone barely sustains
  undulation (wormuse H4 result, §2.3 of charter).
- **WP5 — Real-worm phenotype:** wild-type N2 *C. elegans* locomotion is characterised by
  the 256-feature open-worm-analysis-toolbox battery over the Open Worm Movement Database
  (DA4).
- **WP6 — Hardware constraints:** the real-time target is a single commodity GPU (RTX 3060
  class); HPC clusters are used for batch parameter studies (DA5).

### 3.2 Domain assumptions (DA — hold independently of the machine)

- **DA1** — The connectome topology, 95-muscle map, and the network model parameters
  (provided in NEURON or NeuroML2 format) are fixed known inputs; the machine reads them
  but does not derive them.
- **DA2** — Forward crawling on agar is quasi-periodic, so periodic drive plus proprioceptive
  feedback is a sufficient basis for sustained locomotion.
- **DA3** — Body-shape variance is low-rank (~4 eigenworm modes ≈ 95%); a reduced-order body
  model is physically justified.
- **DA4** — The Open Worm Movement Database WCON tracks and the 256-feature toolbox are
  the community-accepted behavioural ground truth.
- **DA5** — A commodity GPU (RTX 3060 class) is the reference target for the real-time claim;
  HPC/cluster use is for batch parameter studies, not the interactive path.

---

## 4. Functional Requirements

### 4.1 Network input and perturbation (FR-NET)

| ID | Requirement | Traces to |
|---|---|---|
| **FR-NET-01** | The system shall accept the user's neural network model in **either** of two formats and produce an identical internal `NetworkConfig` from each: (a) **NEURON format** — a `.hoc` network description (neuron objects, synapse connectivity, gap junctions) plus NMODL `.mod` channel-kinetics files; (b) **NeuroML2 format** — a `.nml` XML network description (neuron list, compartment geometry, synapse parameters, gap-junction conductances, ion-channel references resolved from `.mod` or inline ChannelML). The format is detected from the input file extension or an explicit configuration flag. | G1, G5, WP1, WP2, ISSUE-009 |
| **FR-NET-02** | For both input formats, the system shall extract the following into `NetworkConfig`: neuron list and compartment geometry; chemical synapse parameters (max conductance, reversal potential, rise/decay time constants); gap-junction conductances; ion-channel HH kinetics (gate-variable ODEs, α/β rate functions, reversal potentials, per-neuron conductance densities). | G1, G5, WP2 |
| **FR-NET-03** | The system shall accept **network-level perturbation** specifications via a configuration file, without code changes, covering: (a) ion-channel conductance scalings (knock-outs, pharmacological modifications, parameter mutations); (b) chemical synapse weight modifications; (c) gap-junction conductance modifications; (d) neuron excitability parameter overrides (membrane capacitance, resting potential). Perturbations are applied to `NetworkConfig` after parsing, regardless of the input format used. | G5 |
| **FR-NET-04** | Neither input format parser shall be hard-coded to c302; any conformant NEURON `.hoc`/`.mod` or NeuroML2 `.nml` description shall be accepted, making the simulator applicable beyond *C. elegans*. | G5, G6 |

### 4.2 Core simulation (FR-SIM)

| ID | Requirement | Traces to |
|---|---|---|
| **FR-SIM-01** | The system shall integrate a Hodgkin–Huxley ODE network over the connectome (chemical synapses + gap junctions) as configured by FR-NET-01 and FR-NET-02. | G1, WP1, WP2 |
| **FR-SIM-02** | The system shall maintain a 95-muscle activation map derived from motoneuron membrane potentials according to the established anatomical neuromuscular connectivity. | G1, WP1 |
| **FR-SIM-03** | The system shall integrate a corotated linear-elastic FEM body driven by per-quadrant muscle activations, supporting large rotations (omega-turns, swimming amplitudes). | G2 |
| **FR-SIM-04** | The system shall model at least two environments: high-drag agar (crawl) and low-drag liquid (swim), with a contact/drag model enabling the documented gait transition. | G2 |
| **FR-SIM-05** | The system shall implement a proprioceptive feedback path from body curvature to stretch-receptor channels in the motor circuit, closing the sensorimotor loop. | G3, WP4 |
| **FR-SIM-06** | The system shall support arbitrary-duration simulations (no hard upper bound on simulated time). | G4 |

### 4.3 Backend and compute (FR-COMPUTE)

| ID | Requirement | Traces to |
|---|---|---|
| **FR-COMPUTE-01** | The system shall provide a portable CPU/OpenMP reference backend that produces correct results on any x86-64 or ARM machine without a GPU (the primary development and CI target). | G6, E2 |
| **FR-COMPUTE-02** | The system shall provide a CUDA backend for NVIDIA GPUs that achieves (near) real-time throughput for the neural integration step on a commodity GPU (RTX 3060 class or equivalent). | G1, E2 |
| **FR-COMPUTE-03** | The system shall provide an OpenCL backend (optional) enabling execution on non-NVIDIA GPUs (e.g. integrated Intel/AMD devices). | E2 |
| **FR-COMPUTE-04** | All backends shall produce numerically identical results (within floating-point reproducibility limits) for the same configuration, verified by the CI suite. | G6, E4 |
| **FR-COMPUTE-05** | The backend selection shall be a build-time or runtime configuration option with no changes required to simulation logic or data files. | E2, ISSUE-001 |

### 4.4 Output and web viewer (FR-OUT)

| ID | Requirement | Traces to |
|---|---|---|
| **FR-OUT-01** | The system shall serialise simulation state — per-frame spine positions (x, y, z per body segment) and 95-muscle activations — to a documented, web-friendly format (JSON and/or binary; schema-versioned). | G7, E6 |
| **FR-OUT-02** | The system shall export simulation trajectories to **WCON** format compatible with the open-worm-analysis-toolbox. | G4 |
| **FR-OUT-03** | The system shall ship a self-contained browser-based 3D viewer (HTML + JavaScript, no server, no build step) that loads the FR-OUT-01 output format and renders animated spine + muscle activation. | G7, E6 |
| **FR-OUT-04** | The output schema (FR-OUT-01) and viewer (FR-OUT-03) shall be decoupled from the C++ engine core; the engine writes to the schema, the viewer reads from it. | G7 |

### 4.5 Validation pipeline (FR-VALID)

| ID | Requirement | Traces to |
|---|---|---|
| **FR-VALID-01** | Given a simulated trajectory in WCON format, the system shall support running the open-worm-analysis-toolbox 256-feature battery and producing a per-feature comparison report against the wild-type N2 distribution from the Open Worm Movement Database. | G4, WP5 |
| **FR-VALID-02** | The system shall reproduce the Fang-Yen et al. 2010 crawl/swim **velocity, wavelength, and frequency** values within the published experimental ranges on a nominal simulation run. | G4 |
| **FR-VALID-03** | The system shall demonstrate that closing the proprioceptive loop (FR-SIM-05) produces sustained undulation where open-loop drive alone does not (positive-control experiment). | G3 |
| **FR-VALID-04** | The system shall demonstrate at least one **network-level perturbation** (FR-NET-03) producing a predicted, interpretable change in the 256-feature phenotype. | G5 |

### 4.6 Deployment and reproducibility (FR-DEPLOY)

| ID | Requirement | Traces to |
|---|---|---|
| **FR-DEPLOY-01** | The system shall provide a container definition (Apptainer .def + Docker) enabling a reproducible build on any Linux x86-64 system. | G6, E4 |
| **FR-DEPLOY-02** | The system shall provide a SLURM job script for batch parameter-study deployment on HPC clusters. | G6, DA5 |
| **FR-DEPLOY-03** | The CI/CD pipeline (GitHub Actions) shall: build the CPU/OpenMP backend, run the full unit and integration test suite, run static analysis (cppcheck), and report coverage on every pull request. | G6, E4 |

---

## 5. Non-Functional Requirements

| ID | Requirement | Category | Traces to |
|---|---|---|---|
| **NFR-PERF-01** | The neural integration step (FR-SIM-01) shall complete in < 1 ms per simulation frame (dt = 0.025 ms, 30 FPS target) on an RTX 3060-class GPU. | Performance | G1, DA5 |
| **NFR-PERF-02** | The FEM body solve (FR-SIM-03) shall complete in < 10 ms per simulation frame on the same GPU target. | Performance | G1, G2 |
| **NFR-PERF-03** | The CPU/OpenMP reference backend shall produce a complete forward-crawl simulation of ≥ 1 s simulated time in ≤ 60 min wall-clock on the developer machine (Intel i5-8257U, 4 cores). | Performance | FR-COMPUTE-01 |
| **NFR-PARSE-01** | The NeuroML2 / NMODL parser (FR-NET-01, FR-NET-02) shall complete network loading in < 10 s for the c302 model on any supported platform. | Performance | FR-NET-01 |
| **NFR-PORT-01** | The codebase shall compile and pass all tests on Linux (x86-64) and macOS (Apple Silicon or x86-64 Intel) with GCC ≥ 11 or Clang ≥ 14 and CMake ≥ 3.22. | Portability | G6, E1 |
| **NFR-PORT-02** | The CPU/OpenMP backend shall require no proprietary SDK or GPU driver; the CUDA backend shall require CUDA ≥ 12.0. | Portability | G6, E2 |
| **NFR-REPR-01** | Given the same NeuroML2/NMODL inputs, configuration, and random seed, a simulation run shall produce bit-identical output across repeated executions on the same binary. | Reproducibility | G6, E4 |
| **NFR-REPR-02** | All published numerical results shall be reproducible from the tagged release and the supplied configuration and network files without manual intervention. | Reproducibility | G6 |
| **NFR-MAINT-01** | The codebase shall follow a traceable architecture: every functional requirement maps to at least one named component in the design document, and every component maps to at least one test. | Maintainability | G6, E4 |
| **NFR-MAINT-02** | The public API of the compute backends shall be specified through a documented kernel interface; adding a new backend shall not require changes to the neural/FEM simulation logic. | Maintainability | E2, ISSUE-001, ISSUE-010 |
| **NFR-QUAL-01** | All C++ source shall conform to the project's modern-C++ best-practice checklist (`docs/cpp-guidelines.md`), covering: C++20 or later idioms (concepts, ranges, `std::span`, `std::string_view::starts_with`, `std::format`, smart pointers, `std::optional`/`variant`, structured bindings, `if constexpr`, CTAD, `[[nodiscard]]`), RAII resource management (no raw `new`/`delete` outside custom allocators), const-correctness, cache-friendly data layouts (SoA for performance-critical arrays), thread-safety annotations on shared data, and static-analysis cleanliness (`cppcheck --std=c++20` and `clang-tidy` with the project `.clang-tidy` configuration). | Quality | G6, E1, E4 |
| **NFR-SCALE-01** | The SLURM batch path (FR-DEPLOY-02) shall support parameter sweeps of ≥ 100 independent simulation runs with no manual intervention beyond job submission. | Scalability | G6, DA5 |
| **NFR-WEB-01** | The browser 3D viewer (FR-OUT-03) shall load and begin rendering a 30 s simulation in < 5 s on a modern browser (Chrome/Firefox, 2023+) without a plugin or build step. | Usability | G7 |

---

## 6. Use Cases

### UC-01 — Run a locomotion simulation

**Actor:** Researcher  
**Pre-condition:** System is built (CPU/OpenMP or CUDA backend); network model files
(NEURON `.hoc`/`.mod` or NeuroML2 `.nml`) and a run configuration (environment, duration,
optional perturbations) are available.  
**Main flow:**

1. Researcher invokes the simulation executable with the configuration file.
2. The system loads and parses the network model files (NEURON or NeuroML2 format)
   and applies any perturbations from the configuration (FR-NET-01 – FR-NET-03).
3. The system initialises the corotated-FEM body from the mesh geometry (FR-SIM-03).
4. The system integrates the coupled neural/body/proprioceptive system for the requested
   duration (FR-SIM-01 – FR-SIM-05).
5. The system serialises the output to the documented schema (FR-OUT-01) and optionally
   exports WCON (FR-OUT-02).
6. The researcher inspects the output in the browser 3D viewer (UC-03) or proceeds to
   validation (UC-02).

**Alternative flow (HPC batch):** Researcher submits a SLURM job (FR-DEPLOY-02); the
scheduler executes the same simulation executable; outputs are written to a shared
filesystem.

**Post-condition:** Spine-position + muscle-activation time series written to the output
schema; WCON file optionally written.

---

### UC-02 — Validate simulated locomotion against real-worm data

**Actor:** Validator (may be the same person as Researcher)  
**Pre-condition:** A WCON export file from UC-01 exists; the Open Worm Movement Database
real-worm reference set is available locally; the open-worm-analysis-toolbox is installed
in the Python environment.  
**Main flow:**

1. Validator runs the Python validation notebook on the WCON file (FR-VALID-01).
2. The toolbox computes the 256-feature battery on both the simulated and real-worm tracks.
3. The notebook produces a per-feature comparison report (KS test / effect size vs. wild-type
   N2 distribution).
4. Validator reviews the report and cross-checks the biomechanical metrics
   (velocity, wavelength, frequency) against Fang-Yen et al. 2010 (FR-VALID-02).

**Post-condition:** Validation report produced; features within / outside wild-type
distribution are flagged.

---

### UC-03 — Explore simulation output in the browser viewer

**Actor:** Web viewer user  
**Pre-condition:** A simulation output file (FR-OUT-01 schema) is accessible on the local
filesystem or a static HTTP server.  
**Main flow:**

1. User opens the self-contained HTML viewer in any modern browser (Chrome/Firefox).
2. The viewer loads the output file — no server, no build step, no native install (FR-OUT-03).
3. The viewer renders the worm body (3D spine + per-muscle activation colours) animated over
   time.
4. The user scrubs the timeline, rotates the view, and inspects muscle-activation patterns.

**Post-condition:** User has explored the simulation interactively.

---

### UC-04 — Network-level perturbation study

**Actor:** Researcher / Biophysicist  
**Pre-condition:** Baseline (wild-type) simulation output from UC-01 exists and passes
UC-02 validation.  
**Main flow:**

1. Researcher edits the configuration file to specify one or more network-level
   perturbations (FR-NET-03), for example:
   - zero KD conductance → potassium channel knock-out;
   - synaptic weight scaling on a specific motoneuron class → connectivity perturbation;
   - gap-junction conductance reduction → electrical-coupling change.
2. Researcher runs the perturbed simulation via UC-01.
3. Researcher runs validation (UC-02) on the perturbed output.
4. The 256-feature report shows the phenotypic shift relative to baseline.
5. Researcher compares the shift to the predicted phenotype from the biophysical hypothesis.

**Post-condition:** Perturbation phenotype documented; mechanistic interpretability
demonstrated (FR-VALID-04).

---

### UC-05 — Deploy on an HPC cluster (batch parameter sweep)

**Actor:** HPC scheduler + Researcher  
**Pre-condition:** Container image built and transferred to the cluster; SLURM job script
configured for the parameter sweep; network model files (NEURON or NeuroML2) and
perturbation configurations prepared.  
**Main flow:**

1. Researcher submits the SLURM job array (FR-DEPLOY-02); each job runs one parameter
   configuration in the container (FR-DEPLOY-01).
2. The scheduler allocates nodes / GPUs; jobs run independently in parallel.
3. Output files are written to a shared filesystem.
4. Researcher retrieves outputs and runs the validation pipeline (UC-02) over the result set.

**Post-condition:** Parameter sweep completed; outputs available for statistical analysis
and visualisation.

---

## 7. Traceability Matrix

### Objectives → Requirements

| Objective | FRs | NFRs |
|---|---|---|
| **G1** Real-time GPU-HH | FR-NET-01, FR-NET-02, FR-SIM-01, FR-SIM-02, FR-COMPUTE-02 | NFR-PERF-01, NFR-PERF-02 |
| **G2** Corotated FEM | FR-SIM-03, FR-SIM-04 | NFR-PERF-02 |
| **G3** Proprioceptive loop | FR-SIM-05, FR-VALID-03 | — |
| **G4** 256-feature validation | FR-SIM-06, FR-OUT-02, FR-VALID-01, FR-VALID-02 | — |
| **G5** Mechanistic interpretability | FR-NET-03, FR-NET-04, FR-VALID-04 | — |
| **G6** Open, reproducible, HPC | FR-NET-04, FR-COMPUTE-01, FR-COMPUTE-04, FR-DEPLOY-01, FR-DEPLOY-02, FR-DEPLOY-03 | NFR-PORT-01, NFR-PORT-02, NFR-REPR-01, NFR-REPR-02, NFR-MAINT-01, NFR-MAINT-02, NFR-SCALE-01 |
| **G7** Web 3D viewer | FR-OUT-01, FR-OUT-03, FR-OUT-04 | NFR-WEB-01 |
| **E2** Backend-agnostic | FR-COMPUTE-01 – FR-COMPUTE-05 | NFR-PORT-01, NFR-PORT-02, NFR-MAINT-02 |
| **E4** V&V / CI | FR-COMPUTE-04, FR-DEPLOY-03 | NFR-REPR-01, NFR-REPR-02, NFR-MAINT-01 |

### Use Cases → Requirements

| Use Case | Key FRs exercised |
|---|---|
| UC-01 (Run simulation) | FR-NET-01, FR-NET-02, FR-NET-03, FR-SIM-01 – 06, FR-OUT-01, FR-OUT-02 |
| UC-02 (Validate) | FR-VALID-01, FR-VALID-02, FR-OUT-02 |
| UC-03 (Browser viewer) | FR-OUT-01, FR-OUT-03, FR-OUT-04 |
| UC-04 (Network perturbation) | FR-NET-03, FR-VALID-04 |
| UC-05 (HPC batch) | FR-DEPLOY-01, FR-DEPLOY-02, FR-COMPUTE-02 |

---

## 8. Open Requirements Issues

| ISSUE | Impact |
|---|---|
| ISSUE-002 | Body mesh choice (tetrahedral vs eigenworm) changes the mesh-input interface in FR-SIM-03 |
| ISSUE-003 | Neural granularity (single vs multi-compartment) affects FR-SIM-01 compartment count and flop budget |
| ISSUE-004 | Proprioceptive coupling topology (which stretch-receptor neurons/channels) specifies FR-SIM-05 in detail |
| ISSUE-005 | Output schema + viewer stack resolves the implementation detail behind FR-OUT-01 and FR-OUT-03 |
| ISSUE-006 | Primary acceptance feature subset refines FR-VALID-01 acceptance criterion |
| ISSUE-009 | NeuroML2 / NMODL parser strategy (FR-NET-01, FR-NET-02): implement in C++ vs use Python jNeuroML/libNeuroML as a pre-processing step |
| ISSUE-010 | Backend interface (hand-rolled vs Kokkos/SYCL) specifies the implementation behind NFR-MAINT-02 |
| ISSUE-011 | Scope of FR-NET-04 (general NeuroML2 parser): which NeuroML2 features are required beyond c302? Affects parser complexity and test surface |

See [docs/ISSUES.md](../ISSUES.md) for full details and status.

---

*This document drives the architecture design in `docs/design/02-architecture-design.md`.*

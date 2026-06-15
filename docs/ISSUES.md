# wormsim2 — Issues & Improvements Tracker

A lightweight, version-controlled tracker for bugs, improvements, research questions,
and design decisions. Each item has a stable ID (`ISSUE-NNN`) so it can be referenced
from commits, code comments, and the requirements/design docs.

**Type:** `bug` · `improvement` · `research` · `decision` · `task`
**Status:** `open` · `in-progress` · `resolved` · `wontfix`
**Priority:** `P0` (blocker / core contribution) · `P1` (high) · `P2` (medium) · `P3` (low)

> IDs are never reused. When an item is resolved, move it to the **Resolved** section
> with a resolution note and date; do not delete it.

---

## Open

| ID | Type | Pri | Title | Notes / context |
|---|---|---|---|---|
| ISSUE-002 | research | P1 | Body discretisation choice | Tetrahedral mesh (MetaWorm-style), prior hex/Cosserat representation, or a new reduced-order eigenworm body? Drives the FEM module design. (synthesis §7.1) |
| ISSUE-003 | research | P1 | Neural model granularity | Single-compartment vs multi-compartment Hodgkin–Huxley per neuron — accuracy vs GPU cost trade-off. (synthesis §7.2) |
| ISSUE-004 | research | P0 | Proprioceptive coupling design | Which stretch-receptor neurons/channels, and what feedback gain/topology close the sensorimotor loop. This is the project's core contribution (G3). (synthesis §7.4) |
| ISSUE-005 | decision | P1 | Web output schema + viewer stack | Serialization format (JSON vs binary threshold for large runs) and browser rendering stack (Canvas2D vs WebGL/Three.js) for the 3D viewer (G7). (synthesis §7.6) |
| ISSUE-006 | task | P1 | Primary acceptance feature subset | Select which of the 256 open-worm-analysis-toolbox features are primary acceptance criteria (G4). (synthesis §7.5) |
| ISSUE-007 | task | P1 | Acquire/curate real-worm WCON dataset | Pull wild-type N2 spine tracks from the Open Worm Movement Database (Zenodo) and define the validation reference set (DA4). |
| ISSUE-008 | research | P2 | Long-duration ground-truth strategy | No long pre-computed Sibernetic run exists publicly; decide how to obtain multi-second cross-reference data (own GPU run vs MetaWorm Zenodo data vs real-worm only). |
| ISSUE-009 | decision | P2 | Dual-format parser implementation strategy (NEURON + NeuroML2) | The system accepts two input formats (FR-NET-01): **NEURON** (`.hoc` + `.mod`) via NEURONLoader and **NeuroML2** (`.nml`) via NeuroMLLoader; both produce the same `NetworkConfig`. Implementation options for each loader: (a) native C++ parser in `src/cpp/io/`; (b) Python jNeuroML/libNeuroML / python-neuroml as a pre-processing step that serialises `NetworkConfig` to a binary/JSON intermediate loaded by C++; (c) hybrid. Trade-offs: (a) no Python dep at runtime, simpler deployment; (b) full spec coverage, lower C++ implementation cost; (c) flexibility. Also decide what to port from prior wormuse work (JAX HH model, Cosserat rod) vs reimplement in C++. |
| ISSUE-010 | decision | P2 | Backend interface: hand-rolled vs portability layer | Implement the ISSUE-001 backend abstraction as a thin hand-rolled interface, or adopt a portability layer (Kokkos / SYCL). Trade-off: control/simplicity vs less glue code and broader hardware reach. |
| ISSUE-011 | research | P2 | Conformance scope of NEURON and NeuroML2 loaders beyond c302 (FR-NET-04) | Determine which NEURON HOC/NMODL and NeuroML2 features must be supported to make the parsers general (FR-NET-04). c302 uses a subset of each format. Decide whether to formally claim conformance to a standard or explicitly scope to a "c302-compatible subset" to bound test surface. |

---

## Resolved

| ID | Type | Title | Resolution | Date |
|---|---|---|---|---|
| ISSUE-001 | decision | GPU compute backend | **Backend-agnostic core.** Kernels sit behind a thin internal interface with selectable backends: **CPU/OpenMP** (portable reference — full dev, correctness, and CI on the developer machine), **CUDA** (NVIDIA real-time/deploy target — RTX / HPC clusters), and **OpenCL** (optional — lets an integrated/non-NVIDIA dev GPU execute the kernels). Rationale: the dev machine has no NVIDIA GPU and cannot run CUDA at all, so a portable CPU reference path is required; CUDA remains the performance/deploy target. Supersedes the earlier CUDA-only decision. | 2026-06-15 |

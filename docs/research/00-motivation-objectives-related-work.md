# wormsim2 — Motivation, Objectives, and Related Work

> **Document type:** Research synthesis / project charter (publication-oriented).
> **Status:** Draft for review (v0.1.0-dev).
> **Audience:** Computational neuroscience / scientific-computing readers.
> **Scope:** This document captures the scientific rationale, the related-work
> landscape, the gap being addressed, the proposed approach, the validation
> strategy, and the explicit project objectives and assumptions for **wormsim2**.

---

## Abstract

*Caenorhabditis elegans* is the only organism with a fully mapped connectome
(302 neurons, ~7,000 synapses) and a well-characterised body plan (95 body-wall
muscles), making it the canonical target for whole-organism *in silico* modelling.
Two reference simulators dominate the locomotion-modelling landscape: **Sibernetic**,
a smoothed-particle-hydrodynamics (SPH) body coupled to the **c302** Hodgkin–Huxley
nervous system, which is biophysically detailed but ~5×10⁴ slower than real time; and
**MetaWorm/BAAIWorm**, a corotated finite-element (FEM) body coupled to a *trained
neural-network* brain, which runs in real time but replaces the ion-channel
biophysics with a black-box approximation. **wormsim2** aims to occupy the unfilled
quadrant: a **real-time, GPU-accelerated, biophysically-faithful** simulator that
keeps Hodgkin–Huxley ion-channel dynamics *and* a corotated-FEM body, closes the
**proprioceptive sensorimotor loop** (absent from both reference simulators), and is
**validated quantitatively against real-worm behavioural data** using the
community-standard 256-feature phenotyping pipeline rather than a handful of summary
statistics. Results are exposed through a **web-friendly interactive 3D
visualization**, so simulated locomotion can be explored in a browser without a native
install.

---

## 1. Motivation

Existing whole-worm simulators force a trade-off between **biophysical fidelity** and
**runtime**:

- **Fidelity-first (Sibernetic + c302):** authentic ion-channel kinetics and a
  full SPH soft body, but ~857 s of wall-clock for 15 ms of simulated time on CPU
  (≈ 5.7×10⁴ × slower than real time). GPU-accelerating the SPH body would cut a 5 s
  run from ~79 h (CPU) to an estimated ~4–8 h, but the **c302 nervous system has no
  practical GPU path** — the standard OpenWorm toolchain runs c302 on NEURON via
  jNeuroML (CPU-only), and NEURON's only GPU route (CoreNEURON, OpenACC/CUDA) targets
  large-scale networks and offers little benefit for a 302-neuron model — so c302
  (~27 h CPU for 5 s) then *becomes the pipeline bottleneck*. Long-duration behavioural
  studies are therefore infeasible, and no long pre-computed trajectory dataset has
  ever been released publicly.
- **Speed-first (MetaWorm/BAAIWorm):** real-time (30 FPS) corotated-FEM body on a
  commodity GPU, but the nervous system is a trained network optimised to reproduce
  calcium-imaging correlations — it cannot model ion-channel pharmacology, channel
  mutations, or exact spike timing, and its mapping from neural state to behaviour is
  not mechanistically interpretable.

Neither simulator closes the **proprioceptive loop** — the body's own bending is
believed to feed back through stretch-sensitive channels into the motor circuit and
sustain the undulatory gait. Both also validate weakly: Sibernetic against ~6 summary
statistics from three papers, MetaWorm against a single qualitative trajectory shape.

wormsim2 targets the **missing quadrant**: real-time **and** ion-channel-faithful,
with the proprioceptive loop closed and validation against a community-standard
behavioural feature battery.

---

## 2. Background and Related Work

### 2.1 Sibernetic (PCISPH body + c302 nervous system)

Sibernetic [Palyanov et al. 2018] implements a Predictive-Corrective Incompressible
SPH solver with contractile-matter and impermeable-membrane extensions. The worm body
is ~5×10³ elastic particles; the environment adds liquid/boundary particles
(up to 5×10⁵ at full resolution). The timestep is 20 µs (CFL-limited). Muscle drive
comes from **c302**, a NEURON/jNeuroML Hodgkin–Huxley model of the 302-neuron
connectome. Validation (its Table 1) compares crawling/swimming **velocity,
wavelength, and frequency** to Fang-Yen et al. 2010, Shen et al. 2012, and
Lüersen et al. 2014; all fall within experimental ranges, and the
frequency–wavelength slope (0.59) matches the literature (0.64).

**Strengths:** authentic ion channels, true fluid–structure interaction, agar/liquid
environments. **Limitations:** ~5.7×10⁴× slower than real time on CPU; GPU OpenCL path
under-utilised at ~5×10³ particles; no proprioceptive feedback; no long public dataset.

**Runtime estimates for a 5 s target (back-of-envelope).** The published Sibernetic
benchmark (857 s wall-clock for 15 ms simulated time) extrapolates to **~79 h CPU** for
5 s of body simulation alone; a GPU port is realistically ~10–20× at this particle
count (~5×10³), i.e. **~4–8 h**. The c302 nervous system, however, runs on CPU in the
standard OpenWorm toolchain (NEURON via jNeuroML); NEURON's only GPU route is
**CoreNEURON** (OpenACC/CUDA, separate build), which is engineered for *large-scale*
networks and brings little benefit to a 302-neuron model (kernel-launch overhead
dominates; TABLE-statement and Random123 restrictions apply). c302 therefore stays
CPU-bound (**~27 h** for 5 s) and *dominates the GPU-accelerated pipeline*. This is
exactly why wormsim2 puts a purpose-built, GPU-batched HH solver at layer 1 (§4.1):
it sidesteps the c302/NEURON bottleneck rather than inheriting it.

**NEURON and NeuroML2 files as the network-model source.** Although the NEURON
*runtime* is the bottleneck above, c302 exposes the full nervous system in two
machine-readable formats that wormsim2 accepts as first-class inputs:

- **NEURON format** — a `.hoc` network description (neuron objects, synapse
  connectivity, gap junctions) combined with **NMODL `.mod` files** encoding HH-style
  gate-variable ODEs (α/β rate functions, reversal potentials, per-neuron conductance
  densities) for channels such as NCA (non-specific cation), KA (fast K⁺),
  KD (delayed-rectifier K⁺), KQS and KVS (slow K⁺), and IR (inward rectifier).
  This is the native c302 representation used in Sibernetic.
- **NeuroML2 format** — `.nml` XML files describing the complete network: neuron list,
  compartment geometry, chemical synapse parameters (max conductance, reversal
  potential, rise/decay time constants), gap-junction conductances, and ion-channel
  model references (resolved from `.mod` files or inline ChannelML definitions).
  c302 can export to NeuroML2 via jNeuroML.

Both formats are peer-reviewed, community-validated biophysical specifications for the
302-neuron connectome.

wormsim2 adopts a **dual-format input model**: the user may supply either a
**NeuroML2** (`.nml`) network description or a **NEURON** (`.hoc` + `.mod`) description;
both are parsed to the same internal `NetworkConfig` struct that drives the GPU-HH
solver (ISSUE-009). The NEURON format (`.hoc` network description + NMODL `.mod` channel
kinetics) is the native c302 representation used in Sibernetic; NeuroML2 is the
community XML exchange format that c302 can also export. Accepting both formats lets
users work in whichever representation they already have, *bypassing the NEURON runtime
entirely* while preserving its biophysical ground truth. Any network expressible in
either format can drive the GPU-HH solver — not only c302.

### 2.2 MetaWorm / BAAIWorm (corotated FEM body + trained-NN brain)

MetaWorm [BAAIWorm, *Nat. Comput. Sci.* 2024] couples a **corotated linear-elastic
FEM** body (984 vertices, 3,341 tetrahedra, 7,899 constraints: 4,558 muscle + 3,341
elasticity) to a multi-compartment, morphologically realistic **neural network whose
dynamics are trained** to reproduce whole-brain calcium imaging (65 neurons,
Uzel et al. 2022; MSE = 0.076 on the correlation matrix). It runs **real-time at
30 FPS on an RTX 3060** and reproduces crawling, swimming, and omega-turns plus
zigzag chemotaxis.

**Strengths:** real-time; corotated formulation handles large rotations (omega-turns);
two-way fluid coupling; anatomically detailed muscle constraints; arbitrary-duration
runs. **Limitations:** the brain is a black box — no ion-channel pharmacology, no
mutation modelling, no exact spike timing; body-movement validation is essentially
qualitative (trajectory-shape similarity, no per-feature behavioural metrics).

### 2.3 Prior work by the author (wormuse)

The author's earlier project (*wormuse*) established several components reused here:
a JAX Hodgkin–Huxley / ion-channel model (PINN-calibrated, F1 = 0.933); a deal.II
Q1-hexahedral FEM body using a **per-quadrant axial eigenstrain** muscle model
(M_abs = 0.933 vs Sibernetic curvature, 1760× faster than the full OpenWorm pipeline);
and a 1D **Cosserat/Kirchhoff rod** prototype (Hermite-cubic FEM, Crank–Nicolson,
unconditionally stable, 4761× faster than Sibernetic, F3D = 0.805 / 3.21 µm mean
per-muscle error). A key negative result (wormuse H4): the frozen 15 ms c302
activation has a very small dorsal–ventral differential (Pearson r ≈ +0.015), i.e.
**open-loop c302 alone barely bends the body** — strong evidence that the
proprioceptive loop is what sustains realistic locomotion. wormsim2 generalises these
components into a new, real-time, closed-loop, GPU-first architecture.

### 2.4 Other locomotion models

**ElegansBot** [*eLife* 2024] models the body as a chain of rigid rods with damped
torsional springs and Newtonian equations of motion, reproducing crawling, swimming,
and omega/delta turns analytically — fast and interpretable, but without ion-channel
biophysics or a 3D continuum body. Eigenworm analyses [Stephens et al. 2008] show
~4 postural modes capture ~95% of body-shape variance, motivating reduced-order body
representations.

### 2.5 Validation datasets

- **Real-worm behaviour:** the **Open Worm Movement Database** (Zenodo) provides
  minutes-long *C. elegans* spine tracks in **WCON** format (Schafer-lab Worm
  Tracker 2.0 lineage), the substrate for the community **256-feature** behavioural
  phenotyping implemented by the **open-worm-analysis-toolbox**.
- **Neural reference:** whole-brain calcium imaging (Uzel et al. 2022) and
  per-neuron patch-clamp recordings (e.g. AWC, AIY, AVA, RIM, VD5).
- **Simulation cross-reference:** Sibernetic remains a useful per-muscle 3D
  cross-check for the body solver even though it is not real-worm ground truth.

---

## 3. Gap Analysis

| Capability                          | Sibernetic             | MetaWorm           | **wormsim2 (target)**     |
| ----------------------------------- | ---------------------- | ------------------ | ------------------------------- |
| Ion-channel (HH) nervous system     | ✅ (c302, CPU)         | ❌ (trained NN)    | ✅**GPU-batched HH**      |
| Real-time runtime                   | ❌ (~5.7×10⁴× slow) | ✅ (30 FPS)        | ✅**real-time target**    |
| 3D continuum body                   | ✅ SPH                 | ✅ corotated FEM   | ✅**corotated FEM**       |
| Large rotations (omega-turn)        | ✅                     | ✅                 | ✅                              |
| Proprioceptive closed loop          | ❌                     | ❌                 | ✅**core contribution**   |
| Channel pharmacology / mutation     | ✅                     | ❌                 | ✅                              |
| Quantitative behavioural validation | partial (6 stats)      | qualitative        | ✅**256-feature battery** |
| Long-duration capability            | ❌                     | ✅                 | ✅                              |
| Web-friendly interactive 3D viz     | ❌ (offline mp4/VTK)   | ❌ (native viewer) | ✅**browser 3D viewer**   |
| Open, reproducible, HPC-deployable  | partial                | partial            | ✅**first-class goal**    |

**The unfilled quadrant** is *real-time + ion-channel-faithful + closed-loop +
rigorously validated*. That is wormsim2.

---

## 4. Proposed Approach

A four-layer, GPU-first, closed-loop architecture with a web-friendly output layer:

```
┌──────────────────────────────────────────────────────────────────────┐
│ (1) Neural layer — GPU-batched Hodgkin–Huxley                          │
│     302 neurons × multi-compartment · connectome synapses + gap jns    │
│     dt ≈ 0.025 ms · O(10^4) ODEs · << 1 ms/frame on a commodity GPU    │
├──────────────────────────────────────────────────────────────────────┤
│ (2) Neuromuscular mapping — motoneuron V → 95 muscle activations       │
├──────────────────────────────────────────────────────────────────────┤
│ (3) Body mechanics — corotated linear-elastic FEM (large rotations)    │
│     per-quadrant muscle drive · real-time implicit solve               │
├──────────────────────────────────────────────────────────────────────┤
│ (4) Environment — agar (crawl) ↔ liquid (swim), contact + drag         │
└──────────────────────────────────────────────────────────────────────┘
        ▲                                                      │
        └────── (5) Proprioceptive feedback: body curvature ───┘
                    → stretch-receptor channels → motor circuit
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ (6) Output layer — compact web-friendly serialization                  │
│     spine + per-muscle activation time series (JSON / binary)          │
│     → WCON export (validation) · → browser 3D viewer (exploration)     │
└──────────────────────────────────────────────────────────────────────┘
```

**4.1 Neural layer (GPU-HH).** A first-principles Hodgkin–Huxley network over the
connectome. A back-of-envelope budget (≈3×10³ compartments × 4 state variables ×
~1.3×10³ steps/frame at 30 FPS ≈ 3×10⁸ flops/frame) is ~0.03 ms on a ~10-TFLOP GPU —
the neural layer is **not** the bottleneck. Network topology, synapse parameters, and channel kinetics are configured via a
**NeuroML2 / NMODL parser** (§2.1, ISSUE-009), preserving biophysical continuity
with the c302 ground truth while bypassing the NEURON runtime entirely.
This restores everything MetaWorm's NN gives up: pharmacology, mutations, exact spike
timing, and mechanistic interpretability.

**4.2 Neuromuscular mapping.** Motoneuron membrane potential → per-muscle activation
(linear/sigmoid), preserving the established 95-muscle anatomical map.

**4.3 Body mechanics (corotated FEM).** Adopt MetaWorm's corotated linear-elastic
formulation so large body rotations (omega-turns, swimming amplitudes) are handled
correctly — the regime where the author's prior small-strain eigenstrain model breaks
down. Muscle drive enters per dorsal/ventral/left/right quadrant.

**4.4 Proprioceptive closed loop (core contribution).** Body curvature is sensed by
stretch-sensitive channels and fed back into the motor circuit, closing the
sensorimotor loop that neither Sibernetic nor MetaWorm models. This is the mechanism
hypothesised to **sustain** undulation given that open-loop c302 drive alone is nearly
flat (§2.3).

**4.5 Environment.** Agar (high-drag crawl) and liquid (low-drag swim) with a contact
model, enabling the gait transition documented by Fang-Yen et al. 2010.

**4.6 Output and web visualization.** The engine serializes its state — spine
positions plus per-muscle activations over time — into a compact, web-friendly format
(JSON, with an optional binary path for large runs). This single output feeds two
consumers: (i) a **WCON exporter** for the validation pipeline (§5), and (ii) a
**self-contained browser-based 3D viewer** that renders the worm crawling/dancing on
the agar plate, building directly on the author's prior browser visualization
(*wormuse* worm-dance). The viewer requires no native install and no server, making
results shareable and interactively explorable on the web. Decoupling the engine from
the viewer via a documented output schema keeps the real-time C++/GPU core independent
of any visualization stack.

---

## 5. Validation Strategy

Validation proceeds at three levels, strongest-first:

1. **Behavioural (primary).** Convert simulated spine tracks to **WCON**, run the
   **open-worm-analysis-toolbox** to compute the **256-feature** battery, and compare
   the simulated feature distributions to **real-worm wild-type N2** distributions from
   the Open Worm Movement Database (per-feature KS test / effect size; target:
   a large majority of features within the wild-type distribution).
2. **Biomechanical (secondary).** Reproduce Fang-Yen et al. 2010 crawl↔swim
   **velocity / wavelength / frequency** trends; cross-check per-muscle 3D body shape
   against Sibernetic on an identical stimulus (per-muscle 3D position error after
   Procrustes alignment).
3. **Neural (secondary).** Compare network activity statistics to whole-brain calcium
   imaging (correlation structure) and single-neuron patch-clamp traces; demonstrate a
   channel knock-out producing a predicted behavioural phenotype.

This is a substantially more rigorous validation than either reference simulator.

---

## 6. Project Objectives and Assumptions  ⟵ *please verify this section*

### 6.1 Primary objectives (G — goals)

- **G1 — Real-time, biophysically-faithful simulator.** Build a *C. elegans*
  locomotion simulator that runs at (near) real time on a single commodity GPU **while
  retaining Hodgkin–Huxley ion-channel dynamics** (no trained-NN substitute for the
  nervous system).
- **G2 — Corotated-FEM body.** Implement a corotated linear-elastic FEM body that
  handles large rotations (crawling, swimming, omega-turns) driven by per-quadrant
  muscle activation.
- **G3 — Closed proprioceptive loop.** Model stretch-receptor feedback from body
  curvature into the motor circuit and demonstrate it sustains undulatory locomotion
  where open-loop drive does not.
- **G4 — Quantitative behavioural validation.** Validate against real-worm data using
  the 256-feature open-worm-analysis-toolbox battery (WCON pipeline), not just summary
  statistics.
- **G5 — Mechanistic interpretability.** Demonstrate at least one **network-level
  perturbation** — ion-channel knock-out / conductance scaling, synaptic weight
  modification, or gap-junction change — producing a predicted behavioural phenotype;
  a capability the trained-NN approach cannot offer by construction.
- **G6 — Open, reproducible, HPC-deployable software.** Deliver a professionally
  engineered, open-source, well-tested, containerised, cluster-deployable codebase
  with full requirements/design/V&V documentation.
- **G7 — Web-friendly interactive visualization.** Emit a documented, web-friendly
  output format and a self-contained browser-based 3D viewer (building on the author's
  *wormuse* worm-dance visualization) so simulated locomotion is explorable and
  shareable on the web with no native install — neither reference simulator offers
  this.

### 6.2 Engineering objectives (how G6 is met)

- **E1 — Modern C++** core (numerics, FEM, neural ODE integration) with a clean,
  testable module architecture.
- **E2 — Backend-agnostic GPU acceleration** for the HH network and FEM solve: a
  CPU/OpenMP reference backend (portable — used for development and CI), a CUDA backend
  (NVIDIA real-time/deploy target), and an optional OpenCL backend, all behind one
  internal kernel interface (ISSUE-001).
- **E3 — Parallelism** (shared- and/or distributed-memory) where it buys scalability.
- **E4 — Reproducibility & V&V:** deterministic test suite, CI/CD, containerised
  builds, semantic versioning, traceable requirements → design → tests.
- **E5 — Analysis layer** in Python (validation notebooks, WCON/feature pipeline,
  figures) consuming the C++ engine's outputs.
- **E6 — Web visualization layer:** a documented output schema plus a self-contained
  browser 3D viewer (no server, no build step) decoupled from the engine core (G7).

### 6.3 Domain assumptions (DA — hold independent of the software)

- **DA1** — The *C. elegans* connectome (302 neurons, synapses, gap junctions) and the
  95 body-wall-muscle map are fixed, known inputs.
- **DA2** — Forward crawling on agar is quasi-periodic (~0.3–0.5 Hz), so periodic
  drive plus proprioceptive feedback is a reasonable basis for sustained locomotion.
- **DA3** — Body-shape variance is low-rank (~4 eigenworm modes ≈ 95%), so a
  reduced-order body model is physically justified.
- **DA4** — Real-worm WCON behavioural data and the 256-feature toolbox are the
  community-accepted ground truth for locomotion phenotyping.
- **DA5** — A commodity GPU (e.g. RTX 3060 class) is the reference target for the
  real-time claim; HPC/cluster deployment is for batch parameter studies, not for the
  real-time interactive path.

### 6.4 Non-goals (explicit scope limits)

- **N1** — Not modelling development, reproduction, feeding, or non-locomotor behaviour.
- **N2** — Not aiming to surpass Sibernetic's full-resolution fluid detail; fluid is
  simplified for real-time operation.
- **N3** — Not re-deriving c302 from scratch; existing connectome/channel parameters
  are reused and refined.

---

## 7. Open questions for the design phase

1. Body discretisation: reuse a MetaWorm-style tetrahedral mesh, the author's prior
   hex/Cosserat representation, or a new reduced-order (eigenworm) body?
2. Neural model granularity: single-compartment vs multi-compartment HH per neuron?
3. GPU strategy — *resolved (ISSUE-001):* backend-agnostic core (CPU/OpenMP reference +
   CUDA deploy + optional OpenCL). Remaining sub-question: hand-rolled interface vs a
   portability layer (Kokkos/SYCL) (ISSUE-010).
4. Proprioceptive coupling: which stretch-receptor neurons/channels, and what feedback
   gain/topology?
5. Validation target subset: which of the 256 features are primary acceptance criteria?
6. Web output: which serialization schema (JSON vs binary threshold) and which browser
   rendering stack (Canvas2D vs WebGL/Three.js) for the 3D viewer?

---

## 8. References (to be completed in publication form)

- Hodgkin & Huxley (1952). A quantitative description of membrane current. *J. Physiol.*
- Palyanov, Khayrulin, Vella et al. (2018). Three-dimensional simulation of the
  *C. elegans* body and muscle cells in liquid and gel environments. *Phil. Trans. R. Soc. B.*
- BAAIWorm / MetaWorm (2024). An integrative data-driven model simulating *C. elegans*
  brain, body and environment interactions. *Nature Computational Science.*
- Fang-Yen et al. (2010). Biomechanical analysis of gait adaptation in *C. elegans*. *PNAS.*
- Shen et al. (2012). Undulatory locomotion of *C. elegans* on wet surfaces. *Biophys. J.*
- Lüersen et al. (2014). Gait-specific adaptation of locomotor activity. *J. Exp. Biol.*
- Stephens et al. (2008). Dimensionality and dynamics in the behavior of *C. elegans*. *PLoS Comput. Biol.*
- ElegansBot (2024). Equation of motion deciphering locomotion including omega turns. *eLife.*
- Uzel et al. (2022). A set of hub neurons and non-local connectivity in the *C. elegans* connectome. *Curr. Biol.*
- Open Worm Movement Database (Zenodo) + WCON / tracker-commons + open-worm-analysis-toolbox.

---

*This document is the scientific foundation for the requirements-analysis and
architecture-design documents that follow.*

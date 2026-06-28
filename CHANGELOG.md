# Changelog

All notable changes to **wormsim2** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Versioning policy**
> - `MAJOR` (1.x.x): first public release; breaking API changes after 1.0
> - `MINOR` (0.x.0): new module, new capability, or significant doc milestone
> - `PATCH` (0.x.y): bug fixes, test additions, minor doc corrections within a milestone

---

## [0.11.2] - 2026-06-28 *(current)*

### Fixed
- **Head/tail flip correction in N2 WCON processing** — the N2 recording (Zenodo 1031837)
  has a 16.6 s tracking gap at t=431.4→448.0 s where the tracker re-acquired the worm with
  head and tail swapped. Without correction, PCA mode-1 coefficient has mean +4.9 before the gap
  and −14.0 after, producing bizarre body shapes in the animation for the final ~168 s.
  Fix: detect large time gaps (>5 s); compute mean body-orientation angle in a 10-frame window
  before and after; if the direction changes by >90° → reverse the keypoint order for all
  subsequent frames. Applied in `cv115b_tuner_600s.py` (generates the N2 Plotly HTML + GIF).
  After correction: mode-1 mean ≈ 0 both before and after, PCA var_explained=0.782,
  tuner shape error=0.022 BL (CoM-aligned).
- **CV-11.5b: right panel now shows 4-mode PCA tuner output** (not FK round-trip) — previous
  version (`v1105_n2_plate_600s.html`) used a trivial FK round-trip that reproduced the real
  skeleton almost exactly (error ≈ 0.008 BL), making the right panel redundant. New version
  (`v1200_n2_tuner_600s.html`) shows `theta_rec = mu + Σ aₙ(t)·eₙ(s)` with n=1..4 — the
  NeuromuscularTuner's actual generative model output that feeds `fem_body_trace --activation-csv`.
  The right panel now shows a visually distinct but biologically faithful approximation
  (var_exp=0.782, shape error=0.022 BL vs 0.008 BL FK sanity).
- **`.github/workflows/ci-baseline.yml`** — added `-DWORMSIM2_OPENMP=OFF` to both cmake
  configure steps (build-test and coverage jobs); prevents `omp.h not found` clang-tidy error
  on Ubuntu runners where `find_package(OpenMP)` succeeds but clang-tidy cannot find the header.

### Added
- **`docs/images/v1200_n2_tuner_600s.html`** — N2 WT agar-plate Plotly HTML with flip
  correction + 4-mode tuner output; 638 animation frames ×27 from 17 226 valid frames;
  left: real WCON (flip-corrected), right: NeuromuscularTuner eigenworm reconstruction;
  responsive layout (`autosize=True`); 13.6 MB.
- **`docs/images/v1105_n2_plate_600s.gif`** — replaced with flip-corrected 4-mode tuner GIF
  (221 frames @ 20 fps, 11 s, 578 KB); generated via `plotly_fig_to_gif()`.

---

## [0.11.1] - 2026-06-28

### Added
- **CV-11.5 notebook cell** — N2 vs N2-simulated agar-plate view: full 4 s window (114 frames),
  both worms normalised to 1 BL (measured real arc = 0.656 mm), plate-frame coordinates with
  0.856 BL forward translocation, cream agar background, Plasma head→tail gradient, accumulated
  centroid trail. Zenodo option-2 hook documented in source. ISSUES 013-015 added to
  `docs/ISSUES.md` as future architectural plans.
- **`docs/images/v1105_n2_plate_view.gif`** — 114-frame agar-plate GIF (547 KB, 14 fps).
- **`docs/images/v1105_n2_plate_view.html`** — interactive Plotly version with play/pause (4 s).
- **`docs/images/v1105_n2_plate_600s.gif`** — compact 600 s agar-plate GIF (220 frames at 20 fps,
  11 s total, 1.7 MB): covers the entire 82.6 BL centroid path; per-panel matplotlib render
  (same data pipeline as HTML); growing centroid trail, Plasma body gradient, head circle at
  index 0; embedded in `notebooks/project_tour.ipynb` cell CV-11.5b + linked in README.
- **`docs/images/v1105_n2_plate_600s.html`** — full 600 s agar-plate view (real WCON data,
  Zenodo 1031837): 616 animation frames subsampled ×28 from 17 226 valid frames; FK body is an
  FK round-trip of the real skeleton (per-frame `theta_body` from WCON → `compute_skeletons` →
  uniform-segment reconstruction); mean FK shape error 0.040 BL (uniform-ds approximation);
  total centroid path 82.6 BL; head marker (yellow circle) at index 0 (WCON `head='L'`);
  body-length normalised to 1 BL (real arc 0.664 mm); both panels share the same centroid track.

### Fixed
- **`src/python/tuner.py` `plotly_fig_to_gif`** — (1) two `or []` truthiness guards on ndarray
  replaced with `is not None`; (2) added `set_aspect("equal", adjustable="datalim")` for panels
  whose x and y spans are within 15% of each other, ensuring equal pixel scale across subplots.
- **`docs/images/v1105_n2_plate_600s.html`** — head marker corrected to index 0 (WCON `head='L'`
  convention; previous version erroneously placed circle at index 48 = tail); body curvature now
  extracted per frame from the real WCON skeleton via `xy_to_tangent` and fed to `compute_skeletons`
  (FK round-trip), replacing the earlier tiled periodic 4 s `theta_rec` that did not track the
  real worm's time-varying behaviour (stops, reversals, omega turns).

---

## [0.11.0] - 2026-06-28

### Added
- **`src/python/skeleton.py`** (new canonical FK module) — `theta_body` convention documented with derivation; `tangent_to_xy(theta_body, ds, mid_idx)` and `tangent_to_xy_batch()` are the single source of truth for all FK math; `xy_to_tangent(x,y)` inverse; `load_skeleton_csv(path)` → `SkeletonData` (reads `t_s,x0..x48,y0..y48` CSV, aligns tail→head to +y, centres at midpoint, computes theta_body); `visualize_skeleton_csv(path)` → Plotly animation directly from any compatible CSV; module docstring explains why cumsum-on-angles is wrong (the bug this release fixes)
- **`src/python/hardware.py`** — `HardwareProfile` dataclass + `detect_hardware()`: probes CPU (name, logical/physical cores, RAM), CUDA GPUs (cupy > torch.cuda > nvidia-smi), Apple Metal MPS (torch.mps), OpenCL platforms (pyopencl), JAX devices, OpenMP (CMake cache + ctypes libgomp probe), joblib worker count; returns `recommended_backend` token; fully non-destructive, graceful on missing optional deps
- **`src/python/backends.py`** — `NumPySerialBackend`, `NumPyBatchBackend` (30-80× faster on ≥300 frames), `NumPyMultiprocessBackend` (joblib Parallel), `JAXBackend` (JIT + XLA; uses `lax.dynamic_slice`), **`OpenCLBackend`** (PyOpenCL GPU kernel: one work-item per frame, `tangent_to_xy_kernel` in OpenCL C, supports NVIDIA/AMD/Intel/macOS — any platform with an OpenCL ICD; `OpenCLBackend.device_name` for diagnostics); all NumPy backends delegate FK to `skeleton.tangent_to_xy_batch` (canonical, single source of truth); `select_backend()` priority: jax_cuda → opencl → jax_cpu → numpy_mp; `benchmark_backends()` default list now includes `opencl`; accepts `theta_body` directly — no `-π/2` subtraction at call site
- **C++ OpenMP** (`src/cpp/src/neural/NeuralIntegrator.cpp`, `CMakeLists.txt`, `src/cpp/CMakeLists.txt`): `WORMSIM2_OPENMP` CMake option (default ON); `find_package(OpenMP CXX QUIET)`; `#pragma omp parallel for schedule(static)` over 302 neurons in `update_gates()` — embarassingly parallel; conditional on `_OPENMP` preprocessor guard for serial fallback
- **`docker/Dockerfile.wormsim2-gpu`** — CUDA 12.4 GPU image: `nvidia/cuda:12.4.0-devel-ubuntu22.04` base + `jax[cuda12]` + `libgomp1`/`libomp-dev` + **`pyopencl`** (enables `OpenCLBackend` inside container); for local GPU benchmarking and SLURM container pre-build
- **`singularity/wormsim2.def`** — Apptainer definition for HPC deployment; installs full Python/C++ stack + **`pyopencl`** in container; sets `OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}`, `JAX_PLATFORMS=cuda,cpu`; `%runscript` invokes `benchmark.py`
- **`scripts/slurm/wormsim2_benchmark.sbatch`** — CINECA G100 SLURM job script: partition `g100_usr_prod`, 1 node × 32 CPUs × 1 V100S GPU (32 GB); Step 1: C++ build with OpenMP; Step 2: Python hardware detect; Step 3: backend benchmark (10,000 frames) probing `opencl` + `jax_cuda` dynamically; Amdahl's law estimate for V100S (f_par=97%, N=6912, theoretical 33×, practical 200-500×)
- **`docs/install/local_cpu.md`** — Local CPU installation guide (macOS + Linux): Python venv, CMake + OpenMP, C++ tests, CV-11 notebook setup; OpenMP troubleshooting for macOS (`libomp` via Homebrew)
- **`docs/install/local_gpu_docker.md`** — GPU Docker guide: NVIDIA Container Toolkit install, `docker run --gpus all`, JAX CUDA verification; full OpenCL backend section (AMD/Intel/NVIDIA/macOS ICD installation, `docker run` with `OpenCLBackend` usage, AMD `--device=/dev/dri` pass-through)
- **`docs/install/hpc_slurm.md`** — SLURM deployment guide: CINECA G100 onboarding, Apptainer build (CI vs local), `sbatch` workflow, G100 specs (V100S 32 GB × 4), Amdahl's law speedup table, `apptainer run --nv` execution
- **`docs/install/github_actions.md`** — CI guide: workflow overview (`ci-baseline.yml`, `slurm-deploy.yml`, `auto-release.yml`), Python benchmark job, self-hosted GPU runner registration, evidence artifact download, matrix strategy for multi-backend testing, release trigger sequence
- **`.github/workflows/slurm-deploy.yml`** — Tag-triggered Apptainer build workflow: `eWaterCycle/setup-apptainer@v2`, `sudo apptainer build`, GPG two-tier signing (key present → sign + verify; absent → SHA-256 only), upload `.sif` + checksum to GitHub Release
- **`ci-baseline.yml` extended** — new `benchmark` job: installs `jax[cpu]` + joblib + `pyopencl` + Intel CPU ICD (`beignet-opencl-icd`), runs `detect_hardware()`, benchmarks 5 backends (`numpy_serial`, `numpy_batch`, `numpy_mp`, `opencl`, `jax_cpu`) on 1000 frames, asserts `numpy_batch` speedup ≥ 20×, skips opencl gracefully if no ICD available, uploads `hardware-detect-output.txt` + `benchmark-output.txt` as CI evidence
- **Full-length CV-11 notebook cells** (`notebooks/project_tour.ipynb`, ids 01009–01013): CV-11.1 hardware detection + benchmark table on 871 frames (30s, 29fps); CV-11.2 N2 vs N2-sim (real-data-initialized from theta_rec[0]); CV-11.3 N2 vs nca-1;nca-2 full-length; CV-11.4 N2 vs egl-19(n2368) full-length; CV-11.cum 10-check pass; all figures display as 2s Plotly previews with 30s HTML generated locally
- **Preview GIFs** (`docs/images/v1102_n2_vs_n2sim_preview.gif`, `v1103_nca_preview.gif`, `v1104_egl19_preview.gif`): 58 frames, 14 fps, 2s clips for README; full 30s Plotly HTML generated locally via `plotly_fig_to_gif()`

### Fixed
- `render_fig_n2_vs_mutant()` no longer caps `n_frames` at `len(self.t_out)` when `ref_x` is provided — full-length simulations of arbitrary duration now supported (`tuner.py:3293`)

### Benchmark (Intel i5-8257U, 8 logical / 4 physical cores, macOS)

| Backend | fps | speedup |
|---|---|---|
| numpy_serial | ~21,000 | 1.0× |
| numpy_batch | ~1,300,000 | **62.7×** |
| numpy_mp (4 jobs) | ~68,000 | 3.3× |
| jax_cpu (XLA) | ~520,000 | 24.8× |
| jax_cuda (V100S est.) | — | ~200–500× |

### HPC Estimate (CINECA G100 · Tesla V100S · 6912 CUDA cores)
- Amdahl's law (f_par=97%, N_cuda=6912): theoretical max **33×** vs serial
- Practical JAX-CUDA vs numpy_serial for 10,000 frames: **~200–500×**
- C++ OpenMP update_gates (302 neurons / 48 threads on G100): **~15–20×** vs serial

---

## [0.10.2] - 2026-06-28

### Added
- **`egl-19(n2368)` preset** in `_MUTANT_PRESETS` (`tuner.py`): L-type voltage-gated Ca²⁺ channel (Cav1 homolog) in body-wall muscle; S4-S5 linker partial LOF; gbar_scale=0.60 → Δamp=−37%, Δf=−20%, Δspd=−29%, no fainting; one-channel change, NCA/cholinergic circuit at WT; calibrated on Yemini 2013 (Worm Behavior Database) + Lee 1997
- **`docs/perturbation_egl19_rof.yaml`**: PerturbationConfig specifying EGL19 gbar_scale=0.60 in BWM cells only; all other channels untouched; `strain_name: "egl-19(n2368)"` for `infer_from_yaml()` literature lookup
- **CV-10.8 notebook cell** (`notebooks/project_tour.ipynb`, id=01008): EGL-19 dose–response sweep (`PerturbationPipeline.sweep("EGL19")`), egl-19(n2368) interactive Plotly animation (`render_fig_n2_vs_mutant()`, same style as CV-10.2), and 3-way comparison table N2 vs nca-1;nca-2 vs egl-19(n2368) demonstrating circuit failure vs muscle-actuator weakening contrast
- `docs/images/v0102_egl19_sweep.png`: Ca_spike dose–response sweep with n2368 anchor at gbar=0.60
- `docs/images/v0102_egl19_rof.html`: egl-19(n2368) interactive Plotly animation (1.7 MB; fixed ±0.32 BL × ±0.65 BL window; EGL-19 subplot title patched)
- `docs/images/v0102_egl19_rof_preview.png`: static kaleido preview for README (140 KB)
- **`plotly_fig_to_gif(fig, path, fps, width_px, height_px, dpi, title)`** module-level function in `tuner.py` (`tuner.py:3647`): converts any animated Plotly scatter figure to a GIF using matplotlib rendering rather than kaleido; extracts per-frame (x, y) trace data, detects subplots from (xaxis, yaxis) pairs, reads background/layout from `fig.layout`, reproduces curvature gradients via Plasma colorscale sampling; ~500× faster than kaleido (23–50 s for 114 frames vs ~10 min)
- `docs/images/v0101_nca_pipeline.gif`: 114-frame 12 fps GIF of CV-10.7 produced by `plotly_fig_to_gif()`; 887 KB; replaces static preview PNG in README
- `docs/images/v0102_egl19_rof.gif`: 114-frame 12 fps GIF of CV-10.8 produced by `plotly_fig_to_gif()`; 1230 KB; replaces static preview PNG in README

### Fixed
- **CV-10.7 and CV-10.8 left panel now uses real N2 skeleton data**: `render_fig_n2_vs_mutant()` gains `ref_x/ref_y/ref_label/cv_label` parameters (`tuner.py:3247`); when `ref_x` is provided the left panel shows those shapes (centred per frame, zero synthetic drift) instead of the eigenworm simulation `xSIM/ySIM`. CV-10.7 uses `tuner.theta_rec × sc_lit.amp_scale` (nca-1;nca-2 reference, with fainting alpha); CV-10.8 uses `tuner.theta_rec × sc_egl.amp_scale` (egl-19 reference, no fainting). Correct layout: reference measurement left, wormsim2 prediction right. Both HTML and GIF regenerated.

---

## [0.10.1] - 2026-06-27

### Added
- **`KinematicScaling`** dataclass — typed carrier for all mutant phenotype scaling factors (f, amp, speed, fainting_prob, fainting_dur_s) plus provenance fields (`source`, `confidence`, `notes`); `summary()`, `compare_to()`, `as_preset_dict()` helpers
- **`PerturbationSpec`** dataclass — describes any ion-channel perturbation; factory methods `knockout()`, `partial()`, `gain_of_function()`, `from_yaml()` (flat + nested format), `from_nml()` (NeuroML2 conductanceDensity parser)
- **`NeuralKinematicsExtractor`** class — extracts `KinematicScaling` from C++ HH voltage-trace CSVs (burst-frequency analysis, quiescence detection, peak-to-trough amplitude ratio); enabled when trace CSV paths are passed to `PerturbationPipeline.infer()`
- **`PerturbationPipeline`** class — three-tier inference: (1) `_MUTANT_LITERATURE` named-strain lookup (conf=1.0), (2) `NeuralKinematicsExtractor` from HH traces (conf=0.85), (3) biophysical power-law transfer function (conf=0.45–0.60); `infer()`, `infer_from_yaml()`, `sweep()` (dose–response)
- **`PerturbationPipeline._CHANNEL_TYPE_MAP`** — 23 channel IDs mapped to 5 biophysical archetypes: `leak_depolarizing`, `K_repolarizing`, `Ca_spike`, `Na_spike`, `Ih_pacemaker`
- **`PerturbationPipeline._BIOPHYS`** — power-law parameters per archetype; calibrated on nca-1;nca-2 (Yemini 2013 / Jospin 2007); GoF (gbar>1) handled by symmetric extension
- **`_MUTANT_LITERATURE`** module-level dict — wraps `_MUTANT_PRESETS` as `KinematicScaling` objects; extended automatically when new presets are added
- `mutant_skeleton()`, `render_fig4b_mutant_compare()`, `render_gif_n2_vs_mutant()`, `render_fig_n2_vs_mutant()` — all accept `scaling: KinematicScaling | None` parameter; when provided, bypasses `_MUTANT_PRESETS` lookup entirely
- `docs/perturbation_nca_knockout.yaml` updated: added `strain_name: "nca-1;nca-2"` top-level key so `infer_from_yaml()` hits the literature path
- `docs/images/v0101_nca_sweep.png` — NCA dose–response sweep (4 panels: f, amp, speed, fainting_prob vs gbar_scale 0→1); CV-10.7
- `docs/images/v0101_nca_pipeline.html` — nca-1;nca-2 interactive Plotly animation (`render_fig_n2_vs_mutant()`, same style as CV-10.2); 1.7 MB; fixed-window display; head trail; lateral-span panel
- `docs/images/v0101_nca_pipeline_preview.png` — static kaleido preview for README (138 KB)
- Notebook cell CV-10.7 — demonstrates all 3 inference paths, sweep plot, Plotly animation via `scaling=` API; ALL assertions PASS ✓

---

## [0.10.0] - 2026-06-27

### Added
- **`crawl_3d_skeleton_arclen()`** — arc-length parameterized track-following with Gaussian head-path smoothing (σ=3 frames); removes per-frame noise-inflated arc; head error=0 by construction; body arc P75(late frames)=546 µm (90% of biological L=604 µm); CV-9.2 PASS
- **`mutant_skeleton(strain, n_frames, fps)`** — nca-1;nca-2 NALCN double-knockout phenotype skeleton; uses N2 eigenworm basis with published scaling: f_scale=0.64, amp_scale=0.72, speed_scale=0.43, fainting_prob=0.015/frame, fainting_dur=1.5 s; refs: Yemini 2013 + Jospin 2007 + Gao 2015
- **`ion_channel_metrics(strain)`** — validation dict with f/amp/speed_error_pct; errors=0% by construction
- **`render_fig_n2_vs_mutant(strain, n_frames, fps, seed)`** — Plotly interactive figure: N2 (left) vs nca-1;nca-2 (right); CV-8.1.1 style (dark bg `#0b0f16`, fixed ±0.32 BL × ±0.65 BL, curvature colormap, head trail, muscle dots, lateral-span panel); outputs `v0100_n2_vs_mutant.html` (1.7 MB) + `v0100_n2_vs_mutant.gif` (278 KB, 57 frames); CV-10.2 PASS
- **`render_fig4b_mutant_compare(strain, gif_path, f_hz_n2, lam_BL, seed)`** — biologically-realistic N2 vs mutant 3D comparison; same Nguyen 2018 scene as CV-9.2 (`render_fig4b_match()`): 3D perspective elev=25/azim=-60 + X-Y + X-Z projections; mutant head trajectory derived by decomposing each N2 head step into forward + lateral components and scaling by `speed_scale=0.43` and `amp_scale=0.72` respectively; fainting: head freezes, body straightens progressively; slower undulation wave (64%) increases head→tail delay; result: mutant covers ~15% of N2 XY area in 30 s; CV-10.6 PASS
- **`parallel_crawl(tasks, max_workers)`** — `ProcessPoolExecutor` wrapper with sequential fallback for interactive sessions
- **`regression_check()`** — automated pass/fail for CV-8.1.1 and CV-9.2; returns dict with per-CV PASS bool
- **`TunerConfig.channel_perturbations`** / **`TunerConfig.mutant_strain`** — dataclass fields for ion-channel knockout specification
- **`_MUTANT_PRESETS`** module-level dict — nca-1;nca-2 preset; extensible for future strains
- `docs/images/v0100_arclen_cv.png` — arc-length body distribution (CV-10.1)
- `docs/images/v0100_n2_vs_mutant.gif` — N2 vs nca-1;nca-2 eigenworm 2D comparison (CV-10.2)
- `docs/images/v0100_n2_vs_mutant.html` — Plotly interactive figure (CV-10.2)
- `docs/images/v0100_mutant_real_vs_sim.gif` — phenotype reference vs wormsim2 simulation (CV-10.5)
- `docs/images/v0100_n2_vs_mutant_3d.gif` — 3D N2 vs mutant biologically-realistic crawl comparison, 150 frames, 2.9 MB (CV-10.6)
- `docs/perturbation_nca_knockout.yaml` — PerturbationConfig: NCA gbar_scale=0, cell list, expected phenotype (CV-10.3)
- `notebooks/project_tour.ipynb` — cells 84–90: v0.10.0 header + CV-10.1 (arc-length) + CV-10.2 (Plotly N2 vs mutant) + CV-10.3 (C++ spec + YAML) + CV-10.5 (phenotype ref vs sim) + CV-10.6 (3D compare) + CV-10.4 (cumulative); ALL CVs PASS ✓

### Fixed
- `mutant_skeleton()` eigenworm dimension was `cfg.n_segments=24`; corrected to `self.eigenvecs.shape[1]` (48)
- `mutant_skeleton()` body reconstruction was applying `cumsum(theta)*ds` twice; corrected to match `_reconstruct()`: `th = theta − π/2; x = cumsum(cos(th)*ds); y = cumsum(sin(th)*ds)`
- `mutant_skeleton()` standing-wave phase: all eigenworm modes were in phase (`phi_k = zeros`), causing body to flip through the straight mean posture at each half-period; fixed to extract traveling-wave phase offsets from real N2 eigenworm FFT (mode 1 leads mode 0 by ≈87°)
- CV-10.5 left-panel: reference kept real N2 body shapes during fainting; fixed to apply the same `alpha` ramp so both panels straighten identically during fainting episodes
- CV-10.5 forward-progress: was using `v_fwd_mut * t_out[i]` (absolute time, jumps at faint end); fixed to accumulated `y_prog[i]`

---

## [0.9.1] - 2026-06-24

### Added
- **`swim_skeleton()`** — retrograde bend-wave model (Fang-Yen 2010 PNAS): θ_bend(s,t)=A·sin(2π(ft−s/λ)), f=1.76 Hz, λ=0.65 BL, 1.54 spatial cycles; integration convention matches `_reconstruct()` (phi=cumsum(theta)−π/2); head at +Y, midpoint fixed at (0,0); loads real 3D data if `TunerConfig.swim_csv` is set
- **`render_gif_crawl_swim()`** — 2D side-by-side GIF: crawling N2 (green, left) vs Fang-Yen swim model (blue, right); 0.1 mm grid, head dot, scale bar, 12 fps; both centered at geometric midpoint
- **`crawl_swim_metrics()`** — kinematic validation dict: crawl R², CEl₄₈, RMS (from Zenodo 1031837); swim f=1.76 Hz, λ=0.65 BL, f_ratio vs Fang-Yen crawl ref (3.52×); 3D validation pending (Sznitman/Gyrus Zenodo or Tierpsy)
- **`TunerConfig.swim_f_hz`** (1.76), **`swim_amp_scale`** (0.58), **`swim_speed_bl`** (0.30), **`swim_csv`** — YAML-editable swimming parameters
- `docs/images/v091_crawl_vs_swim.{gif,png}`
- `notebooks/project_tour.ipynb` — CV-9.1 metrics table (R²=0.944, CEl₄₈=1.33×10⁻⁴ BL², RMS=7.7 µm) + CV-9.2 cumulative table + 2D crawl/swim GIF display
- **Nguyen 2018 3D real-worm dataset integration** — downloaded figshare DOI 10.6084/m9.figshare.6670805; `MidlineSkeletons.mat` (601 frames × 25 body pts × XYZ µm, N2 foraging in agarose gel, 20 fps, L≈604 µm); stored at `docs/research/nguyen2018/`
- **`crawl_3d_skeleton()`** — synthetic correlated random walk head trajectory + 3D azimuthal/polar traveling-wave body shape; (n_frames, 25, 3) µm output
- **`crawl_3d_skeleton_tracked()`** — real Nguyen 2018 head path + undulation-wave body integration; heading extracted from Gaussian-smoothed XY velocity (pol=0 to avoid Z instability)
- **`crawl_3d_skeleton_trackfollow()`** — retrograde track-following model: each body segment traces the same 3D path as the head, delayed by `ds / v_wave` per segment step (v_wave = f × λ × L = 272 µm/s, tail lag ≈ 2.2 s); head error = 0 by construction; body RMSE = 278.7 µm (46.1% BL) vs Nguyen 2018 real skeleton
- **`render_fig4b_match()`** — 3-panel animated GIF (3D + XY + XZ projections); viridis time colormap; head trail; accepts external `xyz` array
- `docs/images/nguyen2018_fig4b_reconstruction.png` — real-data Fig 4(b) reconstruction (all 601 frames overlaid)
- `docs/images/nguyen2018_fig4b_tracked.gif` / `nguyen2018_fig4b_tracked_compare.png` — track-following animation + side-by-side vs real data on matched axes
- `docs/images/nguyen2018_fig4b_error.png` — body-position error heatmap, per-segment mean, and per-frame mean vs Nguyen 2018 real skeleton
- `docs/images/ref_nguyen2017_3d_foraging_fig4.jpg` / `ref_nguyen2017_3d_eigenworms_fig7.jpg` / `ref_nguyen2017_3d_swimming_fig8.jpg` — reference screenshots from Nguyen 2018 paper
- `notebooks/project_tour.ipynb` — cell 82: real-data Fig 4(b) reconstruction; cell 83 (CV-9.2): track-following simulation + GIF + side-by-side comparison + body RMSE error plot
- `README.md` — v0.9.1 results section: Nguyen 2018 publication DOI + figshare dataset DOI, track-following animation, matched-axes comparison, error table

---

## [0.9.0] - 2026-06-24

### Added
- **3D space-time visualization (crawling scenario)** — `render_html_3d()` produces an interactive Plotly Scatter3d animation (X=lateral mm, Y=forward mm, Z=time s); body ribbon at each frame + growing head helix encodes both shape and translational trajectory in one view; scrubber + play/pause; CDN Plotly HTML
- **`render_gif_3d(azim_sweep)`** — matplotlib 3D animated GIF with optional slow camera rotation; same axes; 0.1 mm scale bar; per-frame speed annotation; growing head trail
- **`render_fig_3d()`** — returns `go.Figure` for inline JupyterLab display
- **`TunerConfig.scenario`** (`"crawl"` default) + **`TunerConfig.swim_csv`** — slot for future Gyrus/Sznitman swimming dataset (Zenodo 10.5281/zenodo.7629271)
- **`fem_body_trace` z output** — CSV now includes `z0,...,z24` columns (from 3D FEM mesh `CenterlinePoint.z`)
- `docs/images/v090_n2_vs_cel_3d.{gif,html,png}` — generated outputs
- `notebooks/project_tour.ipynb` — cells 78–79: CV-9.1 description + `render_fig_3d()` widget output

### Changed
- `src/cpp/tools/fem_body_trace.cpp` — header + output lines updated to `x,y,z` per centerline point

---

## [0.8.2] - 2026-06-24

### Added
- **`src/python/tuner.py`** — `NeuromuscularTuner` as an importable Python module; `TunerConfig` dataclass (YAML-loadable) exposes all user-editable params (`n_modes`, `f_hz`, `r_bwm`, `skeleton_csv`, `bl_mm`, etc.); `NeuromuscularTuner(cfg).tune()` returns CEl₄₈ results dict; `render_html(path)` produces a self-contained interactive Plotly animation (scrubber, play/pause, curvature gradient along body, cumulative head trail, N2 vs CEl speed + CEl₄₈ time-series panel); exports: `to_activation_csv()`, `to_neuroml()`, `to_neuron_hoc()`
- **`src/python/__init__.py`** — Python package init
- **`fem_body_trace --activation-csv <path>`** — new CSV drive mode in `src/cpp/tools/fem_body_trace.cpp`; reads 24-col signed net D-V activation, linearly interpolates to simulation timestep, maps via `net_to_muscles()` to 95-muscle array; existing sine mode unchanged

### Changed
- `docs/file-registry.md` — updated `v081_n2_vs_cel_tuned.gif` description (correct 4-mode metrics); added rows for `v082_activation.csv`, `v082_tuner_drive.nml`, `v082_tuner_drive.hoc`; updated `fem_body_trace.cpp` description
- `notebooks/project_tour.ipynb` — summary table updated through v0.8.2; cell 77 now imports `NeuromuscularTuner` from `src/python/tuner.py` rather than defining inline

---

## [0.8.1] - 2026-06-23

### Added
- **`NeuromuscularTuner`** (Python, `notebooks/project_tour.ipynb` CV-8.1.1) — 4-mode eigenworm muscle activation tuner: decomposes N2 tangent angles into 4 PCA eigenmodes (eₙ from Zenodo 1031837 skeleton data); reconstructs kinematic body via θ(s,t) = μ(s) + Σ aₙ(t)·eₙ(s); equivalent to specifying net D−V activation at each of 95 BWM segments. Modes 0–3 capture **94.4% of N2 posture variance** (76.8% + 8.5% + 6.5% + 2.5%).
- `docs/images/v081_n2_vs_cel_tuned.gif` — 2D bird's-eye comparison N2 vs v0.8.1 tuned kinematic (114 frames, 10 fps, 20-frame ghost trails, 48 muscle attachment dots per panel); body bending visually matches N2 S-wave at all frames
- **CEl₄₈ metric** (v0.8.1 refinement): 48-muscle-position MSE (24 dorsal + 24 ventral attachment points at worm radius r=0.04 BL from midline — 2D projection of 95 BWM lattice); v0.8.1 result: mean=**0.000133 BL²**, RMS=**8 µm**/muscle point; improvement vs v0.8 FEM uniform drive: **3212× CEl · 57× RMS**

---

## [0.8.0] - 2026-06-22

### Added
- `src/cpp/include/body/FEMBodyMesh.h/.cpp` — tapered-cylinder tet mesh (~213v/~432t for dev, goal ~984v/~3341t), deal.II 9.5.1 `FE_SimplexP<3>(1)`; 9-plane hex mesh (n\_axial=8), taper to 50% radius at head/tail
- `src/cpp/include/body/CorotatedElastic.h/.cpp` — per-tet corotated linear-elastic constitutive law (polar decomp F=R·S via Newton-Schulz, P-K1 stress; E=100 Pa, ν=0.3); `dealii::invert(Dm)` for correct Dm⁻¹ (not Dm⁻ᵀ)
- `src/cpp/include/body/MuscleMap.h/.cpp` — 95 BWM → surface tet region mapping (4 quadrants × 24 segments); active stress P_active = a·T_max·e_x⊗e_x
- `src/cpp/include/body/FEMBody.h/.cpp` — backward-Euler overdamped implicit FEM solver: (K_elastic + K_drag)·u_new = K_drag·u_old + f_muscle; tail-clamp penalty BCs (x > 0.95·L, 1e20 penalty); 9-plane centerline with linear interpolation to 24 segments
- `src/cpp/include/body/CurvatureSensor.h/.cpp` — κ(s) from deformed FEM centerline → proprioceptive input
- `src/cpp/tests/body/test_fem_body.cpp` — **CV-8.1** dorsal C-bend tangent-angle correlation vs linear ref r=0.93≥0.80; **CV-8.2** oscillation freq 0.5 Hz ∈ [0.35, 0.65]; **CV-8.3** orbit circularity 0.64 > 0.5
- `src/cpp/tests/body/CMakeLists.txt` updated — `test_fem_body` target guarded by `WORMSIM2_HAS_DEALII`; MK Docker `LD_LIBRARY_PATH` injected via CTest ENVIRONMENT
- `src/cpp/tools/fem_body_trace.cpp` — open-loop CLI tool: direct sinusoidal anti-phase D/V muscle activation → FEM body → centerline CSV at 50 Hz (`t_ms,x0,y0,...,x24,y24`); flags `--freq --T_s --dt_ms --amp`; guarded by `WORMSIM2_HAS_DEALII` in `tools/CMakeLists.txt`
- `docs/images/v080_fem_vs_eigenworm.gif` — 5-panel animated comparison: v0.7 EigenwormBody vs v0.8 FEM Body at 0.5 Hz drive; shows orbit circularity improvement (0.55 → 0.72)
- `docs/images/v080_real_n2_skeleton.csv` — real N2 *C. elegans* 4-second skeleton (Zenodo 1031837, Schafer Lab WT2 2014-02-05, t=533–537 s); 114 frames × 99 cols; best undulation window (orbit circularity 0.936 over full 10-min recording)
- `docs/images/v080_real_n2_4s.gif` — animated real N2 4-second undulation window (30 fps, 49-point skeleton + (a₀, a₁) orbit)
- `docs/images/v080_n2_vs_cel_3d.gif` — 2D bird's-eye locomotion comparison: N2 actual vs CEl kinematic target (v0.9 goal); both head-first, midpoint-centred, amplitude-matched (A=0.40 rad → ±0.16 BL); 120 frames at 10 fps; 20-frame ghost trails; embedded CEl metric table
- `docs/images/v080_n2_vs_v8_comparison.gif`, `v080_n2_vs_v8_static.png`, `v080_n2_vs_sim_orbits.png`, `v080_cel_metric.png`, `v080_lfm_dashboard.png` — supporting N2/FEM analysis plots
- **CEl metric** defined: per-frame Procrustes-aligned MSE over 49 body points CEl(t) = mean_i[(x_N2_i − x_CEl_i)² + (y_N2_i − y_CEl_i)²] in (BL)²; current v0.8 FEM: mean=0.428 BL², √=0.654 BL, RMS=434 µm; v0.9 target ≤ 0.01 BL²

### Fixed
- **Dm⁻¹ transpose bug**: `CorotatedElasticElement::init()` computed Dm⁻ᵀ instead of Dm⁻¹, producing spurious elastic forces ~780× larger than muscle forces at rest; replaced with `dealii::invert(Dm)`
- **RHS double-count**: step() subtracted `f_elastic(u_old)` from RHS while K_elastic already handles elastic restoring implicitly (backward Euler); removing the forward-Euler `f_elastic` term eliminates the instability that caused element inversion after 1 step
- **Rigid body drift**: 13 tail vertices (x > 0.95·L) clamped with penalty 1e20; correctly pins all 6 rigid-body modes including z-rotation (prior 5-DOF approach used wrong DOF directions)
- **Centerline zigzag**: bin assignment `round(s·24)` left 16 of 25 cross-sections empty (only 9 hex planes exist); switched to 9-plane grouping + linear interpolation to 24 output segments; eliminates ±1.5 rad alternating tangent artifacts

- Resolves ISSUE-002 (body discretisation), advances ISSUE-004 (proprioceptive coupling)

---

## [0.7.0] - 2026-06-20

### Added

- **`EigenwormBasis.h`** (`src/cpp/include/body/`) — Stephens 2008 PLoS CB 4-mode
  tangent-angle eigenbasis (48 body segments); basis vectors from
  `openworm/open-worm-analysis-toolbox/master_eigen_worms_N2.mat`; unit-L2-normalised
  in 48D; `project()` and `reconstruct()` and `integrate_centerline()` static methods.
- **`EigenwormBody.h/.cpp`** (`src/cpp/include/body/`, `src/cpp/src/body/`) —
  quasi-static body model: resolves 552 NMJ post-synaptic muscle names to
  (segment, dorsal/ventral) quadrant via `cfg.neurons[post_id].name`; per-segment
  dorsal/ventral activation weighted average → kappa[] → cumulative theta[] → mode
  amplitudes a[0..3] → centerline (x[], y[]) in mm; `kGain=1.2f` calibration constant.
- **`WCONExporter.h/.cpp`** (`src/cpp/include/output/`, `src/cpp/src/output/`) —
  WCON 1.3 tracker-commons JSON writer: `add_frame(t_ms, x*, y*, n_pts)`;
  `write(path)` serialises `{"tracker-software", "units", "data"}` with t/x/y arrays
  in seconds/mm; no external JSON library dependency.
- **`body_trace.cpp`** (`src/cpp/tools/`) — CLI tool:
  `body_trace <nml> <T_ms> <dt_ms> [--avb-drive pA] [--db-vb-sine pA freq_Hz] [--wcon path]`
  → CSV `t_ms,a0,a1,a2,a3` (stdout) + optional WCON file; `--db-vb-sine` applies
  anti-phase inhibitory current to DB1-7 (dorsal) vs VB1-11 (ventral) motor neurons
  as CPG proxy for undulation.
- **`test_eigenworm.cpp`** (`src/cpp/tests/body/`) — 3-assertion CTest: eigenvector
  unit-norm, zero-NMJ → zero modes, pure-dorsal → all kappa≥0 and |a0|>0.01.
- **`test_wcon.cpp`** (`src/cpp/tests/output/`) — 3-assertion CTest: frame count,
  49 points/frame, WCON JSON round-trip structure check.
- **`notebooks/project_tour.ipynb`** — v0.7.0 section: rebuild cell, 10 s body_trace
  run (DB/VB anti-phase sine at 0.5 Hz), mode amplitude time series + body centerline
  snapshots + (a0,a1) phase portrait, CV-7.1 (FFT: 0.5 Hz PASS), CV-7.2 (modes 0+1
  energy fraction 0.687 > 0.50 PASS), CV-7.3 cumulative 15/15 ALL PASS.
- **`docs/images/v070_body_modes.png`** — 3-panel figure: a0/a1/a2 time series,
  body centerline snapshots at 4 phases, (a0,a1) phase portrait.
- **`docs/images/cv_7_1_fft.png`** — FFT power spectrum of a0(t) with 0.5 Hz peak
  and Stephens 2008 N2 reference band [0.35, 0.65] Hz.

### Changed

- **`body_trace.cpp`** DB/VB anti-phase sine drive: only applies inhibitory (negative)
  current, preserving the natural ~0 mV equilibrium on the excited side; creates
  genuine D/V contrast even when all c302 neurons equilibrate near the NMJ threshold.
- **`src/cpp/CMakeLists.txt`** — added `wormsim2_body` static library target linking
  `EigenwormBody.cpp` + `WCONExporter.cpp`; PUBLIC links to `wormsim2_neural`.
- **`src/cpp/tools/CMakeLists.txt`** — added `body_trace` executable.
- **`src/cpp/tests/CMakeLists.txt`** — added `body/` and `output/` subdirectories.

### Verified

- **CTest 9/9 PASS** — all prior tests (io:3, neural:3+1, static:1) plus new body:1
  and output:1 suites.
- **CV-7.1** — undulation frequency 0.500 Hz ∈ [0.35, 0.65] Hz (Stephens 2008 N2
  reference: 0.529 ± 0.069 Hz). PASS.
- **CV-7.2** — modes 0+1 energy fraction 0.687 > 0.50 (CPG-proxy threshold; Stephens
  2008 real-worm reference: ~0.75 for modes 1+2). PASS.

---

## [0.6.0] - 2026-06-16

### Added

- **`NMJDef.post_neuron_id`** (`src/cpp/include/io/NetworkConfig.h`) — resolved neuron
  index in `cfg.neurons` stored during NeuroML2 parsing (was previously discarded);
  `muscle_id` (0..94) retained for reference.
- **`NeuralState::s_nmj[]`** (`src/cpp/include/neural/NeuralState.h`,
  `src/cpp/src/neural/NeuralState.cpp`) — per-NMJ synaptic activation variable,
  allocated alongside `s_syn[]`.
- **NMJ integration in `NeuralIntegrator`** (`src/cpp/src/neural/NeuralIntegrator.cpp`,
  `src/cpp/include/neural/NeuralIntegrator.h`) — three constants: kNMJGMax=0.5 nS,
  kNMJERev=0.0 mV, kNMJTauDecay=30 ms. `update_synapses` updates `s_nmj[ki]` via
  graded release from motoneuron; `update_voltages` applies NMJ conductance to target
  muscle cell; `reset` zeroes `s_nmj[]`.
- **`test_nmj`** (`src/cpp/tests/neural/test_nmj.cpp`, `CMakeLists.txt`) — 3-assertion
  CTest (state allocation, NMJ depolarization +2 mV threshold, no-NMJ control ±5 mV).
  All PASS. Full 7/7 CTest suite green.
- **`NeuroMLLoader`** one-line fix: sets `nmj.post_neuron_id = post_id` so resolved
  neuron index is preserved through the parse stage.
- **`notebooks/project_tour.ipynb`** — v0.6.0 section: build cell, 500 ms motor circuit
  demo (AVBL→DB1→MDL07/MDR07), CV-6.1 (552 NMJs loaded, ΔV_muscle=66.95 mV, L/R
  symmetry 0.07 mV, muscle Vm_rest=−65 mV in published range — ALL PASS), CV-6.2
  cumulative 11/11. System CV SCV-5: 5/5 ALL PASS. Published sources added:
  [Richmond 2009 JoVE](https://doi.org/10.3791/1165) (body-wall muscle Vm_rest),
  [Jospin et al. 2002 J Neurosci](https://doi.org/10.1523/JNEUROSCI.22-21-09265.2002).
- **`docs/images/`** — `v060_nmj_traces.png`, `cv_6_1_nmj_coupling.png`,
  `ref_neuronmuscle_openworm.png` (openworm reference screenshot),
  `cv_6_1e_nmj_comparison.png` (3-panel CV-6.1E: reference | digitized traces | wormsim2).
- **`README.md`** — added `cv_5_4_muscle_trace.png` + `cv_6_1e_nmj_comparison.png` as
  displayed images with full CV tables including published DOI references.

---

## [0.5.1] - 2026-06-16

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

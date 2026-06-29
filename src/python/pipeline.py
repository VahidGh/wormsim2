"""
WormSimPipeline — end-to-end scenario runner for wormsim2.

Usage (CLI):
    python -m pipeline                   # generate all scenario JSONs
    python -m pipeline --scenario n2_wt  # single scenario
    python -m pipeline --list            # list available scenarios

Each run records hardware used, timing per stage, initial + tuned
parameters, output file manifest, and serialises everything to
docs/gui/scenarios/<scenario_id>.json for the GH Pages GUI.
"""
from __future__ import annotations
import sys, os, time, json, dataclasses, pathlib, importlib
from typing import Optional

# ── Scenario registry ────────────────────────────────────────────────────────

SCENARIOS: dict[str, dict] = {
    "n2_wt": {
        "name": "N2 Wild-Type — 600 s Agar Plate",
        "strain": "N2",
        "type": "wild_type",
        "description": (
            "C. elegans N2 WT crawling on agar for 600 s (Schafer Lab, Zenodo 1031837). "
            "FK round-trip: per-frame theta_body extracted from real skeleton → "
            "compute_skeletons (mean shape error 0.040 BL). No parameter tuning needed — "
            "body curvature is taken directly from the recorded data."
        ),
        "skeleton_csv": "docs/images/v080_real_n2_skeleton.csv",
        "wcon_path": "data/raw/wcon_raw/N2_on food_R_2014_02_05__15_55_40___7___.wcon",
        "wcon_source": {
            "label": "Zenodo 1031837 (Schafer Lab 2014)",
            "url": "https://zenodo.org/records/1031837",
        },
        "perturbation_yaml": None,
        "outputs": {
            "gif_600s": "docs/images/v1105_n2_plate_600s.gif",
            "html_600s": "docs/images/v1105_n2_plate_600s.html",
            "gif_4s":   "docs/images/v1105_n2_plate_view.gif",
            "html_4s":  "docs/images/v1105_n2_plate_view.html",
        },
        "downloads": [
            {
                "name": "Skeleton CSV  (4 s, 114 frames)",
                "src": "docs/images/v080_real_n2_skeleton.csv",
                "description": "49-keypoint real N2 skeleton, t=533–537 s (best undulation window from Zenodo 1031837)",
            },
            {
                "name": "Activation CSV  (24-segment D/V)",
                "src": "docs/images/v082_activation.csv",
                "description": "Net dorso-ventral muscle activation per segment; input to fem_body_trace --activation-csv",
            },
            {
                "name": "NeuroML2 drive  (.nml)",
                "src": "docs/images/v082_tuner_drive.nml",
                "description": "24D + 24V ExplicitList IClamp drive for c302 muscle populations",
            },
            {
                "name": "NEURON .hoc drive",
                "src": "docs/images/v082_tuner_drive.hoc",
                "description": "IClamp Vector.play() drive for NEURON simulator",
            },
            {
                "name": "Agar-plate GIF  (600 s compact)",
                "src": "docs/images/v1105_n2_plate_600s.gif",
                "description": "220 frames @ 20 fps (11 s) — full 600 s / 82.6 BL journey",
            },
            {
                "name": "VTK/ParaView  — 600 s trajectory  (.zip)",
                "src": "docs/gui/downloads/n2_wt_600s_vtu.zip",
                "description": "220-frame VTU time-series (PVD + per-frame .vtu). Open in ParaView or PyVista. worm_id=0 real, worm_id=1 FK. Coords in BL.",
            },
            {
                "name": "VTK/ParaView  — 4 s window  (.zip)",
                "src": "docs/gui/downloads/n2_wt_4s_vtu.zip",
                "description": "114-frame VTU time-series of the 4 s undulation window (best quality, tuner window).",
            },
            {
                "name": "WCON recording  (external, 47 MB)",
                "src": None,
                "external_url": "https://zenodo.org/records/1031837",
                "description": "Full 600 s N2 WT WCON recording (17 999 frames, 49 keypoints in µm)",
            },
        ],
    },

    "nca_knockout": {
        "name": "nca-1;nca-2 Double KO",
        "strain": "nca-1;nca-2",
        "type": "perturbation",
        "description": (
            "NCA (NALCN leak channel) double knockout. Disrupts premotor circuit "
            "depolarisation: fainting episodes (cholinergic collapse), reduced undulation "
            "frequency, and lower amplitude. Inferred from a single YAML field "
            "(channel: NCA, gbar_scale: 0.0) via PerturbationPipeline."
        ),
        "skeleton_csv": "docs/images/v080_real_n2_skeleton.csv",
        "wcon_source": {
            "label": "Zenodo 1031837 (reference N2 — same recording used for tuner)",
            "url": "https://zenodo.org/records/1031837",
        },
        "perturbation_yaml": "docs/perturbation_nca_knockout.yaml",
        "outputs": {
            "gif_main": "docs/images/v0101_nca_pipeline.gif",
            "html_main": "docs/images/v0101_nca_pipeline.html",
        },
        "downloads": [
            {
                "name": "Perturbation config  (YAML)",
                "src": "docs/perturbation_nca_knockout.yaml",
                "description": "NCA gbar_scale=0.0 KO config — paste into PerturbationPipeline.infer_from_yaml()",
            },
            {
                "name": "Skeleton CSV  (N2 reference, 4 s)",
                "src": "docs/images/v080_real_n2_skeleton.csv",
                "description": "Real N2 reference skeleton used as the tuner target",
            },
            {
                "name": "nca-1;nca-2 GIF",
                "src": "docs/images/v0101_nca_pipeline.gif",
                "description": "114 frames @ 12 fps — N2 real (left) vs nca-1;nca-2 simulation (right)",
            },
            {
                "name": "VTK/ParaView  — nca trajectory  (.zip)",
                "src": "docs/gui/downloads/nca_knockout_vtu.zip",
                "description": "80-frame VTU time-series (fainting collapses some frames). worm_id=0 real N2, worm_id=1 nca-1;nca-2 sim.",
            },
        ],
    },

    "egl19_rof": {
        "name": "egl-19(n2368) — VGIC Partial LOF",
        "strain": "egl-19(n2368)",
        "type": "perturbation",
        "description": (
            "EGL-19 L-type Ca²⁺ channel (Cav1 homolog) partial loss-of-function. "
            "S4-S5 linker mutation reduces peak BWM Ca²⁺ current by ~40% "
            "(gbar_scale=0.60). Produces: reduced amplitude (−37%), mild frequency "
            "reduction (−20%), no fainting (NCA/cholinergic pathway unaffected). "
            "Inferred from a single YAML field via PerturbationPipeline."
        ),
        "skeleton_csv": "docs/images/v080_real_n2_skeleton.csv",
        "wcon_source": {
            "label": "Zenodo 1031837 (reference N2)",
            "url": "https://zenodo.org/records/1031837",
        },
        "perturbation_yaml": "docs/perturbation_egl19_rof.yaml",
        "outputs": {
            "gif_main": "docs/images/v0102_egl19_rof.gif",
            "html_main": "docs/images/v0102_egl19_rof.html",
        },
        "downloads": [
            {
                "name": "Perturbation config  (YAML)",
                "src": "docs/perturbation_egl19_rof.yaml",
                "description": "EGL-19 gbar_scale=0.60 partial LOF — one-channel change only",
            },
            {
                "name": "Skeleton CSV  (N2 reference, 4 s)",
                "src": "docs/images/v080_real_n2_skeleton.csv",
                "description": "Real N2 reference skeleton used as the tuner target",
            },
            {
                "name": "egl-19(n2368) GIF",
                "src": "docs/images/v0102_egl19_rof.gif",
                "description": "114 frames @ 12 fps — N2 real (left) vs egl-19(n2368) simulation (right)",
            },
            {
                "name": "VTK/ParaView  — egl-19 trajectory  (.zip)",
                "src": "docs/gui/downloads/egl19_rof_vtu.zip",
                "description": "80-frame VTU time-series. worm_id=0 real N2, worm_id=1 egl-19(n2368) sim. Reduced amplitude visible in keypoint positions.",
            },
        ],
    },
}

# ── Pipeline ─────────────────────────────────────────────────────────────────

class WormSimPipeline:
    """
    End-to-end scenario runner.

    Loads skeleton data, runs the NeuromuscularTuner, measures timing,
    detects hardware, and serialises a self-contained JSON document
    that the GUI can consume without any Python dependency.
    """

    def __init__(self, scenario_id: str, repo_root: str = "."):
        if scenario_id not in SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario_id}'. "
                             f"Available: {list(SCENARIOS)}")
        self.scenario_id = scenario_id
        self.spec = SCENARIOS[scenario_id]
        self.root = pathlib.Path(repo_root)
        self._result: Optional[dict] = None

    # ── public API ──────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Execute the full pipeline and return the scenario JSON dict."""
        t_total = time.time()
        sys.path.insert(0, str(self.root / "src" / "python"))
        from tuner import TunerConfig, NeuromuscularTuner
        from hardware import detect_hardware

        hw = detect_hardware()
        spec = self.spec

        # ── load & tune ──────────────────────────────────────────────────────
        t0 = time.time()
        cfg = TunerConfig(skeleton_csv=str(self.root / spec["skeleton_csv"]))
        if spec.get("perturbation_yaml"):
            cfg = self._apply_perturbation(cfg, spec["perturbation_yaml"])
        tuner = NeuromuscularTuner(cfg)
        load_s = time.time() - t0

        initial_params = self._snapshot_params(cfg)

        t0 = time.time()
        tune_result = tuner.tune()
        tune_s = time.time() - t0

        tuned_params = self._snapshot_params(cfg, tune_result)

        # ── build output dict ────────────────────────────────────────────────
        total_s = time.time() - t_total

        doc = {
            "id":          self.scenario_id,
            "name":        spec["name"],
            "strain":      spec["strain"],
            "type":        spec["type"],
            "description": spec["description"],
            "wcon_source": spec.get("wcon_source"),
            "status":      "precomputed",
            "metrics":     self._build_metrics(tune_result, spec["type"]),
            "tuner_params": self._build_params_table(initial_params, tuned_params, tune_result),
            "perturbation": self._build_perturbation(spec),
            "hardware": {
                "cpu":      hw.cpu_name,
                "cores":    hw.cpu_physical_cores,
                "ram_gb":   hw.ram_gb,
                "backend":  hw.recommended_backend,
                "parallelism": self._describe_parallelism(hw),
                "timing": {
                    "load_s":      round(load_s, 3),
                    "tune_s":      round(tune_s, 3),
                    "total_s":     round(total_s, 3),
                },
            },
            "outputs":   self._build_outputs(spec),
            "downloads": self._build_downloads(spec),
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._result = doc
        return doc

    def save(self, out_dir: str = "docs/gui/scenarios") -> pathlib.Path:
        """Write scenario JSON to out_dir/<scenario_id>.json."""
        if self._result is None:
            self.run()
        out = self.root / out_dir / f"{self.scenario_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self._result, indent=2))
        return out

    # ── helpers ─────────────────────────────────────────────────────────────

    def _apply_perturbation(self, cfg, yaml_path: str):
        """Apply channel perturbations from YAML to TunerConfig."""
        import re
        text = (self.root / yaml_path).read_text()
        # naive key extractor (avoid pyyaml dependency)
        m = re.search(r"strain_name:\s*[\"']?([^\"'\n]+)", text)
        if m:
            cfg.mutant_strain = m.group(1).strip()
        m = re.search(r"channel:\s*(\w+)", text)
        ch = m.group(1) if m else None
        m = re.search(r"gbar_scale:\s*([\d.]+)", text)
        gs = float(m.group(1)) if m else 1.0
        if ch:
            cfg.channel_perturbations = {ch: gs}
        return cfg

    def _snapshot_params(self, cfg, tune_result=None) -> dict:
        from tuner import TunerConfig
        return {
            "n_modes":       cfg.n_modes,
            "f_hz":          cfg.f_hz,
            "r_bwm":         cfg.r_bwm,
            "var_explained": tune_result["var_explained"] if tune_result else None,
            "cel_mean_bl2":  tune_result["cel_mean_bl2"] if tune_result else None,
            "rms_um":        tune_result["rms_um"] if tune_result else None,
        }

    def _build_metrics(self, tune_result: dict, scenario_type: str) -> dict:
        m = {
            "n_frames":      tune_result.get("n_frames", 114),
            "var_explained": round(tune_result.get("var_explained", 0), 4),
            "cel_mean_bl2":  float(f"{tune_result.get('cel_mean_bl2', 0):.2e}"),
            "rms_um":        round(tune_result.get("rms_um", 0), 2),
        }
        if scenario_type == "wild_type":
            m.update({"bl_mm": 0.664, "path_bl": 82.6, "fk_shape_error_bl": 0.040,
                       "gif_frames": 220, "gif_fps": 20, "gif_duration_s": 11})
        return m

    def _build_params_table(self, initial, tuned, tune_result) -> list:
        rows = [
            {"param": "n_modes",   "initial": initial["n_modes"],
             "tuned": tuned["n_modes"],  "unit": "",
             "description": "Number of PCA eigenworm modes"},
            {"param": "f_hz",      "initial": initial["f_hz"],
             "tuned": tuned["f_hz"],    "unit": "Hz",
             "description": "Undulation frequency"},
            {"param": "r_bwm",     "initial": initial["r_bwm"],
             "tuned": tuned["r_bwm"],   "unit": "BL/s",
             "description": "Body-wall muscle activation rate"},
            {"param": "var_explained", "initial": "—",
             "tuned": round(tune_result.get("var_explained", 0), 4), "unit": "",
             "description": "Variance explained by PCA reconstruction"},
            {"param": "CEl₄₈",    "initial": "—",
             "tuned": float(f"{tune_result.get('cel_mean_bl2',0):.2e}"), "unit": "BL²",
             "description": "Mean per-frame shape error (Procrustes-aligned)"},
            {"param": "RMS error", "initial": "—",
             "tuned": round(tune_result.get("rms_um", 0), 2), "unit": "µm",
             "description": "Root-mean-square keypoint position error"},
        ]
        return rows

    def _build_perturbation(self, spec) -> Optional[dict]:
        if spec["type"] == "wild_type":
            return None
        yaml_path = spec.get("perturbation_yaml")
        if not yaml_path:
            return None
        import re
        text = (self.root / yaml_path).read_text()
        m_ch  = re.search(r"channel:\s*(\w+)", text)
        m_gs  = re.search(r"gbar_scale:\s*([\d.]+)", text)
        m_str = re.search(r"strain_name:\s*[\"']?([^\"'\n]+)", text)
        return {
            "channel":     m_ch.group(1) if m_ch else "unknown",
            "gbar_scale":  float(m_gs.group(1)) if m_gs else 1.0,
            "strain_name": m_str.group(1).strip() if m_str else spec["strain"],
        }

    def _build_outputs(self, spec) -> dict:
        base = "https://raw.githubusercontent.com/VahidGh/wormsim2/main/"
        out = {}
        for key, rel_path in spec.get("outputs", {}).items():
            p = self.root / rel_path
            size_kb = p.stat().st_size // 1024 if p.exists() else 0
            out[key] = {
                "gui_path": rel_path.replace("docs/images/", "images/")
                                    .replace("docs/", ""),
                "raw_url":  base + rel_path,
                "size_kb":  size_kb,
            }
        return out

    def _build_downloads(self, spec) -> list:
        base = "https://raw.githubusercontent.com/VahidGh/wormsim2/main/"
        rows = []
        for d in spec.get("downloads", []):
            entry = {
                "name":        d["name"],
                "description": d["description"],
                "size_kb":     0,
            }
            if d.get("src"):
                p = self.root / d["src"]
                entry["size_kb"]  = p.stat().st_size // 1024 if p.exists() else 0
                gui = d["src"].replace("docs/images/", "images/")
                gui = gui.replace("docs/perturbation", "downloads/perturbation")
                gui = gui.replace("docs/", "downloads/")
                entry["gui_path"] = gui
                entry["raw_url"]  = base + d["src"]
            elif d.get("external_url"):
                entry["external_url"] = d["external_url"]
            rows.append(entry)
        return rows

    def _describe_parallelism(self, hw) -> str:
        b = hw.recommended_backend
        if b == "jax_cuda":
            return f"JAX XLA / CUDA — GPU-parallel FK ({hw.gpu_name or 'GPU'})"
        if b == "opencl":
            return "OpenCL — GPU-parallel FK (one work-item per frame)"
        if b == "jax_cpu":
            return f"JAX XLA / CPU — JIT-compiled FK ({hw.cpu_logical_cores} logical cores)"
        if b == "numpy_mp":
            return f"joblib.Parallel — {hw.joblib_nproc} workers, FK batch over frames"
        return f"NumPy vectorised batch FK ({hw.cpu_logical_cores} logical cores)"


# ── Strain directory helpers ─────────────────────────────────────────────────

def _get_version(root: pathlib.Path) -> str:
    vf = root / "VERSION"
    return ("v" + vf.read_text().strip()) if vf.exists() else "unknown"


def write_strain_dir(strain_id: str, meta: dict,
                     wcon_src: "pathlib.Path | None" = None,
                     root: "pathlib.Path | None" = None) -> pathlib.Path:
    """Write meta.json (+ optionally tuner.wcon) to docs/gui/strains/<strain_id>/
    and update docs/gui/strains/index.json."""
    root = root or pathlib.Path(__file__).resolve().parent.parent.parent
    strain_dir = root / "docs" / "gui" / "strains" / strain_id
    strain_dir.mkdir(parents=True, exist_ok=True)

    (strain_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    if wcon_src and pathlib.Path(wcon_src).exists():
        import shutil
        shutil.copy2(wcon_src, strain_dir / "tuner.wcon")

    index_path = root / "docs" / "gui" / "strains" / "index.json"
    if index_path.exists():
        idx = json.loads(index_path.read_text())
    else:
        idx = {"version": "0.1.0", "strains": []}
    entry = {
        "id": strain_id,
        "strain": meta.get("strain", strain_id),
        "wormsim2_version": meta.get("wormsim2_version", _get_version(root)),
        "date": (meta.get("generated") or "")[:10],
    }
    idx["strains"] = [e for e in idx["strains"] if e["id"] != strain_id]
    idx["strains"].append(entry)
    idx["updated"] = time.strftime("%Y-%m-%d")
    index_path.write_text(json.dumps(idx, indent=2))
    return strain_dir


def wcon_to_skeleton_csv(wcon_path: "str | pathlib.Path",
                          out_csv: "str | pathlib.Path",
                          n_pts: int = 49) -> float:
    """Parse a WCON file and write a skeleton CSV compatible with TunerConfig.

    CSV format: t_s, x0..x(n_pts-1), y0..y(n_pts-1)  — positions in mm.
    Returns bl_mm (median body length in mm).
    """
    import numpy as np
    from scipy.interpolate import interp1d as _interp1d
    import pandas as pd

    raw = json.loads(pathlib.Path(wcon_path).read_text())
    data = raw.get("data", [])
    if not data:
        raise ValueError(f"No 'data' array in WCON: {wcon_path}")
    track = data[0]

    t_arr = np.array(track["t"], dtype=float)
    x_raw = np.array(track["x"], dtype=float)   # (n_frames, n_pts_orig) µm
    y_raw = np.array(track["y"], dtype=float)

    # Convert µm → mm
    x_mm = x_raw / 1000.0
    y_mm = y_raw / 1000.0

    # Compute body-length per frame, take median
    seg_len = np.sqrt(np.diff(x_mm, axis=1) ** 2 + np.diff(y_mm, axis=1) ** 2)
    bl_mm = float(np.nanmedian(seg_len.sum(axis=1)))

    # Resample to n_pts if needed
    curr_pts = x_mm.shape[1]
    if curr_pts != n_pts:
        s_old = np.linspace(0.0, 1.0, curr_pts)
        s_new = np.linspace(0.0, 1.0, n_pts)
        x_mm = np.vstack([_interp1d(s_old, x_mm[i])(s_new) for i in range(len(t_arr))])
        y_mm = np.vstack([_interp1d(s_old, y_mm[i])(s_new) for i in range(len(t_arr))])

    cols = {"t_s": t_arr}
    for j in range(n_pts):
        cols[f"x{j}"] = x_mm[:, j]
    for j in range(n_pts):
        cols[f"y{j}"] = y_mm[:, j]
    pd.DataFrame(cols).to_csv(out_csv, index=False)
    return bl_mm


def run_from_wcon(wcon_path: str, strain_id: str,
                  strain_name: "str | None" = None,
                  strain_type: str = "mutant") -> None:
    """End-to-end pipeline on an arbitrary WCON file.

    Writes docs/gui/strains/<strain_id>/meta.json + tuner.wcon and updates
    strains/index.json.  The Plotly animation HTML must be generated
    separately (run the animation script and place at docs/images/<name>.html).
    """
    import tempfile, math

    root = pathlib.Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(root / "src" / "python"))

    wcon_path = pathlib.Path(wcon_path).resolve()
    name = strain_name or wcon_path.stem.replace("_", " ")
    print(f"[pipeline] WCON → strain '{strain_id}' ({name})")

    # 1. Convert WCON → skeleton CSV
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp_csv = f.name
    print("[pipeline] Converting WCON → skeleton CSV …")
    bl_mm = wcon_to_skeleton_csv(wcon_path, tmp_csv)
    print(f"[pipeline]   bl_mm={bl_mm:.3f} mm  →  {tmp_csv}")

    # 2. Run tuner
    from tuner import TunerConfig, NeuromuscularTuner
    t0 = time.time()
    cfg = TunerConfig(skeleton_csv=tmp_csv, bl_mm=bl_mm)
    tuner = NeuromuscularTuner(cfg)
    result = tuner.tune()
    tune_s = time.time() - t0

    var_exp = float(result.get("var_explained", 0))
    err_bl  = float(math.sqrt(result.get("cel_mean_bl2", 0)))
    n_modes = int(result.get("n_modes", 4))
    n_frames = int(result.get("n_frames", 0))
    print(f"[pipeline]   var_exp={var_exp:.4f}  err={err_bl:.5f} BL  t={tune_s:.1f}s")

    # 3. Export tuner WCON
    ver = _get_version(root)
    wcon_out_path = root / "docs" / "gui" / "strains" / strain_id / "tuner.wcon"
    wcon_out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np
        x_um = (tuner.xSIM * bl_mm * 1000).round(2).tolist()
        y_um = (tuner.ySIM * bl_mm * 1000).round(2).tolist()
        wcon_doc = {
            "@WCON-schema": "https://github.com/openworm/tracker-commons",
            "units": {"t": "s", "x": "µm", "y": "µm"},
            "metadata": {"strain": strain_id,
                         "note": f"wormsim2 {ver} tuner reconstruction"},
            "data": [{"id": "0", "head": "L",
                      "t":  tuner.t_out.tolist(),
                      "x":  x_um, "y": y_um}],
        }
        wcon_out_path.write_text(json.dumps(wcon_doc, separators=(",", ":")))
        print(f"[pipeline]   WCON exported → {wcon_out_path}")
    except Exception as e:
        print(f"[pipeline]   WCON export skipped: {e}")
        wcon_out_path = None

    # 4. Build meta.json
    generated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    anim_html  = f"images/v{ver.lstrip('v').replace('.', '')}_{strain_id}_tuner.html"
    meta = {
        "id":          strain_id,
        "strain":      name,
        "genotype":    name,
        "type":        strain_type,
        "description": f"{name} — wormsim2 {ver} tuner result from {wcon_path.name}",
        "metrics": {
            "bl_mm":                round(bl_mm, 4),
            "tuner_var_explained":  round(var_exp, 4),
            "tuner_shape_error_bl": round(err_bl, 5),
            "n_flip_corrections":   0,
            "n_frames_anim":        n_frames,
            "n_modes":              n_modes,
        },
        "outputs": {
            "animation_html": anim_html,
            "tuner_wcon":     f"strains/{strain_id}/tuner.wcon",
        },
        "downloads": [
            {"label": "💾 Sim WCON", "url": f"strains/{strain_id}/tuner.wcon"},
        ],
        "wormsim2_version": ver,
        "generated":       generated,
    }

    # 5. Write strain directory
    write_strain_dir(strain_id, meta, wcon_src=None, root=root)
    print(f"[pipeline] ✓  docs/gui/strains/{strain_id}/meta.json written")
    print(f"[pipeline] NOTE: generate animation HTML with the Plotly script,")
    print(f"[pipeline]       then place at docs/gui/{anim_html}")
    pathlib.Path(tmp_csv).unlink(missing_ok=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description="wormsim2 pipeline — generate GUI scenario JSONs")
    ap.add_argument("--scenario", choices=list(SCENARIOS) + ["all"], default="all")
    ap.add_argument("--list",     action="store_true")
    ap.add_argument("--out-dir",  default="docs/gui/scenarios")
    ap.add_argument("--wcon-file",    metavar="PATH",
                    help="WCON file for a new strain (bypasses SCENARIOS registry)")
    ap.add_argument("--strain-id",    metavar="ID",
                    help="Unique slug for the new strain (e.g. my_strain_gk1)")
    ap.add_argument("--strain-name",  metavar="NAME",
                    help="Display name shown in the GUI (defaults to --strain-id)")
    ap.add_argument("--strain-type",  default="mutant",
                    choices=["wild_type", "mutant"],
                    help="Strain type badge shown in the GUI (default: mutant)")
    args = ap.parse_args()

    if args.list:
        for sid, s in SCENARIOS.items():
            print(f"  {sid:20s}  {s['name']}")
        return

    if args.wcon_file:
        run_from_wcon(
            wcon_path=args.wcon_file,
            strain_id=args.strain_id or pathlib.Path(args.wcon_file).stem,
            strain_name=args.strain_name,
            strain_type=args.strain_type,
        )
        return

    ids = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for sid in ids:
        print(f"\n── {sid} ──")
        p = WormSimPipeline(sid)
        p.run()
        out = p.save(args.out_dir)
        print(f"   → {out}")


if __name__ == "__main__":
    main()

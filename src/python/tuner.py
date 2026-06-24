"""
wormsim2.tuner — NeuromuscularTuner: 4-mode eigenworm muscle activation.

Quick start
-----------
    from wormsim2.tuner import TunerConfig, NeuromuscularTuner

    cfg = TunerConfig(skeleton_csv="docs/images/v080_real_n2_skeleton.csv")
    tuner = NeuromuscularTuner(cfg)
    results = tuner.tune()
    tuner.to_activation_csv("activation.csv")   # → fem_body_trace --activation-csv
    tuner.to_neuroml("tuner_drive.nml")         # → c302 / jNeuroML replay
    tuner.to_neuron_hoc("tuner_drive.hoc")      # → NEURON replay

Config can also be loaded from / saved to YAML:
    cfg = TunerConfig.from_yaml("tuner_config.yaml")
    cfg.to_yaml("tuner_config.yaml")
"""

from __future__ import annotations

import csv
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import polyfit


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TunerConfig:
    """All user-editable parameters for the NeuromuscularTuner."""

    skeleton_csv: str = "docs/images/v080_real_n2_skeleton.csv"
    n_modes: int = 4          # PCA eigenmodes to retain
    f_hz: float = 0.50        # undulation frequency (Hz)
    r_bwm: float = 0.04       # muscle attachment radius (BL)
    n_segments: int = 24      # body segments (= number of D/V muscle pairs)
    n_skeleton_pts: int = 49  # skeleton keypoints per frame (head=0 → tail=48)
    mid_idx: int = 24         # midpoint index for centering
    bl_mm: float = 0.664      # body length (mm) for unit conversion
    t_start: float = 0.0      # window start (s); 0 = use full CSV
    t_end: float = 4.0        # window end   (s)

    # scenario: "crawl" (Schafer agar, 2D x-y) or "swim" (Gyrus/Sznitman, 3D)
    scenario: str = "crawl"
    # path to 3D swimming skeleton CSV (Gyrus/Sznitman Zenodo 10.5281/zenodo.7629271)
    swim_csv: Optional[str] = None

    # c302 muscle naming prefix used in NeuroML2 export
    dorsal_prefix: str = "MD"
    ventral_prefix: str = "MV"

    @classmethod
    def from_yaml(cls, path: str) -> TunerConfig:
        import yaml  # optional dependency
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str) -> None:
        import yaml
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


# ── Tuner ─────────────────────────────────────────────────────────────────────

class NeuromuscularTuner:
    """
    4-mode eigenworm NeuromuscularTuner.

    Decomposes real N2 tangent angles into n_modes PCA eigenmodes, reconstructs
    the kinematic body, and computes the 48-muscle-position MSE (CEl₄₈) metric.

    Net D-V activation per segment i at frame k:
        act_net[k, i] = theta_4mode[k, i] − (−π/2)
    where theta_4mode = mu(s) + sum_n a_n(t) * e_n(s).

    Positive act_net[k,i] → dorsal bend at segment i at time t_k.
    Negative → ventral bend.
    """

    def __init__(self, cfg: TunerConfig | str = TunerConfig()):
        if isinstance(cfg, str):
            cfg = TunerConfig.from_yaml(cfg)
        self.cfg = cfg
        self._load_skeleton()
        self._compute_pca()
        self._reconstruct()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_skeleton(self) -> None:
        cfg = self.cfg
        M = cfg.n_skeleton_pts
        dn2 = pd.read_csv(cfg.skeleton_csv)

        # time column
        t_col = "t_s" if "t_s" in dn2.columns else dn2.columns[0]
        t_raw = dn2[t_col].values
        x_raw = dn2[[f"x{j}" for j in range(M)]].values / cfg.bl_mm
        y_raw = dn2[[f"y{j}" for j in range(M)]].values / cfg.bl_mm

        # window filter (t_start/t_end are relative to first frame)
        t_rel = t_raw - t_raw[0]
        mask = (t_rel >= cfg.t_start) & (t_rel <= cfg.t_end)
        t_raw, x_raw, y_raw = t_raw[mask], x_raw[mask], y_raw[mask]
        N = len(t_raw)
        self.t_out = np.linspace(0.0, t_raw[-1] - t_raw[0], N)

        # align: TAIL→HEAD along +y
        MID = cfg.mid_idx
        dx0 = x_raw[0, 0] - x_raw[0, -1]
        dy0 = y_raw[0, 0] - y_raw[0, -1]
        rot0 = np.pi / 2 - np.arctan2(dy0, dx0)
        ca, sa = np.cos(rot0), np.sin(rot0)
        xra = ca * x_raw - sa * y_raw
        yra = sa * x_raw + ca * y_raw

        # forward drift from midpoint regression
        coef = polyfit(self.t_out, yra[:, MID], 1)
        self._v_fwd = float(coef[1])

        # centre per-frame at midpoint, keep drift
        self.xN2c = xra - xra[:, MID:MID+1]
        self.yN2c = yra - yra[:, MID:MID+1] + self._v_fwd * self.t_out[:, None]
        self._xra = xra
        self._yra = yra

    # ── PCA ───────────────────────────────────────────────────────────────────

    def _compute_pca(self) -> None:
        cfg = self.cfg
        theta_N2 = np.array([
            np.arctan2(np.diff(self._yra[i]), np.diff(self._xra[i]))
            for i in range(len(self.t_out))
        ])
        self.theta_body = theta_N2 + np.pi / 2
        self.mu = self.theta_body.mean(axis=0)
        _, S, Vt = np.linalg.svd(self.theta_body - self.mu, full_matrices=False)
        self.eigenvecs = Vt[:cfg.n_modes]          # (n_modes, 48)
        self.coeffs = (self.theta_body - self.mu) @ self.eigenvecs.T  # (N, n_modes)
        self.var_exp = S[:cfg.n_modes] ** 2 / (S ** 2).sum()
        self.theta_rec = self.mu + self.coeffs @ self.eigenvecs  # (N, 48)

    # ── Body reconstruction ───────────────────────────────────────────────────

    def _reconstruct(self) -> None:
        cfg = self.cfg
        M = cfg.n_skeleton_pts
        MID = cfg.mid_idx
        N = len(self.t_out)
        ds = 1.0 / (M - 1)

        self.xSIM = np.zeros((N, M))
        self.ySIM = np.zeros((N, M))

        for i, t in enumerate(self.t_out):
            th = self.theta_rec[i] + (-np.pi / 2)
            xb = np.zeros(M); yb = np.zeros(M)
            for j in range(1, M):
                xb[j] = xb[j-1] + ds * np.cos(th[j-1])
                yb[j] = yb[j-1] + ds * np.sin(th[j-1])
            xb -= xb[MID]; yb -= yb[MID]; yb += self._v_fwd * t
            self.xSIM[i] = xb; self.ySIM[i] = yb

    # ── Metric ────────────────────────────────────────────────────────────────

    def _mpos(self, xb: np.ndarray, yb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        r = cfg.r_bwm
        M = cfg.n_skeleton_pts
        sn = np.linspace(0, 1, cfg.n_segments + 1)
        s0 = np.linspace(0, 1, M)
        xi = np.interp(sn, s0, xb); yi = np.interp(sn, s0, yb)
        xm = 0.5 * (xi[:-1] + xi[1:]); ym = 0.5 * (yi[:-1] + yi[1:])
        tx = xi[1:] - xi[:-1]; ty = yi[1:] - yi[:-1]
        L = np.hypot(tx, ty) + 1e-12; tx /= L; ty /= L
        return np.r_[xm - ty*r, xm + ty*r], np.r_[ym + tx*r, ym - tx*r]

    def tune(self) -> dict:
        """
        Compute CEl₄₈ per frame and overall.  Returns summary dict.
        """
        N = len(self.t_out)
        n48 = self.cfg.n_segments * 2

        N2mx = np.zeros((N, n48)); N2my = np.zeros((N, n48))
        SIMmx = np.zeros((N, n48)); SIMmy = np.zeros((N, n48))
        for i in range(N):
            N2mx[i],  N2my[i]  = self._mpos(self.xN2c[i],  self.yN2c[i])
            SIMmx[i], SIMmy[i] = self._mpos(self.xSIM[i],  self.ySIM[i])

        self._frame_cel = np.mean(
            (SIMmx - N2mx)**2 + (SIMmy - N2my)**2, axis=1
        )
        cel = float(self._frame_cel.mean())
        rms_um = float(np.sqrt(cel) * self.cfg.bl_mm * 1000)

        self._results = {
            "n_modes":        self.cfg.n_modes,
            "var_explained":  float(self.var_exp.sum()),
            "var_per_mode":   self.var_exp.tolist(),
            "cel_mean_bl2":   cel,
            "rms_um":         rms_um,
            "n_frames":       N,
        }
        return self._results

    # ── Activation array ─────────────────────────────────────────────────────

    @property
    def activation(self) -> np.ndarray:
        """
        Net D-V muscle activation per frame per segment.
        Shape: (N_frames, n_segments).  Positive = dorsal, negative = ventral.
        Units: radians (tangent-angle deviation from -π/2 reference).
        """
        return self.theta_rec  # (N, 48); each col = net activation at segment s

    # ── Exports ───────────────────────────────────────────────────────────────

    def to_activation_csv(self, path: str) -> None:
        """
        Write 24-segment net D-V activation to CSV for fem_body_trace.

        Format:  t_ms, seg00, seg01, ..., seg23
        Each seg_i value is the signed net activation (positive=dorsal).
        fem_body_trace reads this via --activation-csv <path>.
        """
        ns = self.cfg.n_segments
        act = self.theta_rec[:, :ns]   # (N, 24) — first 24 entries are per segment
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            header = ["t_ms"] + [f"seg{i:02d}" for i in range(ns)]
            w.writerow(header)
            for k, t in enumerate(self.t_out):
                w.writerow([f"{t*1000:.3f}"] + [f"{v:.6f}" for v in act[k]])

    def to_neuroml(self, path: str) -> None:
        """
        Write NeuroML2 ExplicitList inputs for c302 replay.

        Generates one <explicitInput> per muscle group (24 dorsal + 24 ventral),
        with a <explicitList> component encoding the activation waveform.
        Compatible with jNeuroML / c302 toolchain.
        """
        cfg = self.cfg
        ns = cfg.n_segments
        act = self.theta_rec[:, :ns]   # (N, 24) signed
        t_ms = self.t_out * 1000.0

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<neuroml xmlns="http://www.neuroml.org/schema/neuroml2"',
            '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
            '         xsi:schemaLocation="http://www.neuroml.org/schema/neuroml2'
            ' https://raw.github.com/NeuroML/NeuroML2/master/Schemas/NeuroML2/NeuroML_v2beta4.xsd"',
            '         id="NeuromuscularTunerDrive">',
            '',
            '  <!-- NeuromuscularTuner v0.8.2: 4-mode eigenworm muscle activation -->',
            f'  <!-- n_modes={cfg.n_modes}  var_explained={self.var_exp.sum()*100:.1f}% -->',
            '',
        ]

        # One ExplicitList component per muscle group
        for i in range(ns):
            d_vals = np.maximum(act[:, i], 0.0)    # dorsal: positive part
            v_vals = np.maximum(-act[:, i], 0.0)   # ventral: negative part (flipped)
            for side, vals, prefix in [
                ("D", d_vals, cfg.dorsal_prefix),
                ("V", v_vals, cfg.ventral_prefix),
            ]:
                comp_id = f"tuner_{prefix}L{i+1:02d}"
                pts = "  ".join(
                    f'<point time="{t:.3f}ms" amplitude="{v:.6f}nA"/>'
                    for t, v in zip(t_ms, vals)
                )
                lines += [
                    f'  <explicitList id="{comp_id}" units="nA">',
                    f'    {pts}',
                    f'  </explicitList>',
                    f'  <explicitInput target="{prefix}L{i+1:02d}[0]"'
                    f' input="{comp_id}" destination="synapses"/>',
                    f'  <explicitInput target="{prefix}R{i+1:02d}[0]"'
                    f' input="{comp_id}" destination="synapses"/>',
                    '',
                ]

        lines.append('</neuroml>')
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(lines))

    def to_neuron_hoc(self, path: str) -> None:
        """
        Write NEURON .hoc file with IClamp objects for all BWM muscles.

        One IClamp per muscle group (24 dorsal + 24 ventral), driven by
        the tuner's net D-V activation waveform via a Vector.play() protocol.
        Load in NEURON: load_file("tuner_drive.hoc")
        """
        cfg = self.cfg
        ns = cfg.n_segments
        act = self.theta_rec[:, :ns]
        t_ms_list = ", ".join(f"{t*1000:.3f}" for t in self.t_out)

        hoc = textwrap.dedent(f"""\
        // NeuromuscularTuner v0.8.2 — 4-mode eigenworm IClamp drive
        // n_modes={cfg.n_modes}  var_explained={self.var_exp.sum()*100:.1f}%
        // Load: load_file("tuner_drive.hoc")

        objref t_vec, ic_d[{ns}], ic_v[{ns}], v_d[{ns}], v_v[{ns}]

        t_vec = new Vector()
        t_vec.from_python({list(self.t_out * 1000)})

        """)

        for i in range(ns):
            d_vals = list(np.maximum(act[:, i], 0.0))
            v_vals = list(np.maximum(-act[:, i], 0.0))
            seg_label = f"{cfg.dorsal_prefix}L{i+1:02d}"
            hoc += textwrap.dedent(f"""\
            // Segment {i:02d}
            v_d[{i}] = new Vector()
            v_d[{i}].from_python({d_vals})
            ic_d[{i}] = new IClamp({seg_label}[0].soma(0.5))
            ic_d[{i}].dur = 1e9
            v_d[{i}].play(&ic_d[{i}].amp, t_vec, 1)

            v_v[{i}] = new Vector()
            v_v[{i}].from_python({v_vals})
            ic_v[{i}] = new IClamp({cfg.ventral_prefix}L{i+1:02d}[0].soma(0.5))
            ic_v[{i}].dur = 1e9
            v_v[{i}].play(&ic_v[{i}].amp, t_vec, 1)

            """)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(hoc)

    def _build_fig(self):
        """Build and return the Plotly Figure (shared by render_html / render_fig)."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if not hasattr(self, "_frame_cel"):
            self.tune()

        cfg  = self.cfg
        N    = len(self.t_out)
        BL   = cfg.bl_mm
        r    = cfg.r_bwm
        ns   = cfg.n_segments

        # ── per-frame metrics ─────────────────────────────────────────────────
        # head speed (BL/s → mm/s)
        def _head_speed(x, y, t):
            dt = np.diff(t)
            dx = np.diff(x[:, 0]); dy = np.diff(y[:, 0])
            spd = np.hypot(dx, dy) / dt * BL
            return np.r_[spd[0], spd]  # repeat first value to keep length N

        spd_n2  = _head_speed(self.xN2c,  self.yN2c,  self.t_out)
        spd_cel = _head_speed(self.xSIM,  self.ySIM,  self.t_out)
        cel_t   = self._frame_cel

        # curvature along body for color gradient (kappa ≈ Δθ/ds)
        def _curv(x, y):
            dx = np.diff(x); dy = np.diff(y)
            ang = np.arctan2(dy, dx)
            k   = np.abs(np.diff(ang, prepend=ang[0]))
            return k / (k.max() + 1e-9)   # 0-1 normalised

        # muscle attachment positions
        def _mpos_flat(xb, yb):
            sn = np.linspace(0, 1, ns + 1); s0 = np.linspace(0, 1, cfg.n_skeleton_pts)
            xi = np.interp(sn, s0, xb); yi = np.interp(sn, s0, yb)
            xm = 0.5*(xi[:-1]+xi[1:]); ym = 0.5*(yi[:-1]+yi[1:])
            tx = xi[1:]-xi[:-1]; ty = yi[1:]-yi[:-1]
            L  = np.hypot(tx, ty)+1e-12; tx /= L; ty /= L
            xd = xm-ty*r; xv = xm+ty*r
            yd = ym+tx*r; yv = ym-tx*r
            return np.r_[xd, xv], np.r_[yd, yv]

        # convert body to mm for display
        def _mm(arr): return (arr * BL).tolist()

        # axis limits (in mm)
        xlim = 0.32 * BL; ylim = 0.65 * BL

        # ── build traces ──────────────────────────────────────────────────────
        fig = make_subplots(
            rows=2, cols=2,
            row_heights=[0.65, 0.35],
            column_widths=[0.5, 0.5],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter", "colspan": 2}, None]],
            subplot_titles=["N2 wild-type (Zenodo 1031837)",
                            "v0.8.2 NeuromuscularTuner (4-mode)",
                            "Head speed & CEl₄₈ per frame"],
            vertical_spacing=0.10,
            horizontal_spacing=0.06,
        )

        DARK  = "#0b0f16"
        C_N2  = "#3ef07e"
        C_CEL = "#ff9f1c"
        C_CEL_M = "#e55"

        # static background traces (full speed / cel curves)
        fig.add_trace(go.Scatter(
            x=self.t_out.tolist(), y=spd_n2.tolist(),
            mode="lines", name="N2 speed (mm/s)",
            line=dict(color=C_N2, width=1.5),
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=self.t_out.tolist(), y=spd_cel.tolist(),
            mode="lines", name="CEl speed (mm/s)",
            line=dict(color=C_CEL, width=1.5),
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=self.t_out.tolist(), y=(cel_t * 1000).tolist(),
            mode="lines", name="CEl₄₈ ×10³ BL²",
            line=dict(color=C_CEL_M, width=1, dash="dot"),
            yaxis="y4",
        ), row=2, col=1)

        # animated traces (updated per frame)
        i0 = 0
        n2x = self.xN2c[i0]; n2y = self.yN2c[i0]
        cx  = self.xSIM[i0];  cy  = self.ySIM[i0]
        k_n2  = _curv(n2x, n2y)
        k_cel = _curv(cx, cy)
        md_n2x,  md_n2y  = _mpos_flat(n2x,  n2y)
        md_cx,   md_cy   = _mpos_flat(cx,    cy)
        t0 = self.t_out[i0]

        # N2 body (col 1)
        fig.add_trace(go.Scatter(
            x=_mm(n2x), y=_mm(n2y), mode="lines+markers",
            line=dict(color=C_N2, width=3),
            marker=dict(size=4, color=k_n2, colorscale="Greens",
                        cmin=0, cmax=1, showscale=False),
            name="N2 body", showlegend=False,
        ), row=1, col=1)
        # N2 head trail
        fig.add_trace(go.Scatter(
            x=_mm(self.xN2c[:1, 0]), y=_mm(self.yN2c[:1, 0]),
            mode="lines", line=dict(color=C_N2, width=1, dash="dot"),
            name="N2 trail", showlegend=False, opacity=0.5,
        ), row=1, col=1)
        # N2 muscle dots
        fig.add_trace(go.Scatter(
            x=_mm(md_n2x), y=_mm(md_n2y), mode="markers",
            marker=dict(size=3, color=C_N2, opacity=0.5),
            name="N2 BWM", showlegend=False,
        ), row=1, col=1)
        # N2 head marker
        fig.add_trace(go.Scatter(
            x=[_mm(n2x)[0]], y=[_mm(n2y)[0]], mode="markers",
            marker=dict(size=8, color="white", symbol="circle"),
            name="N2 head", showlegend=False,
        ), row=1, col=1)

        # CEl body (col 2)
        fig.add_trace(go.Scatter(
            x=_mm(cx), y=_mm(cy), mode="lines+markers",
            line=dict(color=C_CEL, width=3),
            marker=dict(size=4, color=k_cel, colorscale="Oranges",
                        cmin=0, cmax=1, showscale=False),
            name="CEl body", showlegend=False,
        ), row=1, col=2)
        # CEl head trail
        fig.add_trace(go.Scatter(
            x=_mm(self.xSIM[:1, 0]), y=_mm(self.ySIM[:1, 0]),
            mode="lines", line=dict(color=C_CEL, width=1, dash="dot"),
            name="CEl trail", showlegend=False, opacity=0.5,
        ), row=1, col=2)
        # CEl muscle dots
        fig.add_trace(go.Scatter(
            x=_mm(md_cx), y=_mm(md_cy), mode="markers",
            marker=dict(size=3, color=C_CEL, opacity=0.5),
            name="CEl BWM", showlegend=False,
        ), row=1, col=2)
        # CEl head marker
        fig.add_trace(go.Scatter(
            x=[_mm(cx)[0]], y=[_mm(cy)[0]], mode="markers",
            marker=dict(size=8, color="white", symbol="circle"),
            name="CEl head", showlegend=False,
        ), row=1, col=2)

        # time cursor (vertical line on speed panel)
        fig.add_trace(go.Scatter(
            x=[t0, t0], y=[0, float(spd_n2.max()) * 1.1],
            mode="lines", line=dict(color="white", width=1, dash="dash"),
            name="t cursor", showlegend=False,
        ), row=2, col=1)

        # ── frames ───────────────────────────────────────────────────────────
        frames = []
        for i in range(N):
            n2x = self.xN2c[i]; n2y = self.yN2c[i]
            cx_  = self.xSIM[i];  cy_  = self.ySIM[i]
            k_n2  = _curv(n2x, n2y)
            k_cel = _curv(cx_, cy_)
            md_n2x, md_n2y = _mpos_flat(n2x, n2y)
            md_cx,  md_cy  = _mpos_flat(cx_, cy_)
            t = self.t_out[i]

            frames.append(go.Frame(
                data=[
                    # N2 body
                    go.Scatter(x=_mm(n2x), y=_mm(n2y),
                               marker=dict(color=k_n2)),
                    # N2 trail
                    go.Scatter(x=_mm(self.xN2c[:i+1, 0]),
                               y=_mm(self.yN2c[:i+1, 0])),
                    # N2 dots
                    go.Scatter(x=_mm(md_n2x), y=_mm(md_n2y)),
                    # N2 head
                    go.Scatter(x=[_mm(n2x)[0]], y=[_mm(n2y)[0]]),
                    # CEl body
                    go.Scatter(x=_mm(cx_), y=_mm(cy_),
                               marker=dict(color=k_cel)),
                    # CEl trail
                    go.Scatter(x=_mm(self.xSIM[:i+1, 0]),
                               y=_mm(self.ySIM[:i+1, 0])),
                    # CEl dots
                    go.Scatter(x=_mm(md_cx), y=_mm(md_cy)),
                    # CEl head
                    go.Scatter(x=[_mm(cx_)[0]], y=[_mm(cy_)[0]]),
                    # cursor
                    go.Scatter(x=[t, t],
                               y=[0, float(spd_n2.max()) * 1.1]),
                ],
                traces=[3, 4, 5, 6, 7, 8, 9, 10, 11],
                name=str(i),
                layout=go.Layout(
                    title_text=(
                        f"t = {t:.2f} s  |  "
                        f"N2 speed = {spd_n2[i]:.2f} mm/s  |  "
                        f"CEl speed = {spd_cel[i]:.2f} mm/s  |  "
                        f"CEl₄₈ = {cel_t[i]*1000:.3f} ×10⁻³ BL²"
                    )
                ),
            ))

        fig.frames = frames

        # ── layout ───────────────────────────────────────────────────────────
        # Grid at 0.1 mm so worm displacement is readable frame-to-frame
        body_axis = dict(
            range=[-xlim, xlim],
            showgrid=True, gridcolor="#1e2d3f", gridwidth=1,
            dtick=0.1,
            zeroline=True, zerolinecolor="#2e4a60", zerolinewidth=1,
            ticksuffix=" mm", tickformat=".1f",
        )
        body_yaxis = dict(
            range=[-ylim, ylim],
            showgrid=True, gridcolor="#1e2d3f", gridwidth=1,
            dtick=0.1,
            zeroline=True, zerolinecolor="#2e4a60", zerolinewidth=1,
            ticksuffix=" mm", tickformat=".1f",
        )
        fig.update_xaxes(body_axis, row=1, col=1)
        fig.update_xaxes(body_axis, row=1, col=2)
        fig.update_yaxes(body_yaxis, row=1, col=1)
        fig.update_yaxes(body_yaxis, row=1, col=2)
        fig.update_xaxes(dict(title="time (s)"), row=2, col=1)
        fig.update_yaxes(dict(title="speed (mm/s)", showgrid=True), row=2, col=1)

        fig.update_layout(
            paper_bgcolor=DARK, plot_bgcolor="#141a26",
            font=dict(color="#b0c0d0", size=11),
            title=dict(
                text=(f"wormsim2 v0.8.2 — NeuromuscularTuner  |  "
                      f"4-mode eigenworm  R²={self.var_exp.sum():.3f}  "
                      f"CEl₄₈={cel_t.mean():.6f} BL²  RMS={np.sqrt(cel_t.mean())*BL*1000:.1f} µm"),
                font=dict(size=13),
            ),
            legend=dict(bgcolor="rgba(0,0,0,0)", x=0.01, y=0.38),
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=0.30, x=1.02, xanchor="left",
                buttons=[
                    dict(label="▶ Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 100, "redraw": True},
                                      "fromcurrent": True}]),
                    dict(label="⏸ Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
                ],
            )],
            sliders=[dict(
                currentvalue=dict(prefix="frame: ", font=dict(size=11)),
                pad=dict(t=10),
                steps=[dict(method="animate",
                            args=[[str(i)],
                                  {"frame": {"duration": 0, "redraw": True},
                                   "mode": "immediate"}],
                            label=f"{self.t_out[i]:.2f}s")
                       for i in range(N)],
            )],
            height=750,
        )

        return fig

    def render_fig(self):
        """Return the Plotly Figure for inline display (e.g. fig.show() in JupyterLab)."""
        return self._build_fig()

    def render_gif(self, path: str, fps: int = 10, dpi: int = 120) -> None:
        """
        Render an animated GIF: N2 (left) vs CEl tuner (right).

        Features:
          - 0.1 mm grid so absolute displacement is readable
          - Growing head trajectory trail
          - 48 muscle attachment dots (24D + 24V)
          - Speed + t annotation per frame
          - 0.1 mm scale bar
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from matplotlib.lines import Line2D
        import PIL.Image, io

        if not hasattr(self, "_frame_cel"):
            self.tune()

        cfg = self.cfg
        BL  = cfg.bl_mm
        r   = cfg.r_bwm
        ns  = cfg.n_segments
        N   = len(self.t_out)

        xlim_mm = 0.32 * BL
        ylim_mm = 0.65 * BL

        def _mpos_flat(xb, yb):
            sn = np.linspace(0, 1, ns + 1); s0 = np.linspace(0, 1, cfg.n_skeleton_pts)
            xi = np.interp(sn, s0, xb); yi = np.interp(sn, s0, yb)
            xm = 0.5*(xi[:-1]+xi[1:]); ym = 0.5*(yi[:-1]+yi[1:])
            tx = xi[1:]-xi[:-1]; ty = yi[1:]-yi[:-1]
            L  = np.hypot(tx, ty)+1e-12; tx /= L; ty /= L
            return (xm-ty*r)*BL, (ym+tx*r)*BL, (xm+ty*r)*BL, (ym-tx*r)*BL

        BG   = "#0b0f16"
        GC   = "#1a2a3a"   # grid colour
        C_N2 = "#3ef07e"
        C_CL = "#ff9f1c"
        C_TR = "#ffffff"   # trail

        def _head_speed_mm(x, y, t):
            dt = np.diff(t); dx = np.diff(x[:,0]); dy = np.diff(y[:,0])
            return float(np.hypot(dx, dy)[0] / dt[0] * BL) if len(dt) else 0.0

        images = []
        trail_n2_x, trail_n2_y = [], []
        trail_cl_x, trail_cl_y = [], []

        for i in range(N):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5.2),
                                     facecolor=BG, constrained_layout=True)
            t_s = self.t_out[i]

            n2x = self.xN2c[i] * BL;  n2y = self.yN2c[i] * BL
            clx = self.xSIM[i]  * BL;  cly = self.ySIM[i]  * BL
            xd_n2, yd_n2, xv_n2, yv_n2 = _mpos_flat(self.xN2c[i], self.yN2c[i])
            xd_cl, yd_cl, xv_cl, yv_cl = _mpos_flat(self.xSIM[i],  self.ySIM[i])

            trail_n2_x.append(n2x[0]); trail_n2_y.append(n2y[0])
            trail_cl_x.append(clx[0]); trail_cl_y.append(cly[0])

            spd_n2 = np.hypot(np.diff(self.xN2c[:i+2,0]), np.diff(self.yN2c[:i+2,0]))
            spd_n2 = float(spd_n2[-1] / (self.t_out[1]-self.t_out[0]) * BL * 1000) if i>0 else 0.0
            spd_cl = np.hypot(np.diff(self.xSIM[:i+2,0]),  np.diff(self.ySIM[:i+2,0]))
            spd_cl = float(spd_cl[-1] / (self.t_out[1]-self.t_out[0]) * BL * 1000) if i>0 else 0.0

            for ax, bx, by, mx_d, my_d, mx_v, my_v, txl, tyl, col, label in [
                (axes[0], n2x, n2y, xd_n2, yd_n2, xv_n2, yv_n2,
                 trail_n2_x, trail_n2_y, C_N2,
                 f"N2 wild-type  {spd_n2:.1f} µm/s"),
                (axes[1], clx, cly, xd_cl, yd_cl, xv_cl, yv_cl,
                 trail_cl_x, trail_cl_y, C_CL,
                 f"NeuromuscularTuner  {spd_cl:.1f} µm/s"),
            ]:
                ax.set_facecolor(BG)
                ax.set_xlim(-xlim_mm, xlim_mm)
                ax.set_ylim(-ylim_mm, ylim_mm)

                # grid
                ax.set_xticks(np.arange(-0.3, 0.31, 0.1))
                ax.set_yticks(np.arange(-0.7, 0.71, 0.1))
                ax.grid(True, color=GC, linewidth=0.6, zorder=0)
                ax.axhline(0, color="#2e4a60", linewidth=0.8, zorder=1)
                ax.axvline(0, color="#2e4a60", linewidth=0.8, zorder=1)
                ax.tick_params(colors="#5a7a9a", labelsize=7)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#2a3a4a")

                # head trail
                ax.plot(txl, tyl, color=C_TR, lw=0.8, alpha=0.35,
                        linestyle=":", zorder=2)

                # muscle dots
                ax.scatter(mx_d, my_d, s=5, color=col, alpha=0.55,
                           zorder=3, linewidths=0)
                ax.scatter(mx_v, my_v, s=5, color=col, alpha=0.55,
                           zorder=3, linewidths=0)

                # body
                ax.plot(bx, by, color=col, lw=2.2, zorder=4)

                # head dot
                ax.scatter([bx[0]], [by[0]], s=40, color="white",
                           zorder=5, linewidths=0)

                # scale bar (0.1 mm)
                sb_x = xlim_mm * 0.62; sb_y = -ylim_mm * 0.88
                ax.plot([sb_x, sb_x + 0.1], [sb_y, sb_y],
                        color="#aabbcc", lw=2, zorder=6)
                ax.text(sb_x + 0.05, sb_y + 0.02, "0.1 mm",
                        ha="center", va="bottom", fontsize=6.5,
                        color="#aabbcc")

                ax.set_xlabel("x (mm)", color="#5a7a9a", fontsize=8)
                ax.set_ylabel("y (mm)", color="#5a7a9a", fontsize=8)
                ax.set_title(label, color=col, fontsize=9, pad=4)

            fig.suptitle(
                f"wormsim2 v0.8.2 NeuromuscularTuner  |  t = {t_s:.2f} s  "
                f"|  CEl₄₈ = {self._frame_cel[i]*1000:.3f} ×10⁻³ BL²",
                color="#b0c0d0", fontsize=9, y=1.01,
            )

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, facecolor=BG,
                        bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            images.append(PIL.Image.open(buf).copy())

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        dur = int(1000 / fps)
        images[0].save(
            path, save_all=True, append_images=images[1:],
            loop=0, duration=dur, optimize=False,
        )

    def render_html(self, path: str) -> None:
        """Write a self-contained interactive Plotly HTML file (CDN Plotly, no server needed)."""
        fig = self._build_fig()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(path, include_plotlyjs="cdn", full_html=True)

    # ── 3D space-time visualisations ─────────────────────────────────────────

    def _space_time_arrays(self):
        """Return (x_n2, y_n2, t_n2, x_cel, y_cel, t_cel) all in mm / s for 3D plots."""
        if not hasattr(self, "_frame_cel"):
            self.tune()
        BL = self.cfg.bl_mm
        t  = self.t_out
        return (
            self.xN2c * BL, self.yN2c * BL, t,
            self.xSIM  * BL, self.ySIM  * BL, t,
        )

    def render_fig_3d(self):
        """Return the 3D Plotly Figure for inline display in JupyterLab."""
        import plotly.graph_objects as go
        # build fig via write_html capture trick
        _saved = {}
        orig = go.Figure.write_html
        def _cap(self_, p, **kw): _saved['fig'] = self_; orig(self_, p, **kw)
        go.Figure.write_html = _cap
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            tmp = f.name
        try:
            self.render_html_3d(tmp)
        finally:
            go.Figure.write_html = orig
            os.unlink(tmp)
        return _saved['fig']

    def render_html_3d(self, path: str) -> None:
        """
        Interactive 3D space-time animation (Plotly Scatter3d, CDN).

        Axes:  X = lateral (mm),  Y = forward (mm),  Z = time (s).
        Shows N2 (green) and CEl tuner (orange) as animated body ribbons
        with growing head helix and per-frame speed colouring.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if not hasattr(self, "_frame_cel"):
            self.tune()

        cfg = self.cfg
        BL  = cfg.bl_mm
        N   = len(self.t_out)
        ns  = cfg.n_skeleton_pts

        xN, yN, t, xC, yC, _ = self._space_time_arrays()

        DARK  = "#0b0f16"
        C_N2  = "#3ef07e"
        C_CEL = "#ff9f1c"

        def _speed_color(x, y, t_arr, color_hex):
            dt = np.diff(t_arr); dx = np.diff(x[:, 0]); dy = np.diff(y[:, 0])
            spd = np.hypot(dx, dy) / dt * 1000  # µm/s
            spd = np.r_[spd[0], spd]
            # map speed to alpha-modified hex not easily done; use colorscale instead
            return spd

        spd_n2  = _speed_color(xN, yN, t, C_N2)
        spd_cel = _speed_color(xC, yC, t, C_CEL)

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "scene"}, {"type": "scene"}]],
            subplot_titles=["N2 wild-type (Zenodo 1031837)",
                            "v0.9 NeuromuscularTuner (4-mode)"],
            horizontal_spacing=0.02,
        )

        # full head helix (static background)
        fig.add_trace(go.Scatter3d(
            x=xN[:, 0].tolist(), y=yN[:, 0].tolist(), z=t.tolist(),
            mode="lines",
            line=dict(color=spd_n2.tolist(), colorscale="Greens",
                      width=3, cmin=float(spd_n2.min()), cmax=float(spd_n2.max())),
            name="N2 head path", opacity=0.3, showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter3d(
            x=xC[:, 0].tolist(), y=yC[:, 0].tolist(), z=t.tolist(),
            mode="lines",
            line=dict(color=spd_cel.tolist(), colorscale="Oranges",
                      width=3, cmin=float(spd_cel.min()), cmax=float(spd_cel.max())),
            name="CEl head path", opacity=0.3, showlegend=False,
        ), row=1, col=2)

        # animated body ribbon (initial frame)
        i0 = 0
        fig.add_trace(go.Scatter3d(
            x=xN[i0].tolist(), y=yN[i0].tolist(),
            z=[t[i0]] * ns,
            mode="lines+markers",
            line=dict(color=C_N2, width=5),
            marker=dict(size=2, color=C_N2),
            name="N2 body", showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter3d(
            x=xC[i0].tolist(), y=yC[i0].tolist(),
            z=[t[i0]] * ns,
            mode="lines+markers",
            line=dict(color=C_CEL, width=5),
            marker=dict(size=2, color=C_CEL),
            name="CEl body", showlegend=False,
        ), row=1, col=2)

        # growing head helix (animated)
        fig.add_trace(go.Scatter3d(
            x=xN[:1, 0].tolist(), y=yN[:1, 0].tolist(), z=t[:1].tolist(),
            mode="lines", line=dict(color=C_N2, width=2),
            name="N2 trail", showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter3d(
            x=xC[:1, 0].tolist(), y=yC[:1, 0].tolist(), z=t[:1].tolist(),
            mode="lines", line=dict(color=C_CEL, width=2),
            name="CEl trail", showlegend=False,
        ), row=1, col=2)

        # frames
        frames = []
        for i in range(N):
            frames.append(go.Frame(
                data=[
                    go.Scatter3d(x=xN[i].tolist(), y=yN[i].tolist(),
                                 z=[t[i]] * ns),
                    go.Scatter3d(x=xC[i].tolist(), y=yC[i].tolist(),
                                 z=[t[i]] * ns),
                    go.Scatter3d(x=xN[:i+1, 0].tolist(), y=yN[:i+1, 0].tolist(),
                                 z=t[:i+1].tolist()),
                    go.Scatter3d(x=xC[:i+1, 0].tolist(), y=yC[:i+1, 0].tolist(),
                                 z=t[:i+1].tolist()),
                ],
                traces=[2, 3, 4, 5],
                name=str(i),
                layout=go.Layout(
                    title_text=(
                        f"t = {t[i]:.2f} s  |  "
                        f"N2 head speed = {spd_n2[i]:.1f} µm/s  |  "
                        f"CEl head speed = {spd_cel[i]:.1f} µm/s  |  "
                        f"CEl₄₈ = {self._frame_cel[i]*1000:.3f} ×10⁻³ BL²"
                    )
                ),
            ))
        fig.frames = frames

        xlim = 0.32 * BL; ylim = 0.65 * BL
        scene = dict(
            xaxis=dict(title="x — lateral (mm)", range=[-xlim, xlim],
                       gridcolor="#1e2d3f", backgroundcolor=DARK,
                       dtick=0.1),
            yaxis=dict(title="y — forward (mm)", range=[-ylim, ylim],
                       gridcolor="#1e2d3f", backgroundcolor=DARK,
                       dtick=0.1),
            zaxis=dict(title="time (s)", range=[0, float(t[-1])],
                       gridcolor="#1e2d3f", backgroundcolor=DARK),
            bgcolor=DARK,
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2)),
        )

        fig.update_layout(
            paper_bgcolor=DARK,
            font=dict(color="#b0c0d0", size=11),
            title=dict(
                text=(f"wormsim2 v0.9 — 3D space-time (x=lateral, y=forward, z=time)  |  "
                      f"crawling scenario  |  CEl₄₈={self._frame_cel.mean():.6f} BL²"),
                font=dict(size=12),
            ),
            scene=scene, scene2=scene,
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=0.1, x=1.05, xanchor="left",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, {"frame": {"duration": 100, "redraw": True},
                                      "fromcurrent": True}]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
                ],
            )],
            sliders=[dict(
                currentvalue=dict(prefix="frame: ", font=dict(size=10)),
                pad=dict(t=10),
                steps=[dict(method="animate",
                            args=[[str(i)], {"frame": {"duration": 0, "redraw": True},
                                             "mode": "immediate"}],
                            label=f"{t[i]:.2f}s") for i in range(N)],
            )],
            height=650,
        )

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(path, include_plotlyjs="cdn", full_html=True)

    def render_gif_3d(self, path: str, fps: int = 10, dpi: int = 120,
                      elev: float = 25.0, azim_start: float = -60.0,
                      azim_sweep: float = 0.0) -> None:
        """
        Animated GIF: 3D space-time view (x=lateral, y=forward, z=time).

        azim_sweep > 0 slowly rotates the camera while animating.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        import PIL.Image, io

        if not hasattr(self, "_frame_cel"):
            self.tune()

        BL  = self.cfg.bl_mm
        N   = len(self.t_out)
        ns  = self.cfg.n_skeleton_pts
        xN, yN, t, xC, yC, _ = self._space_time_arrays()

        # head speeds for colour
        spd_n2 = np.hypot(np.diff(xN[:, 0]), np.diff(yN[:, 0])) / np.diff(t) * 1000
        spd_n2 = np.r_[spd_n2[0], spd_n2]
        spd_cl = np.hypot(np.diff(xC[:, 0]), np.diff(yC[:, 0])) / np.diff(t) * 1000
        spd_cl = np.r_[spd_cl[0], spd_cl]

        BG    = "#0b0f16"
        C_N2  = "#3ef07e"
        C_CEL = "#ff9f1c"
        GC    = "#1a2a3a"

        xlim = 0.32 * BL; ylim = 0.65 * BL

        images = []
        for i in range(N):
            azim = azim_start + azim_sweep * i / max(N - 1, 1)
            fig = plt.figure(figsize=(11, 5.5), facecolor=BG)
            for col, (xs, ys, trail_x, trail_y, color, label, spd) in enumerate([
                (xN, yN, xN[:i+1, 0], yN[:i+1, 0], C_N2,
                 f"N2  {spd_n2[i]:.0f} µm/s", spd_n2),
                (xC, yC, xC[:i+1, 0], yC[:i+1, 0], C_CEL,
                 f"NeuromuscularTuner  {spd_cl[i]:.0f} µm/s", spd_cl),
            ]):
                ax = fig.add_subplot(1, 2, col + 1, projection="3d",
                                     facecolor=BG)
                ax.view_init(elev=elev, azim=azim)

                # ghost head helix (full trajectory, dim)
                ax.plot(xs[:, 0], ys[:, 0], t, color=color,
                        alpha=0.15, lw=1, linestyle="--")

                # growing head trail (solid)
                ax.plot(trail_x, trail_y, t[:i+1],
                        color=color, lw=1.5, alpha=0.7)

                # body ribbon at current frame
                ax.plot(xs[i], ys[i], [t[i]] * ns,
                        color=color, lw=3, zorder=5)

                # head dot
                ax.scatter([xs[i, 0]], [ys[i, 0]], [t[i]],
                           s=40, color="white", zorder=6)

                # scale indicator at bottom
                sx = xlim * 0.55; sy = -ylim * 0.9
                ax.plot([sx, sx + 0.1], [sy, sy], [0, 0],
                        color="#aabbcc", lw=2)
                ax.text(sx + 0.05, sy - 0.04, 0, "0.1 mm",
                        ha="center", fontsize=6.5, color="#aabbcc")

                # axes style
                for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                    pane.fill = True; pane.set_facecolor(BG)
                    pane.set_edgecolor(GC)
                ax.tick_params(colors="#5a7a9a", labelsize=6.5)
                ax.xaxis.label.set_color("#5a7a9a")
                ax.yaxis.label.set_color("#5a7a9a")
                ax.zaxis.label.set_color("#5a7a9a")
                ax.set_xlabel("x (mm)", labelpad=2)
                ax.set_ylabel("y (mm)", labelpad=2)
                ax.set_zlabel("t (s)",  labelpad=2)
                ax.set_xlim(-xlim, xlim)
                ax.set_ylim(-ylim, ylim)
                ax.set_zlim(0, float(t[-1]))
                ax.set_xticks(np.arange(-0.2, 0.21, 0.1))
                ax.set_yticks(np.arange(-0.4, 0.41, 0.2))
                ax.set_title(label, color=color, fontsize=9, pad=4)
                ax.grid(True, color=GC, linewidth=0.4)

            fig.suptitle(
                f"wormsim2 v0.9 — 3D space-time  |  t = {t[i]:.2f} s  "
                f"|  CEl₄₈ = {self._frame_cel[i]*1000:.3f} ×10⁻³ BL²",
                color="#b0c0d0", fontsize=9, y=1.01,
            )

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, facecolor=BG,
                        bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            images.append(PIL.Image.open(buf).copy())

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        dur = int(1000 / fps)
        images[0].save(path, save_all=True, append_images=images[1:],
                       loop=0, duration=dur, optimize=False)

    def summary(self) -> str:
        r = getattr(self, "_results", None) or self.tune()
        return (
            f"NeuromuscularTuner | modes={r['n_modes']} "
            f"R²={r['var_explained']:.3f} ({r['var_explained']*100:.1f}%) | "
            f"CEl₄₈={r['cel_mean_bl2']:.6f} BL²  RMS={r['rms_um']:.1f} µm"
        )

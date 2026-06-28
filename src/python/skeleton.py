"""wormsim2 canonical skeleton utilities (v0.11.0).

Angle convention — theta_body
------------------------------
All angle arrays in wormsim2 use the **theta_body** convention, defined as:

    theta_body[j] = arctan2(y[j+1] - y[j], x[j+1] - x[j]) + π/2

where (x, y) are skeleton keypoint positions in body-lengths (BL) after
aligning the worm's tail→head axis to +y.

Key values:
    worm pointing +y (crawling up)  → theta_body ≈ π/2 + π/2 = π
    worm pointing +x (crawling right) → theta_body ≈ 0   + π/2 = π/2
    worm pointing -y                → theta_body ≈ −π/2 + π/2 = 0

Forward kinematics (theta_body → skeleton positions):

    th = theta_body − π/2          # lab heading (arctan2 of segment vector)
    x[j+1] = x[j] + ds * cos(th[j])
    y[j+1] = y[j] + ds * sin(th[j])

This is the ONLY correct formula for this convention.  There is NO cumsum
of angles — the cumsum is only over position *steps* (ds·cos, ds·sin).

Using cumsum on the angles (turning-angle convention) gives a worm that
spirals into circles; that is the bug that was in backends.py before v0.11.

Public API
----------
    xy_to_tangent(x, y)                    → theta_body  (1 frame)
    tangent_to_xy(theta_body, ds, mid_idx) → (x, y)      (1 frame)
    tangent_to_xy_batch(theta_seq, ds, mid_idx)
                                           → (xs, ys)    (n_frames batch)
    load_skeleton_csv(path, ...)           → SkeletonData namedtuple
    visualize_skeleton_csv(path, ...)      → plotly.graph_objects.Figure
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Core forward / inverse kinematics
# ---------------------------------------------------------------------------


def xy_to_tangent(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Single frame: (x, y) skeleton → theta_body angles.

    Args:
        x: (n_pts,) x-positions of skeleton keypoints (any unit)
        y: (n_pts,) y-positions

    Returns:
        theta_body: (n_pts − 1,) tangent angles in the wormsim2 convention
    """
    return np.arctan2(np.diff(y), np.diff(x)) + np.pi / 2


def tangent_to_xy(
    theta_body: np.ndarray,
    ds: float,
    mid_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Single frame: theta_body → (x, y) skeleton positions in BL.

    This is the canonical forward-kinematics formula.  See module docstring
    for the convention.

    Args:
        theta_body: (n_seg,) tangent angles (radians, theta_body convention)
        ds:         arc-length per segment in BL  (= 1 / n_seg)
        mid_idx:    if given, shift so that (x[mid_idx], y[mid_idx]) == (0, 0)

    Returns:
        x, y: (n_pts,) arrays where n_pts = n_seg + 1
    """
    th = theta_body - np.pi / 2          # convert to lab heading
    x = np.r_[0.0, np.cumsum(np.cos(th) * ds)]   # cumsum of *steps*
    y = np.r_[0.0, np.cumsum(np.sin(th) * ds)]
    if mid_idx is not None:
        x -= x[mid_idx]
        y -= y[mid_idx]
    return x, y


def tangent_to_xy_batch(
    theta_seq: np.ndarray,
    ds: float,
    mid_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch version of tangent_to_xy — no Python loop over frames.

    Args:
        theta_seq: (n_frames, n_seg) tangent angles (theta_body convention)
        ds:        arc-length per segment in BL
        mid_idx:   if given, shift each frame so midpoint is at origin

    Returns:
        xs, ys: (n_frames, n_pts) float32 arrays
    """
    n_fr, n_seg = theta_seq.shape
    th = theta_seq - np.pi / 2                         # (F, S) lab headings
    cos_th = np.cos(th).astype(np.float32)
    sin_th = np.sin(th).astype(np.float32)
    zeros  = np.zeros((n_fr, 1), dtype=np.float32)
    xs = np.cumsum(np.concatenate([zeros, cos_th * ds], axis=1), axis=1)
    ys = np.cumsum(np.concatenate([zeros, sin_th * ds], axis=1), axis=1)
    if mid_idx is not None:
        xs -= xs[:, mid_idx : mid_idx + 1]
        ys -= ys[:, mid_idx : mid_idx + 1]
    return xs, ys


# ---------------------------------------------------------------------------
# CSV data loading
# ---------------------------------------------------------------------------


@dataclass
class SkeletonData:
    """Output of load_skeleton_csv."""
    t:          np.ndarray   # (N,) time in seconds
    x_bl:       np.ndarray   # (N, n_pts) x-positions in BL, centred at mid_idx
    y_bl:       np.ndarray   # (N, n_pts) y-positions in BL, centred at mid_idx
    theta_body: np.ndarray   # (N, n_seg) tangent angles
    fps:        float        # estimated frames-per-second
    n_pts:      int
    mid_idx:    int
    path:       str

    @property
    def ds(self) -> float:
        return 1.0 / (self.n_pts - 1)

    def roundtrip_error(self) -> float:
        """Max |x_recon − x_bl| over all frames.

        Expected values:
        - FK-generated skeletons (uniform segment lengths): < 1e-5 BL
        - Real CSV data (non-uniform segments, mean ≈ ds, spread ±20%): 0.01–0.10 BL

        The tangent-angle representation captures segment *direction* but not
        *length*.  The canonical FK uses uniform ds = 1/(n_pts−1), so the
        reconstruction will differ from the raw data when segments are not
        uniformly spaced.  This is expected and is not a bug.
        """
        xr, yr = tangent_to_xy_batch(self.theta_body, self.ds, self.mid_idx)
        return float(np.max(np.abs(xr.astype(float) - self.x_bl)))


def load_skeleton_csv(
    path: str | Path,
    n_pts: int = 49,
    bl_mm: float = 1.0,
    mid_idx: int | None = None,
    align_tail_to_y: bool = True,
    t_start: float = 0.0,
    t_end: float = float("inf"),
) -> SkeletonData:
    """Read a worm-skeleton CSV and return aligned, BL-normalised data.

    CSV format (same as docs/images/v080_real_n2_skeleton.csv):
        t_s,  x0, x1, ..., x(n_pts−1),  y0, y1, ..., y(n_pts−1)
    Positions are in mm; time in seconds.
    Column 0 is HEAD, column n_pts−1 is TAIL (as in the Zenodo dataset).

    Args:
        path:            CSV file path
        n_pts:           number of skeleton keypoints (default 49 → 48 segments)
        bl_mm:           body-length in mm for normalisation (default 1.0 → no scaling)
        mid_idx:         index of midpoint (default: n_pts // 2)
        align_tail_to_y: if True, rotate each clip so the worm's mean
                         tail→head axis is aligned to +y
        t_start, t_end:  time window (relative to first frame)

    Returns:
        SkeletonData with fields: t, x_bl, y_bl, theta_body, fps, n_pts, mid_idx, path
    """
    import pandas as pd

    if mid_idx is None:
        mid_idx = n_pts // 2

    df = pd.read_csv(path)
    t_col = "t_s" if "t_s" in df.columns else df.columns[0]
    t_raw  = df[t_col].values
    x_raw  = df[[f"x{j}" for j in range(n_pts)]].values / bl_mm   # BL
    y_raw  = df[[f"y{j}" for j in range(n_pts)]].values / bl_mm

    # time window
    t_rel = t_raw - t_raw[0]
    mask  = (t_rel >= t_start) & (t_rel <= t_end)
    t_raw, x_raw, y_raw = t_raw[mask], x_raw[mask], y_raw[mask]
    N = len(t_raw)
    t = np.linspace(0.0, t_raw[-1] - t_raw[0], N)

    if align_tail_to_y:
        # rotate so the mean tail→head vector (index 0 → index −1) points in +y
        dx0 = float(x_raw[0, 0] - x_raw[0, -1])
        dy0 = float(y_raw[0, 0] - y_raw[0, -1])
        rot  = np.pi / 2 - np.arctan2(dy0, dx0)
        ca, sa = np.cos(rot), np.sin(rot)
        x_rot  = ca * x_raw - sa * y_raw
        y_rot  = sa * x_raw + ca * y_raw
    else:
        x_rot, y_rot = x_raw, y_raw

    # centre at midpoint
    x_bl = x_rot - x_rot[:, mid_idx : mid_idx + 1]
    y_bl = y_rot - y_rot[:, mid_idx : mid_idx + 1]

    # tangent angles
    theta_body = np.array([
        xy_to_tangent(x_bl[i], y_bl[i]) for i in range(N)
    ])

    fps = float(N - 1) / float(t[-1]) if t[-1] > 0 else 1.0

    return SkeletonData(
        t=t, x_bl=x_bl, y_bl=y_bl,
        theta_body=theta_body,
        fps=fps, n_pts=n_pts, mid_idx=mid_idx,
        path=str(path),
    )


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def visualize_skeleton_csv(
    path: str | Path,
    n_pts: int = 49,
    bl_mm: float = 1.0,
    title: str | None = None,
    fps: float | None = None,
    show_reconstructed: bool = False,
    n_frames: int | None = None,
) -> "go.Figure":
    """Animate a skeleton CSV as a Plotly figure.

    Reads the CSV using the canonical pipeline (load_skeleton_csv → tangent_to_xy),
    so the worm is always shown correctly regardless of the raw coordinate system.

    Args:
        path:               CSV file path
        n_pts:              skeleton keypoints per frame
        bl_mm:              body-length mm for normalisation
        title:              figure title (default: filename)
        fps:                animation frames-per-second (default: inferred from CSV)
        show_reconstructed: if True, overlay the FK-reconstructed skeleton to
                            verify round-trip fidelity
        n_frames:           cap at this many animation frames (None = all)

    Returns:
        go.Figure (Plotly animated scatter)
    """
    import plotly.graph_objects as go

    sk = load_skeleton_csv(path, n_pts=n_pts, bl_mm=bl_mm)
    err = sk.roundtrip_error()

    n_fr = len(sk.t) if n_frames is None else min(int(n_frames), len(sk.t))
    _fps = fps or sk.fps
    frame_dur = int(1000 / _fps)

    traces = [go.Scatter(x=sk.x_bl[0], y=sk.y_bl[0],
                         mode="lines+markers", name="real (raw)",
                         line=dict(color="#00CC96", width=3),
                         marker=dict(size=4))]
    if show_reconstructed:
        xr, yr = tangent_to_xy_batch(sk.theta_body, sk.ds, sk.mid_idx)
        traces.append(go.Scatter(x=xr[0], y=yr[0],
                                 mode="lines", name="FK reconstructed",
                                 line=dict(color="#EF553B", width=2, dash="dash")))

    frames = []
    for i in range(n_fr):
        frame_traces = [go.Scatter(x=sk.x_bl[i], y=sk.y_bl[i])]
        if show_reconstructed:
            xr, yr = tangent_to_xy_batch(sk.theta_body, sk.ds, sk.mid_idx)
            frame_traces.append(go.Scatter(x=xr[i], y=yr[i]))
        frames.append(go.Frame(data=frame_traces, name=str(i)))

    name = title or Path(path).stem
    fig = go.Figure(
        data=traces,
        frames=frames,
        layout=go.Layout(
            title=dict(text=f"{name}  (FK round-trip error: {err:.2e} BL)"),
            xaxis=dict(title="x (BL)", range=[-0.6, 0.6], scaleanchor="y"),
            yaxis=dict(title="y (BL)", range=[-0.6, 0.6]),
            updatemenus=[dict(type="buttons",
                              buttons=[dict(label="▶", method="animate",
                                           args=[None, dict(frame=dict(duration=frame_dur),
                                                            fromcurrent=True)])])],
            sliders=[dict(steps=[dict(args=[[f.name], dict(frame=dict(duration=0),
                                            mode="immediate")],
                                     method="animate") for f in frames],
                          currentvalue=dict(prefix="frame: "))],
            annotations=[dict(text=f"n_pts={n_pts}  |  {n_fr} frames  |  {_fps:.1f} fps  "
                                   f"|  FK error {err:.1e} BL",
                              xref="paper", yref="paper", x=0.01, y=-0.08,
                              showarrow=False, font=dict(size=11))],
        ),
    )
    return fig

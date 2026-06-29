"""
vtu_export.py — VTK / ParaView / PyVista export for wormsim2 worm trajectories.

Produces:
  <out_dir>/
    <name>.pvd            ← PVD time-series collection (open in ParaView)
    frames/
      frame_NNNN.vtu      ← one VTU per frame (UnstructuredGrid, LINE cells)
  <name>_trajectory.zip  ← ready-to-download archive

No VTK library required — XML is written directly.

Usage:
    from vtu_export import export_trajectory
    export_trajectory(x_real, y_real, x_sim, y_sim, t_s,
                      name="n2_wt_600s", out_dir="docs/gui/downloads")
"""
from __future__ import annotations
import io, pathlib, zipfile
import numpy as np


# ── single-frame VTU ────────────────────────────────────────────────────────

def _frame_vtu(x_real: np.ndarray, y_real: np.ndarray,
               x_sim:  np.ndarray, y_sim:  np.ndarray,
               t_s: float) -> str:
    """
    Return VTU XML string for one animation frame.

    Two worm polylines (real + simulated) as VTK_LINE cells (type 3).
    PointData fields:
      worm_id       — 0 = real, 1 = simulated
      keypoint_idx  — 0 (head) … 48 (tail)
    """
    n = x_real.shape[0]          # 49 keypoints
    px = np.concatenate([x_real, x_sim])
    py = np.concatenate([y_real, y_sim])
    pz = np.zeros(2 * n)

    n_cells = 2 * (n - 1)       # 48 segments × 2 worms = 96 cells
    conn = []
    offs = []
    for w in range(2):
        base = w * n
        for j in range(n - 1):
            conn.extend([base + j, base + j + 1])
            offs.append(len(conn))

    def arr(vals, fmt=".6f"):
        return " ".join(format(v, fmt) for v in vals)

    pts_block = "\n          ".join(
        f"{px[i]:.6f} {py[i]:.6f} {pz[i]:.6f}" for i in range(2 * n)
    )
    worm_id  = [0] * n + [1] * n
    kp_idx   = list(range(n)) + list(range(n))

    return f"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <FieldData>
      <DataArray type="Float64" Name="TimeValue" NumberOfTuples="1" format="ascii">
        {t_s:.6f}
      </DataArray>
    </FieldData>
    <Piece NumberOfPoints="{2 * n}" NumberOfCells="{n_cells}">
      <Points>
        <DataArray type="Float32" NumberOfComponents="3" format="ascii">
          {pts_block}
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">
          {arr(conn, 'd')}
        </DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">
          {arr(offs, 'd')}
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">
          {' '.join(['3'] * n_cells)}
        </DataArray>
      </Cells>
      <PointData>
        <DataArray type="Int32" Name="worm_id" format="ascii">
          {arr(worm_id, 'd')}
        </DataArray>
        <DataArray type="Int32" Name="keypoint_index" format="ascii">
          {arr(kp_idx, 'd')}
        </DataArray>
      </PointData>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""


# ── PVD collection ────────────────────────────────────────────────────────────

def _make_pvd(n_frames: int, name: str, t_values: np.ndarray) -> str:
    entries = "\n    ".join(
        f'<DataSet timestep="{t_values[i]:.6f}" group="" part="0" '
        f'file="frames/frame_{i:04d}.vtu"/>'
        for i in range(n_frames)
    )
    return f"""<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
  <Collection>
    {entries}
  </Collection>
</VTKFile>
"""


# ── README ───────────────────────────────────────────────────────────────────

_README_TMPL = """wormsim2 — VTK trajectory export
================================
Scenario : {name}
Frames   : {n_frames}
Time     : {t0:.1f} – {t1:.1f} s
Coords   : x, y in BL (body lengths); z = 0
Fields   : worm_id (0=real, 1=simulated), keypoint_index (0=head … 48=tail)

ParaView
--------
  File > Open > {name}.pvd
  Apply  →  Play (▶)
  Colour by worm_id to distinguish real (blue) from simulated (red).

PyVista (Python)
----------------
  import pyvista as pv
  reader = pv.PVDReader("{name}.pvd")
  print(reader.time_values)          # list of timestamps

  pl = pv.Plotter()
  reader.set_active_time_point(0)
  mesh = reader.read()
  pl.add_mesh(mesh.threshold(0.5, scalars="worm_id"),
              color="steelblue", line_width=4, label="real")
  pl.add_mesh(mesh.threshold(0.5, scalars="worm_id", invert=True),
              color="crimson",   line_width=4, label="simulated")
  pl.show_axes(); pl.add_legend(); pl.show()

  # Animate all frames:
  for i, t in enumerate(reader.time_values):
      reader.set_active_time_point(i)
      mesh = reader.read()
      # ... process mesh ...

Source
------
  https://github.com/VahidGh/wormsim2
  Data: Zenodo 1031837 (Schafer Lab, C. elegans N2 WT 2014)
"""


# ── public API ────────────────────────────────────────────────────────────────

def export_trajectory(
    x_real: np.ndarray,      # (N, 49) real skeleton in BL, head=idx0
    y_real: np.ndarray,
    x_sim:  np.ndarray,      # (N, 49) FK/simulated skeleton in BL
    y_sim:  np.ndarray,
    t_s:    np.ndarray,      # (N,) time in seconds
    name:   str = "trajectory",
    out_dir: str = "docs/gui/downloads",
    zip_only: bool = True,
) -> pathlib.Path:
    """
    Export N frames to VTU + PVD and pack into <out_dir>/<name>_vtu.zip.

    Returns the path of the created ZIP file.
    """
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    zip_path = out / f"{name}_vtu.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        n_frames = len(t_s)

        # README
        readme = _README_TMPL.format(
            name=name, n_frames=n_frames,
            t0=float(t_s[0]), t1=float(t_s[-1]),
        )
        zf.writestr(f"{name}/README.txt", readme)

        # PVD
        pvd = _make_pvd(n_frames, name, t_s)
        zf.writestr(f"{name}/{name}.pvd", pvd)

        # Per-frame VTU
        for i in range(n_frames):
            vtu = _frame_vtu(x_real[i], y_real[i], x_sim[i], y_sim[i], float(t_s[i]))
            zf.writestr(f"{name}/frames/frame_{i:04d}.vtu", vtu)

    size_kb = zip_path.stat().st_size // 1024
    print(f"VTU ZIP → {zip_path}  ({n_frames} frames, {size_kb} KB)")
    return zip_path

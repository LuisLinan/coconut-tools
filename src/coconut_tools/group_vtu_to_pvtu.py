from __future__ import annotations

import os
import re
import glob
import numpy as np
import pyvista as pv

VTK_TYPE_MAP = {
    np.dtype("float32"): "Float32",
    np.dtype("float64"): "Float64",
    np.dtype("int32"):   "Int32",
    np.dtype("int64"):   "Int64",
    np.dtype("uint32"):  "UInt32",
    np.dtype("uint64"):  "UInt64",
}

def _vtk_type(arr: np.ndarray) -> str:
    dt = np.asarray(arr).dtype
    return VTK_TYPE_MAP.get(dt, "Float64")

def _ncomp(arr: np.ndarray) -> int:
    a = np.asarray(arr)
    return 1 if a.ndim == 1 else int(a.shape[1])

def write_pvtu_from_pieces(piece_files: list[str], out_pvtu: str) -> str:
    if not piece_files:
        raise ValueError("piece_files is empty")

    m0 = pv.read(piece_files[0])
    
    pts_dtype = _vtk_type(m0.points)
    
    p_arrays = []
    for k in m0.point_data.keys():
        a = np.asarray(m0.point_data[k])
        p_arrays.append((k, _vtk_type(a), _ncomp(a)))
        
    c_arrays = []
    for k in m0.cell_data.keys():
        a = np.asarray(m0.cell_data[k])
        c_arrays.append((k, _vtk_type(a), _ncomp(a)))
        
    out_dir = os.path.dirname(os.path.abspath(out_pvtu))
    rel_sources = [os.path.relpath(os.path.abspath(f), out_dir) for f in piece_files]
    
    lines = []
    lines.append('<?xml version="1.0"?>')
    lines.append('<VTKFile type="PUnstructuredGrid" version="0.1" byte_order="LittleEndian">')
    lines.append('  <PUnstructuredGrid GhostLevel="0">')
    
    if p_arrays:
        lines.append('    <PPointData>')
        for name, vtype, nc in p_arrays:
            if nc == 1:
                lines.append(f'      <PDataArray type="{vtype}" Name="{name}"/>')
            else:
                lines.append(f'      <PDataArray type="{vtype}" Name="{name}" NumberOfComponents="{nc}"/>')
        lines.append('    </PPointData>')
    else:
        lines.append('    <PPointData/>')

    if c_arrays:
        lines.append('    <PCellData>')
        for name, vtype, nc in c_arrays:
            if nc == 1:
                lines.append(f'      <PDataArray type="{vtype}" Name="{name}"/>')
            else:
                lines.append(f'      <PDataArray type="{vtype}" Name="{name}" NumberOfComponents="{nc}"/>')
        lines.append('    </PCellData>')
    else:
        lines.append('    <PCellData/>')

    lines.append('    <PPoints>')
    lines.append(f'      <PDataArray type="{pts_dtype}" NumberOfComponents="3"/>')
    lines.append('    </PPoints>')

    for src in rel_sources:
        lines.append(f'    <Piece Source="{src}"/>')

    lines.append('  </PUnstructuredGrid>')
    lines.append('</VTKFile>')

    os.makedirs(out_dir, exist_ok=True)
    with open(out_pvtu, "w") as f:
        f.write("\n".join(lines))

    return out_pvtu

def write_pvtu_for_all_snapshots(directory: str = ".", prefix: str = "corona-flow", output_directory: str | None = None, max_pieces: int | None = None) -> list[str]:
    """
    Finds snapshots with pattern:
      {prefix}<snap>-P<rank>.vtu
    and writes:
      {prefix}<snap>.pvtu
    in the same directory,
    or in output_directory if != None.
    """
    directory = os.path.abspath(directory)

    if output_directory is None:
        output_directory = directory
    else:
        output_directory = os.path.abspath(output_directory)
    os.makedirs(output_directory, exist_ok=True)
    
    pat = os.path.join(directory, f"{prefix}*-P*.vtu")
    files = glob.glob(pat)
    if not files:
        raise FileNotFoundError(f"No files matching {pat}")

    # Group by snapshot id (the number after prefix, before -P)
    rx = re.compile(rf"^{re.escape(prefix)}(\d+)-P(\d+)\.vtu$")
    groups: dict[str, list[tuple[int, str]]] = {}

    for f in files:
        base = os.path.basename(f)
        m = rx.match(base)
        if not m:
            continue
        snap = m.group(1)
        rank = int(m.group(2))
        groups.setdefault(snap, []).append((rank, f))

    out_files = []
    for snap, lst in sorted(groups.items(), key=lambda kv: int(kv[0])):
        lst_sorted = [f for _, f in sorted(lst, key=lambda x: x[0])]
        if max_pieces is not None:
            lst_sorted = lst_sorted[:max_pieces]   # keep first N pieces
        out_pvtu = os.path.join(output_directory, f"{prefix}{snap}.pvtu")
        write_pvtu_from_pieces(lst_sorted, out_pvtu)
        out_files.append(out_pvtu)

    return out_files

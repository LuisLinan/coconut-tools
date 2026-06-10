"""
coconut_to_jhv_minimal.py

Minimal COCONUT/COOLFluid VTU -> JHelioviewer SunJSON exporter.

Workflow:
1. Read a VTU file with PyVista.
2. Build the magnetic vector field B from COCONUT arrays Bx, By, Bz.
3. Trace magnetic-field streamlines.
4. Show or save a PyVista preview of those lines.
5. Export the streamlines as SunJSON lines for JHelioviewer.

Assumption:
- JSON expected in coordinates = [radius, Carrington longitude, Carrington latitude] ; units = [Rsun, degree, degree]
- mesh.points are in Rsun, as in the usual COCONUT VTU output.
- x-y is the solar equatorial plane and z is solar north.
- longitude is atan2(y, x), with optional offset/sign correction.

Examples:
python3 coconut_to_jhv.py corona-mhd_0.vtu fieldlines.json --show
python3 coconut_to_jhv.py corona-mhd_0.vtu fieldlines.json --screenshot preview.png
python3 coconut_to_jhv.py input.vtu fieldlines.json \
  --longitude-offset-deg 180 \
  --flip-longitude \
  --show
  --progress

NB: you can also read COCONUT CFmesh files by passing a .CFmesh filename instead of .vtu. The script will detect the format and parse it accordingly, as long as the expected structure is present.
contact: Q.Noraz
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone

import numpy as np
import pyvista as pv
import time


RSUN_M = 6.955e8


def read_coconut_vtu(filename):
    """Read VTU and add the magnetic vector field needed by PyVista streamlines."""
    mesh = pv.read(filename)

    required = ["Bx", "By", "Bz"]
    missing = [
        name
        for name in required
        if name not in mesh.point_data and name not in mesh.cell_data
    ]
    if missing:
        raise KeyError(f"Missing magnetic-field arrays in VTU: {missing}")

    # Same dimensional scaling as your plotting script. The scaling does not affect
    # field-line geometry, but keeps B in physical units for consistency.
    if all(name in mesh.point_data for name in required):
        mesh.point_data["B"] = np.column_stack([
            mesh.point_data["Bx"] * 2.2,
            mesh.point_data["By"] * 2.2,
            mesh.point_data["Bz"] * 2.2,
        ])
    elif all(name in mesh.cell_data for name in required):
        mesh.cell_data["B"] = np.column_stack([
            mesh.cell_data["Bx"] * 2.2,
            mesh.cell_data["By"] * 2.2,
            mesh.cell_data["Bz"] * 2.2,
        ])
        mesh = mesh.cell_data_to_point_data(pass_cell_data=True)
    else:
        raise ValueError("Magnetic-field components must all be point data or all be cell data.")

    return mesh


def readstruct(lines):
    """Parse a CFmesh file structure and return section indices."""
    exp = re.compile(r'!NB_ELEM (-?\d+)')
    nbelements, idx0, idx1, idx2, idx3, nend = 0, 0, 0, 0, 0, 0
    comment = []

    for i, line in enumerate(lines):
        if line.startswith("!"):
            comment.append((i, line))
            if line.startswith("!LIST_NODE"):
                idx1 = i + 1
            elif line.startswith("!LIST_STATE 1"):
                idx2 = i + 1
            elif line.startswith("!LIST_ELEM"):
                idx0 = i + 1
            elif line.startswith("!NB_ELEM "):
                nbelements = int(exp.search(line).group(1))
            elif line.startswith("!TRS_NAME Outlet"):
                idx3 = i
            elif line.startswith("!END"):
                nend += 1

    return idx0, idx1, idx2, idx3, nbelements, nend, comment


def read_coconut_cfmesh(filename):
    """Read a CFmesh file and build a PyVista mesh with magnetic cell data."""
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"CFmesh file not found: {filename}")

    with open(filename, "r") as f:
        lines = f.readlines()

    idx0, idx1, idx2, idx3, nbelements, nend, comment = readstruct(lines)
    if nbelements <= 0 or idx0 <= 0 or idx1 <= 0 or idx2 <= 0:
        raise ValueError(f"CFmesh file missing required structure sections: {filename}")
    if nend <= 0:
        raise ValueError(f"CFmesh file missing !END markers: {filename}")

    connectivity = np.loadtxt(lines[idx0:idx0 + nbelements], dtype=int)
    if connectivity.ndim == 1:
        connectivity = connectivity.reshape(1, -1)

    coordinates = np.loadtxt(lines[idx1:idx2 - 1], dtype=float)

    nodes = connectivity[:, :6]
    n_cells = nodes.shape[0]

    # Build VTK cell connectivity for 6-node wedges.
    cells = np.empty(n_cells * 7, dtype=np.int64)
    cells[0::7] = 6
    cells[1::7] = nodes[:, 0]
    cells[2::7] = nodes[:, 1]
    cells[3::7] = nodes[:, 2]
    cells[4::7] = nodes[:, 3]
    cells[5::7] = nodes[:, 4]
    cells[6::7] = nodes[:, 5]

    cell_types = np.full(n_cells, 13, dtype=np.uint8)
    mesh = pv.UnstructuredGrid(cells, cell_types, coordinates)

    bd = comment[-nend - 1][0] + 1
    bf = comment[-nend][0]
    state_data = np.loadtxt(lines[bd:bf], dtype=np.float64)
    if state_data.ndim == 1:
        state_data = state_data.reshape(1, -1)
    if state_data.shape[0] != n_cells:
        raise ValueError(
            f"Expected {n_cells} state records but found {state_data.shape[0]} in CFmesh {filename}"
        )

    Bx = state_data[:, 4]
    By = state_data[:, 5]
    Bz = state_data[:, 6]

    mesh.cell_data["B"] = np.column_stack([Bx * 2.2e-4, By * 2.2e-4, Bz * 2.2e-4])
    mesh = mesh.cell_data_to_point_data(pass_cell_data=True)

    return mesh


def read_coconut_input(filename):
    """Read either a VTU or CFmesh input and return a PyVista mesh with B vectors."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in {".vtu", ".pvtu", ".vtk"}:
        return read_coconut_vtu(filename)
    if ext in {".cfmesh"}:
        return read_coconut_cfmesh(filename)
    raise ValueError(
        f"Unsupported input format for {filename}. Use .vtu or .CFmesh."
    )


def make_seed_grid(radius=1.05, n_points=500, lat_min=-80, lat_max=80, use_tqdm=False):
    """Create an approximately uniform longitude/latitude seed grid.

    If `use_tqdm` is True and the `tqdm` package is available, show a
    progress bar over latitude bands while building the seed points.
    """
    n_points = max(1, int(n_points))
    lat_span = max(1.0, float(lat_max) - float(lat_min))
    n_lats = max(2, int(round(np.sqrt(n_points * lat_span / 360.0))))
    n_lons = max(1, int(np.ceil(n_points / n_lats)))

    lons = np.linspace(0.0, 360.0, n_lons, endpoint=False)
    lats = np.linspace(lat_min, lat_max, n_lats)

    points = []

    lat_iter = lats
    if use_tqdm:
        try:
            from tqdm.auto import tqdm
            lat_iter = tqdm(lats, desc="Seed latitudes", unit="lat")
        except Exception:
            print("tqdm not available; continuing without progress bars")

    for lat in lat_iter:
        for lon in lons:
            lon_rad = np.radians(lon)
            lat_rad = np.radians(lat)

            x = radius * np.cos(lat_rad) * np.cos(lon_rad)
            y = radius * np.cos(lat_rad) * np.sin(lon_rad)
            z = radius * np.sin(lat_rad)

            points.append([x, y, z])

    return pv.PolyData(np.array(points))


def trace_fieldlines(mesh, n_seed_points=500, source_radius=1.05, max_steps=1000):
    """Return PyVista streamline PolyData. This object contains the line geometry."""
    return mesh.streamlines(
        vectors="B",
        n_points=int(n_seed_points),
        source_radius=float(source_radius),
        source_center=(0.0, 0.0, 0.0),
        max_steps=int(max_steps),
    )


def stream_to_lines(stream, max_lines=None, min_points=2, use_tqdm=False):
    """Convert a PyVista streamline object into a list of (N, 3) Cartesian arrays.

    If `use_tqdm` is True and `tqdm` is available, show progress while
    extracting lines from the internal VTK cell array.
    """
    lines = []
    points = np.asarray(stream.points)
    cells = np.asarray(stream.lines)

    # Optionally prepare a progress bar by counting polylines first.
    pbar = None
    if use_tqdm:
        try:
            total = 0
            j = 0
            while j < len(cells):
                n = int(cells[j])
                total += 1
                j += n + 1
            from tqdm.auto import tqdm
            pbar = tqdm(total=total, desc="Extracting lines", unit="line")
        except Exception:
            pbar = None
            print("tqdm not available; continuing without progress bars")

    i = 0
    while i < len(cells):
        n = int(cells[i])
        ids = cells[i + 1 : i + 1 + n]
        if n >= min_points:
            lines.append(points[ids])
            if pbar is not None:
                pbar.update(1)
            if max_lines is not None and len(lines) >= max_lines:
                break
        i += n + 1

    if pbar is not None:
        pbar.close()

    return lines


def xyz_to_sunjson_coordinates(xyz, lon_offset_deg=0.0, flip_longitude=False):
    """
    Cartesian Rsun coordinates -> [radius, Carrington longitude, latitude].
    Output units are [Rsun, degree, degree].
    """
    xyz = np.asarray(xyz, dtype=float)
    x, y, z = xyz.T

    r = np.sqrt(x*x + y*y + z*z)
    sign = -1.0 if flip_longitude else 1.0
    lon = sign * np.degrees(np.arctan2(y, x))
    lon = (lon + lon_offset_deg) % 360.0
    with np.errstate(divide="ignore", invalid="ignore"):
        lat = np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0)))
    lat = np.nan_to_num(lat)

    return np.column_stack([r, lon, lat])


def write_sunjson(lines_xyz, output_json, thickness=0.004,
                  color=(255, 255, 255, 255), lon_offset_deg=0.0,
                  flip_longitude=False, time=None, use_tqdm=False):
    """Write field lines to the simple SunJSON format read by JHelioviewer."""
    if time is None:
        time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    geometry = []
    line_iter = lines_xyz
    if use_tqdm:
        try:
            from tqdm.auto import tqdm
            line_iter = tqdm(lines_xyz, desc="Converting lines", unit="line")
        except Exception:
            print("tqdm not available; continuing without progress bars")

    for line_xyz in line_iter:
        coords = xyz_to_sunjson_coordinates(
            line_xyz,
            lon_offset_deg=lon_offset_deg,
            flip_longitude=flip_longitude,
        )

        geometry.append({
            "type": "line",
            "coordinates": coords.tolist(),
            # One color is enough: JHelioviewer repeats the last color.
            "colors": [list(color)],
            "thickness": float(thickness),
        })

    data = {
        "type": "SunJSON",
        "time": time,
        "geometry": geometry,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))

    n_points = sum(len(line) for line in lines_xyz)
    print(f"Wrote {len(lines_xyz)} field lines, {n_points} points -> {output_json}")


def preview(mesh, stream, show=False, screenshot=None, tube_radius=0.01):
    """Intermediate PyVista preview. The tube is only for display, not export."""
    if not show and screenshot is None:
        return

    plotter = pv.Plotter(off_screen=not show)

    # Try to show an inner sphere for orientation.
    plotter.add_mesh(pv.Sphere(radius=1.0), color="lightgray", opacity=0.35)

    if stream.n_points > 0:
        plotter.add_mesh(stream.tube(radius=tube_radius), color="white")
    else:
        print("Warning: no streamlines to preview.")

    plotter.add_axes()
    plotter.camera_position = [(10, 7, 4), (0, 0, 0), (0, 0, 1)]

    if screenshot is not None:
        plotter.show(interactive=False, auto_close=False)
        plotter.screenshot(screenshot)
        print(f"Saved preview -> {screenshot}")

    if show:
        plotter.show()

    plotter.close()


def export_to_jhv_json(input_file, output_json,
                           n_seed_points=500, source_radius=1.05, max_steps=1000,
                           max_lines=None, thickness=0.004,
                           lon_offset_deg=0.0, flip_longitude=False,
                           show=False, screenshot=None, use_tqdm=False):
    """Complete minimal pipeline with simple stage prints and timings."""
    t0 = time.perf_counter()
    print("Stage: reading input ->", input_file)
    mesh = read_coconut_input(input_file)
    print(f"Read VTU in {time.perf_counter() - t0:.2f}s")

    t1 = time.perf_counter()
    print("Stage: building seed grid")
    seed_source = make_seed_grid(radius=source_radius, n_points=n_seed_points, use_tqdm=use_tqdm)
    print(f"Built seed grid in {time.perf_counter() - t1:.2f}s")

    t2 = time.perf_counter()
    print("Stage: computing streamlines (this may take a while)")
    stream = mesh.streamlines_from_source(
        seed_source,
        vectors="B",
        max_steps=int(max_steps),
    )
    print(f"Computed streamlines in {time.perf_counter() - t2:.2f}s")

    #stream = trace_fieldlines(mesh, n_seed_points, source_radius, max_steps) #random
    t3 = time.perf_counter()
    print("Stage: extracting lines from streamlines")
    lines = stream_to_lines(stream, max_lines=max_lines, use_tqdm=use_tqdm)
    print(f"Extracted {len(lines)} lines in {time.perf_counter() - t3:.2f}s")

    t4 = time.perf_counter()
    print("Stage: preview (show/screenshot)")
    preview(mesh, stream, show=show, screenshot=screenshot)
    print(f"Preview done in {time.perf_counter() - t4:.2f}s")

    t5 = time.perf_counter()
    print("Stage: writing SunJSON ->", output_json)
    write_sunjson(
        lines,
        output_json,
        thickness=thickness,
        lon_offset_deg=lon_offset_deg,
        flip_longitude=flip_longitude,
        use_tqdm=use_tqdm,
    )
    print(f"Wrote SunJSON in {time.perf_counter() - t5:.2f}s")

    print(f"Total elapsed: {time.perf_counter() - t0:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Minimal COCONUT VTU/CFmesh -> JHelioviewer SunJSON exporter")
    parser.add_argument("input_file", help="Path to a .vtu or .CFmesh COCONUT output file")
    parser.add_argument("output_json")
    parser.add_argument("--n-seed-points", type=int, default=500)
    parser.add_argument("--source-radius", type=float, default=1.05)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--thickness", type=float, default=0.004)
    parser.add_argument("--longitude-offset-deg", "--lon-offset-deg", dest="lon_offset_deg", type=float, default=0.0)
    parser.add_argument("--flip-longitude", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--screenshot", default=None)
    parser.add_argument("--progress", action="store_true", help="Enable tqdm progress bars in loops")
    args = parser.parse_args()

    export_to_jhv_json(
        args.input_file,
        args.output_json,
        n_seed_points=args.n_seed_points,
        source_radius=args.source_radius,
        max_steps=args.max_steps,
        max_lines=args.max_lines,
        thickness=args.thickness,
        lon_offset_deg=args.lon_offset_deg,
        flip_longitude=args.flip_longitude,
        show=args.show,
        screenshot=args.screenshot,
        use_tqdm=args.progress,
    )


if __name__ == "__main__":
    main()

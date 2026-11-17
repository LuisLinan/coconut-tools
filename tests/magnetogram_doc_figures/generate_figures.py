"""Generate a documentation sample of magnetogram runs.

Each *group* below runs the real magnetogram pipelines on a small set of
configurations that differ by one option, and produces:

* ``figures/<group>/<run>.png``: the standard two-panel figure written by the
  pipeline (``plot_maps``), one per run;
* ``figures/<group>/comparison.png``: all the variants of the group stacked
  with a common colour scale (and, when the grids match, the difference with
  the first variant), titled with the net-flux imbalance;
* ``gallery.md``: a MyST page listing every group, ready to be included in
  the Sphinx documentation;
* ``summary.json``: run parameters, timings and flux diagnostics.

No boundary ``.dat`` file is written (``write_map=False``). All runs use the
same target time, chosen because the inputs are already in
``tests/_bigdata_cache/longitude`` (HMI, GONG and ADAPT, including the four
maps needed by the temporal interpolation), so nothing should have to be
downloaded. Another cache can be given with ``--cache``; missing files are then
downloaded there (``JSOC_EMAIL`` must be set for HMI products).

Usage (from the repository root)::

    python tests/magnetogram_doc_figures/generate_figures.py --list
    python tests/magnetogram_doc_figures/generate_figures.py
    python tests/magnetogram_doc_figures/generate_figures.py --groups sph_lmax nld_old_vs_new
    python tests/magnetogram_doc_figures/generate_figures.py --reuse   # re-plot only

``--reuse`` reads the arrays saved in ``runs/`` by a previous execution and
only redraws the comparison figures and the gallery.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from coconut_tools.magnetogram import (  # noqa: E402
    NLD_implicit_method,
    Yaroslavsky_filter,
    gaussian_smoothing,
    sph_filtering,
)
from coconut_tools.magnetogram.io.metadata import (  # noqa: E402
    _first_image_shape_and_header,
)
from coconut_tools.magnetogram.processing import pipeline  # noqa: E402
from coconut_tools.magnetogram.processing.flux_balance import (  # noqa: E402
    _flux_summary,
    _surface_mean_area,
)
from coconut_tools.magnetogram.visualization.plotting import (  # noqa: E402
    _plot_magnetogram_axis,
)

HERE = Path(__file__).resolve().parent
REPO_TESTS = HERE.parent
DEFAULT_CACHE = REPO_TESTS / "_bigdata_cache" / "longitude"

DATE = "2026-09-01T00:00:00"
"""Target time of every run (inputs cached in ``_bigdata_cache/longitude``)."""


# --------------------------------------------------------------------------
# Identity step: shows the pipeline input (reading, resize, interpolation,
# longitude frame) without any filter.
# --------------------------------------------------------------------------
IDENTITY_STEP = pipeline.FilterStep(
    resolve_settings=lambda config: None,
    apply=lambda Br, Theta, Phi, settings: (np.array(Br, dtype=float), {}),
)


def _identity_process(config, target_date, method_used="input", output_path_fig=None,
                      unique_names=False):
    return pipeline.run_magnetogram_pipeline(
        config, target_date, IDENTITY_STEP, method_used=method_used,
        output_path_fig=output_path_fig, unique_names=unique_names,
    )


def native_column_order(Br: np.ndarray, phi: np.ndarray, file_path: str):
    """Reorder a Carrington-normalised map into the column order of its FITS file.

    Column ``j`` of the file is at Carrington longitude
    ``CRVAL1 + (j + 1 - CRPIX1) * CDELT1``; each is matched to the nearest
    column of the normalised map. Returns the reordered map, the display
    longitude of each column (its position from the first column) and the
    Carrington longitude of the first column.
    """
    _, header = _first_image_shape_and_header(file_path)
    step = float(header.get("CD1_1") or header["CDELT1"])
    width = Br.shape[1]
    native_lon = (float(header["CRVAL1"])
                  + (np.arange(width) + 1.0 - float(header["CRPIX1"])) * step) % 360.0
    carrington_lon = np.degrees(phi) % 360.0
    gap = np.abs((native_lon[:, None] - carrington_lon[None, :] + 180.0) % 360.0 - 180.0)
    order = np.argmin(gap, axis=1)
    display_phi = np.radians((np.arange(width) + 0.5) * abs(step))
    return Br[:, order], display_phi, float(native_lon[0])


PROCESSORS = {
    "input": _identity_process,
    "native": _identity_process,  # reordered afterwards, see execute_run
    "sph": sph_filtering.process_magnetogram_date,
    "gaussian": gaussian_smoothing.process_magnetogram_date,
    "nld": NLD_implicit_method.process_magnetogram_date,
    "yaroslavsky": Yaroslavsky_filter.process_magnetogram_date,
}


# --------------------------------------------------------------------------
# Run and group definitions
# --------------------------------------------------------------------------
@dataclass
class Run:
    name: str
    label: str
    method: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class Group:
    name: str
    title: str
    description: str
    runs: list[Run]
    panel: str = "processed"  # "processed" or "original" (pipeline input)
    diff: bool = True  # difference column when grids match
    # Filter groups: add the pipeline input (scaled by 1/2.2 like the outputs)
    # as the first row, and take the differences with respect to it, so the
    # figure shows what each filter removes. Ignored for panel="original".
    show_input: bool = True
    diff_ref: str = "input"  # "input" or "first" (first run)
    zoom: tuple[float, float, float, float] | None = None  # set in build_groups
    section: str = "Misc"
    date: str = DATE  # shown in the comparison title


BASE_CONFIG: dict[str, Any] = {
    "date": DATE,
    "map_type": "HMI_polfil",
    "carrington": True,
    "rotate_to_stonyhurst": False,
    "interpolation": False,
    "interpolation_order": 2,
    "resize": True,
    "flux_correct": False,
    "amp": 1,
    "write_map": False,
    "show_map": True,
    "visu_type": "sinlat",
}

# Reference parameters shared by the "new" (physical, degree-based) filters.
SIGMA_DEG = 2.0
NLD_LENGTH_DEG = 2.0
YARO_RADIUS_DEG = 2.0

# Zoom column: active-region belt (Carrington longitude, sine latitude),
# where the removal of small-scale noise and the shape of the strong
# bipoles are visible. Not used where the longitude axis differs per row.
ZOOM_REGION = (165.0, 240.0, -0.05, 0.45)
NO_ZOOM_GROUPS = {"longitude_frame"}

# (map_type, bracketing map before, interpolation target, bracketing map after).
# Map times of the cached files: GONG xx:14, HMI_hourly xx:24, ADAPT every 2 h.
INTERPOLATION_CASES = (
    ("GONG_mrzqs", "2026-08-31T23:14:00", "2026-08-31T23:44:00", "2026-09-01T00:14:00"),
    ("HMI_hourly", "2026-08-31T23:24:00", "2026-08-31T23:54:00", "2026-09-01T00:24:00"),
    ("ADAPT", "2026-08-31T22:00:00", "2026-08-31T23:00:00", "2026-09-01T00:00:00"),
)


def build_groups() -> list[Group]:
    groups: list[Group] = []

    # ---------------- Reading / pre-processing ----------------
    groups.append(Group(
        "resize_hmi", "HMI: native grid vs resize",
        "HMI_polfil read at its native resolution (1440x3600) and with "
        "``resize=True`` (conservative remap to 360x720). Pipeline input shown.",
        [
            Run("native", "HMI_polfil, resize=False", "input", {"resize": False}),
            Run("resized", "HMI_polfil, resize=True (360x720)", "input", {"resize": True}),
        ],
        panel="original", section="Reading and pre-processing",
    ))
    groups.append(Group(
        "resize_gong", "GONG: native grid vs resize",
        "GONG mrzqs read at its native resolution (180x360) and with "
        "``resize=True`` (360x720). Pipeline input shown.",
        [
            Run("native", "GONG_mrzqs, resize=False", "input",
                {"map_type": "GONG_mrzqs", "resize": False}),
            Run("resized", "GONG_mrzqs, resize=True (360x720)", "input",
                {"map_type": "GONG_mrzqs", "resize": True}),
        ],
        panel="original", section="Reading and pre-processing",
    ))
    # Target time in the middle of two available maps, so that the
    # interpolated map differs from every single map; the two bracketing maps
    # are read alone (interpolation=False at their own time) for comparison.
    for map_type, before, target, after in INTERPOLATION_CASES:
        groups.append(Group(
            f"interpolation_{map_type.lower()}",
            f"{map_type}: temporal interpolation between two maps",
            f"{map_type}: four-map temporal interpolation (``interpolation=True``, "
            f"order 2) at {target}, half-way between the maps of {before} and "
            f"{after}, compared with these two maps read alone "
            "(``interpolation=False``). Pipeline input shown; the right column is "
            "the difference with the interpolated map.",
            [
                Run("interpolated", f"interpolation=True, target {target[11:16]}",
                    "input", {"map_type": map_type, "interpolation": True,
                              "date": target}),
                Run("map_before", f"single map {before[11:16]} (interpolation=False)",
                    "input", {"map_type": map_type, "interpolation": False,
                              "date": before}),
                Run("map_after", f"single map {after[11:16]} (interpolation=False)",
                    "input", {"map_type": map_type, "interpolation": False,
                              "date": after}),
            ],
            panel="original", section="Reading and pre-processing", date=target,
        ))
    groups.append(Group(
        "longitude_frame", "Longitude frame: native file, Carrington, Stonyhurst",
        "GONG mrzqs is stored with its first column at Carrington longitude "
        "271.5 deg (the central meridian of the observation is not at the "
        "edge of the map). Top: the map in the column order of the FITS file "
        "(x axis = position from the first column). Middle: normalised to "
        "Carrington longitude 0 first (``carrington=True``). Bottom: rotated "
        "to Stonyhurst at the target time (``rotate_to_stonyhurst=True``).",
        [
            Run("native", "raw FITS column order", "native",
                {"map_type": "GONG_mrzqs", "resize": False}),
            Run("carrington", "carrington=True", "input",
                {"map_type": "GONG_mrzqs", "resize": False}),
            Run("stonyhurst", "rotate_to_stonyhurst=True", "input",
                {"map_type": "GONG_mrzqs", "resize": False,
                 "rotate_to_stonyhurst": True}),
        ],
        panel="original", diff=False, section="Reading and pre-processing",
    ))
    groups.append(Group(
        "adapt_realizations", "ADAPT realizations",
        "Three of the twelve ADAPT realizations (``adapt_map``). Pipeline input shown.",
        [Run(f"adapt_{i}", f"ADAPT, adapt_map={i}", "input",
             {"map_type": "ADAPT", "adapt_map": i}) for i in (0, 6, 11)],
        panel="original", section="Reading and pre-processing",
    ))
    groups.append(Group(
        "products", "Products at the same time",
        "Different magnetogram products for the same target time, all resized "
        "to 360x720. Pipeline input shown.",
        [Run(m.lower(), m, "input", {"map_type": m})
         for m in ("HMI_polfil", "HMI_small", "HMI_hourly", "GONG_mrzqs",
                   "GONG_mrbqs", "ADAPT")],
        panel="original", diff=False, section="Reading and pre-processing",
    ))

    # ---------------- Flux correction ----------------
    # Most filters conserve the net flux almost exactly on these maps
    # (Gaussian, NLD, SPH: < 0.05 % of the unsigned flux), so "before" and
    # "after" would look identical. The spherical Yaroslavsky filter with a
    # large radius and a fixed h is not flux conserving: on this HMI map it
    # moves the net flux from -1.00 % to -0.57 % of the unsigned flux. A
    # correction before the filter then leaves that residual, a correction
    # after the filter removes it.
    yaro = {"radius_deg": 4.0, "h_gauss": 100.0}
    groups.append(Group(
        "flux_correction", "Net-flux correction (method and stage)",
        "HMI_polfil (resized) filtered with the spherical Yaroslavsky filter "
        "(``radius_deg=4``, ``h_gauss=100``), which does not conserve the net "
        "flux (it adds about +0.4 % of the unsigned flux here): without "
        "correction, then ``surface_mean`` and ``polarity_scaling`` applied "
        "before or after the filter. Correcting before the filter leaves the "
        "imbalance introduced by the filter; only a correction *after* the "
        "filter guarantees a written map with zero net flux. The difference "
        "column is taken with respect to the uncorrected map.",
        [
            Run("none", "flux_correct=False", "yaroslavsky", dict(yaro)),
        ] + [
            Run(f"{short}_{stage}", f"{method}, {stage}", "yaroslavsky",
                {**yaro, "flux_correct": True, "flux_correction_method": method,
                 "flux_correction_stage": stage})
            for short, method in (("mean", "surface_mean"),
                                  ("polarity", "polarity_scaling"))
            for stage in ("before", "after")
        ],
        diff_ref="first", section="Flux correction",
    ))

    # ---------------- Spherical harmonics ----------------
    groups.append(Group(
        "sph_lmax", "Spherical harmonics: impact of lmax",
        "HMI_polfil (resized) projected on spherical harmonics up to "
        "``lmax`` = 10, 20, 40, 80 (``alpha=0``).",
        [Run(f"lmax_{l}", f"lmax={l}", "sph", {"lmax": l, "alpha": 0})
         for l in (10, 20, 40, 80)],
        section="Spherical harmonics",
    ))
    groups.append(Group(
        "sph_alpha", "Spherical harmonics: impact of alpha",
        "HMI_polfil (resized), ``lmax=40``, damping ``alpha`` = 0, 3e-6, 3e-5, 3e-4.",
        [Run(f"alpha_{a:g}", f"lmax=40, alpha={a:g}", "sph", {"lmax": 40, "alpha": a})
         for a in (0.0, 3e-6, 3e-5, 3e-4)],
        section="Spherical harmonics",
    ))
    groups.append(Group(
        "sph_gaussian_prefilter", "Spherical harmonics: Gaussian pre-filter",
        "HMI_polfil (resized), ``lmax=40``: no pre-filter, legacy pixel "
        "template (``template_size=9``) and physical Gaussian "
        f"(``sigma_deg={SIGMA_DEG}``).",
        [
            Run("off", "gaussian_filtering=False", "sph", {"lmax": 40}),
            Run("legacy", "template_size=9", "sph",
                {"lmax": 40, "gaussian_filtering": True, "template_size": 9}),
            Run("physical", f"sigma_deg={SIGMA_DEG}", "sph",
                {"lmax": 40, "gaussian_filtering": True, "sigma_deg": SIGMA_DEG}),
        ],
        section="Spherical harmonics",
    ))

    # ---------------- Gaussian smoothing ----------------
    groups.append(Group(
        "gaussian_old_vs_new", "Gaussian: legacy pixel template vs physical width",
        "HMI_polfil (resized): legacy pixel template (``template_size``) vs "
        "great-circle Gaussian with a width in degrees (``sigma_deg``).",
        [
            Run("legacy_ts9", "legacy, template_size=9", "gaussian", {"template_size": 9}),
            Run("legacy_ts35", "legacy, template_size=35", "gaussian", {"template_size": 35}),
            Run("physical_2deg", f"physical, sigma_deg={SIGMA_DEG}", "gaussian",
                {"sigma_deg": SIGMA_DEG}),
        ],
        section="Gaussian smoothing",
    ))
    groups.append(Group(
        "gaussian_grid_dependence", "Gaussian: grid dependence (legacy vs physical)",
        "The same ``template_size=9`` gives a different physical width on a "
        "GONG grid (180x360) and on a resized HMI grid (360x720), whereas "
        f"``sigma_deg={SIGMA_DEG}`` gives the same smoothing on both.",
        [
            Run("legacy_gong", "GONG native, template_size=9", "gaussian",
                {"map_type": "GONG_mrzqs", "resize": False, "template_size": 9}),
            Run("legacy_hmi", "HMI resized, template_size=9", "gaussian",
                {"template_size": 9}),
            Run("physical_gong", f"GONG native, sigma_deg={SIGMA_DEG}", "gaussian",
                {"map_type": "GONG_mrzqs", "resize": False, "sigma_deg": SIGMA_DEG}),
            Run("physical_hmi", f"HMI resized, sigma_deg={SIGMA_DEG}", "gaussian",
                {"sigma_deg": SIGMA_DEG}),
        ],
        diff=False, section="Gaussian smoothing",
    ))
    groups.append(Group(
        "gaussian_sigma", "Gaussian: impact of sigma_deg",
        "HMI_polfil (resized), physical Gaussian with ``sigma_deg`` = 1, 2, 4, 8.",
        [Run(f"sigma_{s:g}", f"sigma_deg={s:g}", "gaussian", {"sigma_deg": s})
         for s in (1.0, 2.0, 4.0, 8.0)],
        section="Gaussian smoothing",
    ))

    # ---------------- Nonlinear diffusion ----------------
    groups.append(Group(
        "nld_old_vs_new", "NLD: legacy pixel solver vs spherical solver",
        "HMI_polfil (resized). Legacy: pixel Gaussian (``gaussian_sigma=1``) + "
        "Perona-Malik on a Cartesian pixel grid (``tau``, ``iterations``). "
        "Spherical: great-circle pre-smoothing + finite-volume diffusion on the "
        f"sphere (``diffusion_length_deg={NLD_LENGTH_DEG}``).",
        [
            Run("legacy_tau5_it7", "legacy, tau=5, iterations=7", "nld",
                {"tau": 5, "iterations": 7, "gaussian_sigma": 1.0}),
            Run("legacy_tau25_it7", "legacy, tau=25, iterations=7", "nld",
                {"tau": 25, "iterations": 7, "gaussian_sigma": 1.0}),
            Run("spherical", f"spherical, L={NLD_LENGTH_DEG} deg, sigma_g=1 deg", "nld",
                {"diffusion_length_deg": NLD_LENGTH_DEG, "gaussian_sigma_deg": 1.0}),
        ],
        section="Nonlinear diffusion",
    ))
    groups.append(Group(
        "nld_length", "NLD (spherical): impact of diffusion_length_deg",
        "HMI_polfil (resized), spherical NLD with ``diffusion_length_deg`` = "
        "1, 2, 4 (``gaussian_sigma_deg=1``, automatic lambda).",
        [Run(f"L_{L:g}", f"diffusion_length_deg={L:g}", "nld",
             {"diffusion_length_deg": L, "gaussian_sigma_deg": 1.0})
         for L in (1.0, 2.0, 4.0)],
        section="Nonlinear diffusion",
    ))
    groups.append(Group(
        "nld_lambda", "NLD (spherical): impact of lambda",
        f"HMI_polfil (resized), ``diffusion_length_deg={NLD_LENGTH_DEG}``: automatic "
        "lambda (area-weighted MAD) vs fixed ``lambda_g_per_deg``. A small lambda "
        "preserves strong gradients (active regions), a large one tends to a "
        "linear (Gaussian) diffusion.",
        [
            Run("auto", "lambda automatic", "nld",
                {"diffusion_length_deg": NLD_LENGTH_DEG, "gaussian_sigma_deg": 1.0}),
            Run("lambda_1", "lambda_g_per_deg=1", "nld",
                {"diffusion_length_deg": NLD_LENGTH_DEG, "gaussian_sigma_deg": 1.0,
                 "lambda_g_per_deg": 1.0}),
            Run("lambda_100", "lambda_g_per_deg=100", "nld",
                {"diffusion_length_deg": NLD_LENGTH_DEG, "gaussian_sigma_deg": 1.0,
                 "lambda_g_per_deg": 100.0}),
        ],
        section="Nonlinear diffusion",
    ))

    # ---------------- Yaroslavsky ----------------
    groups.append(Group(
        "yaroslavsky_old_vs_new", "Yaroslavsky: legacy pixel filter vs spherical filter",
        "GONG mrzqs (native 180x360, the legacy filter is slow on larger grids). "
        "Legacy: neighbourhood ``Rn`` in pixels, ``alpha``, pixel pre-smoothing "
        "``sig``. Spherical: disk of radius ``radius_deg`` on the sphere, "
        "automatic ``h_gauss``, great-circle pre-smoothing ``sig_deg``.",
        [
            Run("legacy", "legacy, alpha=1.4, Rn=3, sig=1", "yaroslavsky",
                {"map_type": "GONG_mrzqs", "resize": False,
                 "alpha": 1.4, "Rn": 3, "sig": 1.0}),
            Run("spherical", f"spherical, radius_deg={YARO_RADIUS_DEG}, sig_deg=1",
                "yaroslavsky",
                {"map_type": "GONG_mrzqs", "resize": False,
                 "radius_deg": YARO_RADIUS_DEG, "sig_deg": 1.0}),
        ],
        section="Yaroslavsky (local weighted) filter",
    ))
    groups.append(Group(
        "yaroslavsky_radius", "Yaroslavsky (spherical): impact of radius_deg",
        "HMI_polfil (resized), spherical filter with ``radius_deg`` = 1, 2, 4, "
        "no pre-smoothing, ``h_gauss=20``. (With the automatic h, about 3 G "
        "here, only pixels within a few Gauss of each other are averaged and "
        "the radius has little effect; see the h_gauss group.)",
        [Run(f"r_{r:g}", f"radius_deg={r:g}, h_gauss=20", "yaroslavsky",
             {"radius_deg": r, "h_gauss": 20.0}) for r in (1.0, 2.0, 4.0)],
        section="Yaroslavsky (local weighted) filter",
    ))
    groups.append(Group(
        "yaroslavsky_h", "Yaroslavsky (spherical): impact of h_gauss",
        f"HMI_polfil (resized), ``radius_deg={YARO_RADIUS_DEG}``: automatic "
        "``h_gauss`` vs fixed values. A small h only averages similar values "
        "(edge preserving), a large h tends to a plain disk average. No "
        "pre-smoothing.",
        [
            Run("auto", "h_gauss automatic", "yaroslavsky",
                {"radius_deg": YARO_RADIUS_DEG}),
            Run("h_5", "h_gauss=5", "yaroslavsky",
                {"radius_deg": YARO_RADIUS_DEG, "h_gauss": 5.0}),
            Run("h_100", "h_gauss=100", "yaroslavsky",
                {"radius_deg": YARO_RADIUS_DEG, "h_gauss": 100.0}),
        ],
        section="Yaroslavsky (local weighted) filter",
    ))

    # ---------------- Cross-method ----------------
    groups.append(Group(
        "methods", "Comparison of the filters",
        "HMI_polfil (resized) processed by each pipeline with comparable "
        f"physical scales: spherical harmonics (lmax=20), Gaussian "
        f"(sigma={SIGMA_DEG} deg), spherical NLD (L={NLD_LENGTH_DEG} deg), spherical "
        f"Yaroslavsky (radius={YARO_RADIUS_DEG} deg). All with surface_mean flux "
        "correction after the filter (before for the spherical harmonics).",
        [
            Run("sph", "spherical harmonics, lmax=20", "sph",
                {"lmax": 20, "flux_correct": True}),
            Run("gaussian", f"Gaussian, sigma_deg={SIGMA_DEG}", "gaussian",
                {"sigma_deg": SIGMA_DEG, "flux_correct": True}),
            Run("nld", f"NLD, L={NLD_LENGTH_DEG} deg", "nld",
                {"diffusion_length_deg": NLD_LENGTH_DEG, "gaussian_sigma_deg": 1.0,
                 "flux_correct": True}),
            Run("yaroslavsky", f"Yaroslavsky, radius={YARO_RADIUS_DEG} deg", "yaroslavsky",
                {"radius_deg": YARO_RADIUS_DEG, "sig_deg": 1.0, "flux_correct": True}),
        ],
        diff=False, section="Comparison of the filters",
    ))
    for group in groups:
        if group.zoom is None and group.name not in NO_ZOOM_GROUPS:
            group.zoom = ZOOM_REGION
    return groups


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------
class _PlotCapture:
    """Wrap ``pipeline.plot_maps`` to keep the arrays it receives."""

    def __init__(self):
        self.original = pipeline.plot_maps
        self.last: dict[str, np.ndarray] | None = None

    def __call__(self, Br, Br_mode, theta, phi, *args, **kwargs):
        self.last = {
            "original": np.asarray(Br, dtype=np.float32),
            "processed": np.asarray(Br_mode, dtype=np.float32),
            "theta": np.asarray(theta, dtype=float),
            "phi": np.asarray(phi, dtype=float),
        }
        return self.original(Br, Br_mode, theta, phi, *args, **kwargs)

    def __enter__(self):
        pipeline.plot_maps = self
        return self

    def __exit__(self, *exc):
        pipeline.plot_maps = self.original
        return False


def _flux_imbalance(values: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> float:
    area = _surface_mean_area(theta, phi, values.shape)
    return float(_flux_summary(values.astype(float), area)[3])


def execute_run(group: Group, run: Run, out_dir: Path, cache: Path, email: str | None,
                capture: _PlotCapture) -> dict[str, Any]:
    fig_path = out_dir / "figures" / group.name / f"{run.name}.png"
    npz_path = out_dir / "runs" / group.name / f"{run.name}.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        **BASE_CONFIG,
        **run.config,
        "output_dir": str(cache),
        "download_dir": str(cache),
        "output_path_fig": str(fig_path),
        "drms_email": email,
    }
    capture.last = None
    start = time.perf_counter()
    result = PROCESSORS[run.method](
        config, config["date"], method_used=run.method, output_path_fig=str(fig_path),
    )
    elapsed = time.perf_counter() - start
    if capture.last is None:
        raise RuntimeError("The pipeline did not call plot_maps.")
    if run.method == "native":
        source = result["local_file"]
        source = source[0] if isinstance(source, list) else source
        native, display_phi, lon0 = native_column_order(
            capture.last["original"], capture.last["phi"], source)
        capture.last.update(original=native, processed=native, phi=display_phi)
        run.label = f"{run.label}, first column = Carrington {lon0:.1f} deg"
        capture.original(native, native, capture.last["theta"], display_phi,
                         config["map_type"], "sinlat", output_path=str(fig_path),
                         date=result["magnetogram_date"])
    np.savez_compressed(npz_path, **capture.last)
    keys = ("nld_mode", "nld_info", "yaroslavsky_mode", "yaroslavsky_info",
            "template_size", "sigma_deg", "timestep", "rotation_angle",
            "flux_correction_stage")
    meta = {key: result[key] for key in keys if key in result}
    meta["effective_date"] = str(result.get("effective_date"))
    return {"elapsed_s": round(elapsed, 2), "metadata": meta}


def load_run(out_dir: Path, group: Group, run: Run) -> dict[str, np.ndarray] | None:
    npz_path = out_dir / "runs" / group.name / f"{run.name}.npz"
    if not npz_path.exists():
        return None
    with np.load(npz_path) as data:
        return {key: data[key] for key in data.files}


# --------------------------------------------------------------------------
# Comparison figure
# --------------------------------------------------------------------------
def _display_longitude(phi: np.ndarray) -> np.ndarray:
    longitude = np.degrees(phi)
    if np.any(np.diff(longitude) < 0.0):
        longitude = np.degrees(np.unwrap(phi))
        if longitude[0] > 180.0:
            longitude = longitude - 360.0
    return longitude


def _robust_limit(values: list[np.ndarray], percentile: float = 99.5) -> float:
    stack = np.concatenate([np.abs(v).ravel() for v in values])
    return float(np.nanpercentile(stack, percentile)) or 1.0


def plot_comparison(group: Group, arrays: list[tuple[Run, dict[str, np.ndarray]]],
                    path: Path) -> list[float]:
    """Stack the runs of a group: map | zoom | difference with the reference.

    Returns the flux imbalance of every run (input row excluded).
    """
    key = group.panel
    unit = "Br [G]" if key == "original" else "Br [G/2.2]"
    rows = [(run.label, a[key], a["theta"], a["phi"]) for run, a in arrays]
    n_input = 0
    same_input = all(a["original"].shape == arrays[0][1]["original"].shape
                     for _, a in arrays)
    if key == "processed" and group.show_input and same_input:
        first = arrays[0][1]
        amp = arrays[0][0].config.get("amp", BASE_CONFIG["amp"])
        rows.insert(0, ("input magnetogram (scaled by 1/2.2 for comparison)",
                        first["original"] / pipeline.OUTPUT_SCALE_DIVISOR * amp,
                        first["theta"], first["phi"]))
        n_input = 1
    # Colour scale from the processed maps, so that the filtered maps stay
    # legible (the noisier input saturates, which is what the filter removes).
    limit = _robust_limit([r[1] for r in rows[n_input:]])

    ref_index = n_input if (group.diff_ref == "first" or not n_input) else 0
    with_diff = group.diff and all(r[1].shape == rows[ref_index][1].shape for r in rows)
    diffs = {}
    diff_limit = 1.0
    if with_diff:
        ref = rows[ref_index][1]
        diffs = {i: r[1] - ref for i, r in enumerate(rows) if i != ref_index}
        if diffs:
            diff_limit = _robust_limit(list(diffs.values()))
    ref_name = "input" if (ref_index == 0 and n_input) else "first run"

    columns = ["map"] + (["zoom"] if group.zoom else []) + (["diff"] if with_diff else [])
    widths = {"map": 2.0, "zoom": 1.25, "diff": 2.0}
    fig, axes = plt.subplots(
        len(rows), len(columns), squeeze=False,
        figsize=(3.9 * sum(widths[c] for c in columns), 2.6 * len(rows) + 0.8),
        gridspec_kw={"width_ratios": [widths[c] for c in columns]},
    )
    imbalances = []
    for i, (label, values, theta, phi) in enumerate(rows):
        latitude = 90.0 - np.degrees(theta)
        longitude = _display_longitude(phi)
        imbalance = _flux_imbalance(values, theta, phi)
        if i >= n_input:
            imbalances.append(imbalance)
        for j, column in enumerate(columns):
            ax = axes[i, j]
            if column == "map":
                im, ylabel = _plot_magnetogram_axis(ax, values, longitude, latitude,
                                                    "sinlat", limit)
                ax.set_title(f"{label}  ({values.shape[0]}x{values.shape[1]}, "
                             f"imbalance {imbalance:+.2f} %)", fontsize=10)
                ax.set_ylabel(ylabel)
                if group.zoom:
                    x0, x1, y0, y1 = group.zoom
                    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                               lw=1.0, ec="k", ls="--"))
                fig.colorbar(im, ax=ax, extend="both").set_label(unit)
            elif column == "zoom":
                _plot_magnetogram_axis(ax, values, longitude, latitude, "sinlat", limit)
                x0, x1, y0, y1 = group.zoom
                ax.set_xlim(x0, x1)
                ax.set_ylim(y0, y1)
                ax.set_title("zoom", fontsize=10)
            else:
                if i not in diffs:
                    ax.axis("off")
                    ax.text(0.5, 0.5, "reference", ha="center", va="center",
                            transform=ax.transAxes, fontsize=12, color="0.4")
                    continue
                d = diffs[i]
                imd, _ = _plot_magnetogram_axis(ax, d, longitude, latitude,
                                                "sinlat", diff_limit)
                rms = float(np.sqrt(np.mean(d.astype(float) ** 2)))
                ax.set_title(f"difference with {ref_name} (RMS {rms:.3g})", fontsize=10)
                fig.colorbar(imd, ax=ax, extend="both").set_label(unit)
            if i == len(rows) - 1:
                ax.set_xlabel("Longitude [deg]")
    fig.suptitle(f"{group.title} - {group.date} UTC", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return imbalances


# --------------------------------------------------------------------------
# Gallery page
# --------------------------------------------------------------------------
def write_gallery(out_dir: Path, groups: list[Group], summary: dict[str, Any]) -> Path:
    lines = [
        "# Magnetogram processing gallery",
        "",
        f"Generated by `tests/magnetogram_doc_figures/generate_figures.py` for the "
        f"target time {DATE} UTC (unless stated otherwise). Each figure stacks the variants of one option "
        "with a common colour scale. For the filters, the first row is the input "
        "map (scaled by 1/2.2 like the outputs) and the right column shows the "
        "difference with it, i.e. what the filter removes; the middle column "
        "zooms on the dashed box (active-region belt). The title of each panel "
        "gives the net-flux imbalance (net / unsigned flux).",
        "",
    ]
    section = None
    for group in groups:
        info = summary["groups"].get(group.name)
        if not info or not info.get("comparison"):
            continue
        if group.section != section:
            section = group.section
            lines += [f"## {section}", ""]
        lines += [f"### {group.title}", "", group.description, "",
                  f"![{group.title}](figures/{group.name}/comparison.png)", "",
                  "| Run | Method | Parameters | Time [s] | Figure |",
                  "|---|---|---|---|---|"]
        for run in group.runs:
            r = info["runs"].get(run.name, {})
            params = ", ".join(f"`{k}={v}`" for k, v in run.config.items())
            status = r.get("elapsed_s", r.get("status", ""))
            lines.append(f"| {run.label} | {run.method} | {params or 'defaults'} | "
                         f"{status} | [png](figures/{group.name}/{run.name}.png) |")
        lines.append("")
    path = out_dir / "gallery.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", nargs="*", help="Groups to run (default: all).")
    parser.add_argument("--list", action="store_true", help="List the groups and exit.")
    parser.add_argument("--reuse", action="store_true",
                        help="Do not run the pipelines; re-plot from runs/*.npz.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Run only the runs without a saved runs/*.npz.")
    parser.add_argument("--out", type=Path, default=HERE, help="Output directory.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                        help="Magnetogram cache / download directory.")
    args = parser.parse_args(argv)

    all_groups = build_groups()
    if args.list:
        for g in all_groups:
            print(f"{g.name:28s} {len(g.runs)} runs  {g.title}")
        return 0
    selected = all_groups
    if args.groups:
        unknown = set(args.groups) - {g.name for g in all_groups}
        if unknown:
            parser.error(f"Unknown groups: {', '.join(sorted(unknown))}")
        selected = [g for g in all_groups if g.name in args.groups]

    summary_path = args.out / "summary.json"
    summary = (json.loads(summary_path.read_text(encoding="utf-8"))
               if summary_path.exists() else {})
    summary.update({"date": DATE, "cache": str(args.cache)})
    summary.setdefault("groups", {})
    email = os.environ.get("JSOC_EMAIL") or os.environ.get("DRMS_EMAIL")

    failures = 0
    with _PlotCapture() as capture:
        for group in selected:
            print(f"\n=== {group.name}: {group.title}", flush=True)
            info = summary["groups"].setdefault(group.name, {})
            info["title"] = group.title
            runs_info = info.setdefault("runs", {})
            arrays = []
            for run in group.runs:
                existing = load_run(args.out, group, run)
                if args.reuse or (args.skip_existing and existing is not None):
                    if existing is None:
                        print(f"  - {run.name}: no saved arrays, skipped")
                        continue
                    arrays.append((run, existing))
                    continue
                print(f"  - {run.name} ({run.label}) ...", flush=True)
                try:
                    run_info = execute_run(group, run, args.out, args.cache, email,
                                           capture)
                except Exception as exc:  # keep going with the other runs
                    failures += 1
                    traceback.print_exc()
                    runs_info[run.name] = {"status": f"failed: {exc}"}
                    continue
                print(f"    done in {run_info['elapsed_s']} s", flush=True)
                runs_info[run.name] = {**run_info, "config": run.config}
                arrays.append((run, load_run(args.out, group, run)))
            if arrays:
                comparison = args.out / "figures" / group.name / "comparison.png"
                imbalances = plot_comparison(group, arrays, comparison)
                for (run, _), imb in zip(arrays, imbalances):
                    runs_info.setdefault(run.name, {})["flux_imbalance_percent"] = round(imb, 4)
                info["comparison"] = str(comparison.relative_to(args.out).as_posix())
                print(f"  -> {comparison}")

    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    gallery = write_gallery(args.out, all_groups, summary)
    print(f"\nGallery: {gallery}\nSummary: {summary_path}\nFailed runs: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

"""Angular power spectra of the filtered maps saved by ``generate_figures.py``.

For each selected run (HMI_polfil resized to 360x720, same input for all), the
script computes the angular power spectrum ``S(l)`` of the input and of the
filtered map (rescaled by 2.2 to undo the output scaling) and plots the
transfer ratio ``S_out(l) / S_in(l)``: 1 = scale kept, 0 = scale removed.
It also prints the effective cut-off degree (ratio = 0.5), the unsigned flux
kept and the strong-field amplitude kept (99.9th percentile of |B|).

The transform is a plain quadrature (FFT in longitude, fully normalised
associated Legendre functions in colatitude, equal-area sine-latitude
weights). It is not an exact transform, but the same error affects every map
on the same grid, so the ratios are meaningful.

Usage (after generate_figures.py)::

    python tests/magnetogram_doc_figures/spectra.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from coconut_tools.magnetogram.processing.flux_balance import _surface_mean_area  # noqa: E402

HERE = Path(__file__).resolve().parent
LMAX = 120
SCALE = 2.2

# (group/run, label, family) -- all on the same HMI_polfil 360x720 input.
RUNS = [
    ("sph_lmax/lmax_20", "SPH lmax=20", "sph"),
    ("sph_lmax/lmax_40", "SPH lmax=40", "sph"),
    ("gaussian_sigma/sigma_1", "Gaussian sigma=1 deg", "gaussian"),
    ("gaussian_sigma/sigma_2", "Gaussian sigma=2 deg", "gaussian"),
    ("gaussian_sigma/sigma_4", "Gaussian sigma=4 deg", "gaussian"),
    ("nld_length/L_2", "NLD spherical L=2 deg (auto lambda)", "nld"),
    ("nld_length/L_4", "NLD spherical L=4 deg (auto lambda)", "nld"),
    ("nld_lambda/lambda_100", "NLD spherical L=2 deg, lambda=100", "nld"),
    ("nld_old_vs_new/legacy_tau25_it7", "NLD legacy tau=25", "nld"),
    ("yaroslavsky_h/auto", "Yaroslavsky r=2 deg (auto h)", "yaro"),
    ("yaroslavsky_radius/r_4", "Yaroslavsky r=4 deg, h=20", "yaro"),
    ("yaroslavsky_h/h_100", "Yaroslavsky r=2 deg, h=100", "yaro"),
]
COLORS = {"sph": "tab:red", "gaussian": "tab:blue", "nld": "tab:green", "yaro": "tab:purple"}
STYLES = ["-", "--", ":", "-."]


def _legendre_table(mu: np.ndarray, lmax: int) -> np.ndarray:
    """Fully normalised P_lm(mu), shape (lmax+1, lmax+1, n), zero for l < m."""
    sin_t = np.sqrt(np.clip(1.0 - mu**2, 0.0, None))
    table = np.zeros((lmax + 1, lmax + 1, mu.size))
    pmm = np.full(mu.size, np.sqrt(1.0 / (4.0 * np.pi)))
    for m in range(lmax + 1):
        if m > 0:
            pmm = -np.sqrt((2.0 * m + 1.0) / (2.0 * m)) * sin_t * pmm
        table[m, m] = pmm
        if m + 1 <= lmax:
            table[m + 1, m] = np.sqrt(2.0 * m + 3.0) * mu * pmm
        for l in range(m + 2, lmax + 1):
            a = np.sqrt((4.0 * l * l - 1.0) / (l * l - m * m))
            b = np.sqrt(((l - 1.0) ** 2 - m * m) / (4.0 * (l - 1.0) ** 2 - 1.0))
            table[l, m] = a * (mu * table[l - 1, m] - b * table[l - 2, m])
    return table


def power_spectrum(B: np.ndarray, theta: np.ndarray, legendre: np.ndarray) -> np.ndarray:
    n_theta, n_phi = B.shape
    lmax = legendre.shape[0] - 1
    weights = _surface_mean_area(theta, np.linspace(0, 2 * np.pi, n_phi, endpoint=False),
                                 B.shape)[:, 0] * n_phi  # ring area = sin(t) dt * 2pi
    fourier = np.fft.rfft(B, axis=1)[:, : lmax + 1] / n_phi  # (n_theta, m)
    spectrum = np.zeros(lmax + 1)
    for m in range(lmax + 1):
        coeffs = legendre[:, m, :] @ (fourier[:, m] * weights)  # a_lm for all l
        factor = 1.0 if m == 0 else 2.0
        spectrum += factor * np.abs(coeffs) ** 2
    return spectrum


def main() -> None:
    reference = np.load(HERE / "runs" / f"{RUNS[0][0]}.npz")
    theta = reference["theta"]
    original = reference["original"].astype(float)
    legendre = _legendre_table(np.cos(theta), LMAX)
    s_in = power_spectrum(original, theta, legendre)
    area = _surface_mean_area(theta, reference["phi"], original.shape)
    flux_in = np.sum(np.abs(original) * area)
    peak_in = np.percentile(np.abs(original), 99.9)

    l = np.arange(LMAX + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    counters: dict[str, int] = {}
    print(f"{'run':42s} {'l(ratio=0.5)':>12s} {'unsigned flux kept':>19s} "
          f"{'|B| p99.9 kept':>15s}")
    for key, label, family in RUNS:
        path = HERE / "runs" / f"{key}.npz"
        if not path.exists():
            print(f"{label:42s} missing ({path.name})")
            continue
        data = np.load(path)
        if not np.allclose(data["original"], reference["original"]):
            print(f"{label:42s} skipped: different input map")
            continue
        out = data["processed"].astype(float) * SCALE
        ratio = power_spectrum(out, theta, legendre) / s_in
        below = np.nonzero(ratio[1:] < 0.5)[0]
        l_half = int(below[0] + 1) if below.size else None
        flux_kept = np.sum(np.abs(out) * area) / flux_in
        peak_kept = np.percentile(np.abs(out), 99.9) / peak_in
        print(f"{label:42s} {str(l_half or f'>{LMAX}'):>12s} {flux_kept:19.0%} "
              f"{peak_kept:15.0%}")
        style = STYLES[counters.get(family, 0) % len(STYLES)]
        counters[family] = counters.get(family, 0) + 1
        ax.plot(l[1:], ratio[1:], style, color=COLORS[family], lw=1.6, label=label)

    ax.axhline(0.5, color="0.6", lw=0.8)
    ax.set_xlim(1, LMAX)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("spherical-harmonic degree l  (angular scale ~ 180/l deg)")
    ax.set_ylabel("power kept  S_out(l) / S_in(l)")
    ax.set_title("How much each filter keeps of each angular scale "
                 "(HMI_polfil 360x720, 2026-09-01)")
    secondary = ax.secondary_xaxis(
        "top", functions=(lambda x: 180.0 / np.maximum(x, 1e-3),
                          lambda x: 180.0 / np.maximum(x, 1e-3)))
    secondary.set_xticks([9, 6, 4, 3, 2, 1.5])
    secondary.set_xlabel("half-wavelength [deg]")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = HERE / "figures" / "filter_spectra" / "transfer.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()

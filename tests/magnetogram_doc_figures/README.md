# Magnetogram documentation figures

`generate_figures.py` runs the magnetogram pipelines on a set of configurations
chosen to illustrate every option of the documentation, and stores the
figures here. It is not a pytest test (no `test_` prefix) and writes no `.dat`
file.

All runs use the target time `2026-09-01T00:00:00`, whose inputs (HMI, GONG,
ADAPT, plus the four maps needed by the temporal interpolation) are already
in `tests/_bigdata_cache/longitude`.

```bash
python tests/magnetogram_doc_figures/generate_figures.py --list              # groups
python tests/magnetogram_doc_figures/generate_figures.py                     # everything
python tests/magnetogram_doc_figures/generate_figures.py --groups sph_lmax   # one group
python tests/magnetogram_doc_figures/generate_figures.py --skip-existing     # resume
python tests/magnetogram_doc_figures/generate_figures.py --reuse             # re-plot only
```

Outputs:

| Path | Content |
|---|---|
| `figures/<group>/comparison.png` | all variants of a group, common colour scale; for the filters the first row is the input map (/2.2) and the right column the difference with it (what the filter removes); middle column = zoom on an active-region box |
| `figures/<group>/<run>.png` | standard pipeline figure (original / processed) of one run |
| `gallery.md` | MyST page listing all the groups, to include in the Sphinx docs |
| `summary.json` | parameters, timings, flux imbalance and filter metadata of every run |
| `runs/` | arrays of each run (`.npz`, git-ignored), used by `--reuse` |

Groups:

| Section | Groups |
|---|---|
| Reading and pre-processing | `resize_hmi`, `resize_gong`, `interpolation_gong_mrzqs`, `interpolation_hmi_hourly`, `interpolation_adapt`, `longitude_frame`, `adapt_realizations`, `products` |
| Flux correction | `flux_correction` (none / surface_mean before / after / polarity_scaling) |
| Spherical harmonics | `sph_lmax`, `sph_alpha`, `sph_gaussian_prefilter` |
| Gaussian smoothing | `gaussian_old_vs_new`, `gaussian_grid_dependence`, `gaussian_sigma` |
| Nonlinear diffusion | `nld_old_vs_new`, `nld_length`, `nld_lambda` |
| Yaroslavsky | `yaroslavsky_old_vs_new`, `yaroslavsky_radius`, `yaroslavsky_h` |
| All filters | `methods` |

To add a case, add a `Group` (or a `Run` in an existing group) in
`build_groups()`; a run only lists the keys that differ from `BASE_CONFIG`.

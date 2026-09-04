import os
import importlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest


DATE = "2011-09-04T12:00:00"
MAP_TYPE = "GONG"


def _center_crop(Br, Theta, Phi, n_theta=8, n_phi=8):
    i0 = Br.shape[0] // 2 - n_theta // 2
    i1 = i0 + n_theta
    return Br[i0:i1, :n_phi], Theta[i0:i1, :n_phi], Phi[i0:i1, :n_phi]


def _assert_artifact(path: Path):
    assert path.exists(), f"Missing output: {path}"
    assert path.stat().st_size > 0, f"Empty output: {path}"


def test_yaroslavsky_spacing_uses_radian_arc_length(monkeypatch):
    from coconut_tools.magnetogram import Yaroslavsky_filter

    captured = {}

    def fake_filter3(image, dx, dy, alpha, Rn):
        captured.update({"dx": dx, "dy": dy, "alpha": alpha, "Rn": Rn})
        return image.copy()

    monkeypatch.setattr(Yaroslavsky_filter, "filter3", fake_filter3)

    Br = np.ones((4, 5))
    theta = np.array([0.0, 0.2, 0.5, 0.9])
    phi = np.linspace(0.0, 2.0 * np.pi, Br.shape[1], endpoint=False)

    result = Yaroslavsky_filter.filter_radial_field_weighted(
        Br,
        phi,
        theta,
        alpha_factor=1.4,
        Rn=2,
        sig=0.0,
    )

    expected_delta = 696.34e6 * max(np.median(np.diff(theta)), np.median(np.diff(phi)))
    assert np.array_equal(result, Br)
    assert captured["dx"] == pytest.approx(expected_delta)
    assert captured["dy"] == pytest.approx(expected_delta)
    assert captured["alpha"] == 1.4
    assert captured["Rn"] == 2


def test_local_weighted_filter_uses_article_h_relation(monkeypatch):
    from coconut_tools.magnetogram.filters import yaroslavsky as local_weigh_filter

    captured = {}

    def fake_main_loop_integration(u, i, j, Rn, h, dx, dy):
        if not captured:
            captured.update({"Rn": Rn, "h": h, "dx": dx, "dy": dy})
        return u[i, j]

    class FakePool:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def starmap(self, func, tasks):
            return [func(*task) for task in tasks]

    monkeypatch.setattr(local_weigh_filter, "main_loop_integration", fake_main_loop_integration)
    monkeypatch.setattr(local_weigh_filter, "Pool", FakePool)

    image = np.arange(4.0).reshape(2, 2)
    result = local_weigh_filter.filter3(image, dx=10.0, dy=20.0, alpha=1.4, Rn=2.0)

    assert np.array_equal(result, image)
    assert captured["Rn"] == pytest.approx(40.0)
    assert captured["h"] == pytest.approx(2.0**1.4)
    assert captured["dx"] == 10.0
    assert captured["dy"] == 20.0


def test_local_weighted_filter_uses_dx_for_columns_and_dy_for_rows():
    from coconut_tools.magnetogram.filters.yaroslavsky import Th

    image = np.ones((5, 5), dtype=float)
    _, weights = Th(image, i=2, j=2, Rn=1.5, h=1.0, dx=1.0, dy=2.0)

    assert weights.shape == (1, 3)
    assert np.count_nonzero(weights) == 3


@pytest.mark.parametrize(
    ("module_name", "filter_name", "filter_result"),
    [
        (
            "coconut_tools.magnetogram.Yaroslavsky_filter",
            "filter_radial_field_weighted",
            lambda Br: Br * 4.4,
        ),
        (
            "coconut_tools.magnetogram.NLD_implicit_method",
            "filter_radial_field",
            lambda Br: (Br * 4.4, 0.5),
        ),
    ],
)
def test_filtered_pipelines_apply_configured_amp_after_normalization(
    monkeypatch,
    tmp_path,
    module_name,
    filter_name,
    filter_result,
):
    module = importlib.import_module(module_name)
    captured = {}
    Br = np.ones((2, 3))
    theta = np.array([0.25, 0.75])
    phi = np.linspace(0.0, 2.0 * np.pi, Br.shape[1], endpoint=False)
    Theta, Phi = np.meshgrid(theta, phi, indexing="ij")
    effective_date = datetime.fromisoformat(DATE)

    monkeypatch.setattr(
        module,
        "generate_output_and_map_names",
        lambda *args, **kwargs: (str(tmp_path / "map.dat"), str(tmp_path / "source.fits")),
    )
    monkeypatch.setattr(module, "read_magnetogram", lambda *args, **kwargs: (Br, Theta, Phi))
    monkeypatch.setattr(module, "magnetogram_effective_date", lambda *args, **kwargs: effective_date)
    monkeypatch.setattr(module, "magnetogram_display_date", lambda *args, **kwargs: effective_date)
    monkeypatch.setattr(
        module,
        "apply_configured_longitude_rotation",
        lambda Br_in, Br_linear, *args, **kwargs: (Br_in, Br_linear, None),
    )
    monkeypatch.setattr(module, filter_name, lambda Br_in, *args, **kwargs: filter_result(Br_in))
    monkeypatch.setattr(
        module,
        "write_bc_file",
        lambda output_name, Br_out, *args, **kwargs: captured.setdefault("Br_out", Br_out.copy()),
    )

    module.process_magnetogram_date(
        {
            "date": DATE,
            "map_type": MAP_TYPE,
            "output_dir": str(tmp_path),
            "amp": 3,
            "write_map": True,
            "show_map": False,
            "interpolation": False,
            "rotate_to_stonyhurst": False,
        },
        DATE,
    )

    assert np.array_equal(captured["Br_out"], Br * 6.0)


@pytest.mark.parametrize("map_type", ["HMI_hourly", "HMI_fdt"])
def test_sph_pipeline_uses_hmi_interpolation_at_requested_time(
    monkeypatch,
    tmp_path,
    map_type,
):
    from coconut_tools.magnetogram import sph_filtering

    target = datetime(2026, 7, 1, 7, 15)
    local_files = [
        str(tmp_path / f"{map_type.lower()}_{hour}.fits")
        for hour in range(6, 10)
    ]
    selection = object()
    Br = np.arange(8.0).reshape(2, 4)
    Br_linear = Br + 1.0
    theta = np.array([0.25, 0.75])
    phi = np.linspace(0.0, 2.0 * np.pi, Br.shape[1], endpoint=False)
    Theta, Phi = np.meshgrid(theta, phi, indexing="ij")
    captured = {}

    monkeypatch.setattr(
        sph_filtering,
        "generate_output_and_interpolation_map_names",
        lambda *args, **kwargs: (
            str(tmp_path / f"map_{map_type.lower()}_sph.dat"),
            local_files,
            selection,
        ),
    )
    monkeypatch.setattr(
        sph_filtering,
        "generate_output_and_map_names",
        lambda *args, **kwargs: pytest.fail(
            f"{map_type} interpolation must not use the single-map download path"
        ),
    )

    def fake_read_interpolated(*args, **kwargs):
        captured["read_resize"] = kwargs["resize"]
        return Br, Theta, Phi, Br_linear

    monkeypatch.setattr(
        sph_filtering,
        "read_interpolated_magnetogram",
        fake_read_interpolated,
    )

    def fake_rotation(Br_in, Br_linear_in, *args, **kwargs):
        captured["use_interpolation"] = args[3]
        captured["effective_date"] = kwargs["effective_date"]
        captured["rotation_resize"] = kwargs["resize"]
        return Br_in, Br_linear_in, None

    monkeypatch.setattr(
        sph_filtering,
        "apply_configured_longitude_rotation",
        fake_rotation,
    )
    monkeypatch.setattr(
        sph_filtering,
        "project_and_reconstruct",
        lambda Br_in, *args, **kwargs: (Br_in, np.array([1.0])),
    )

    result = sph_filtering.process_magnetogram_date(
        {
            "date": target.isoformat(),
            "map_type": map_type,
            "output_dir": str(tmp_path),
            "interpolation": True,
            "resize": True,
            "write_map": False,
            "show_map": False,
            "rotate_to_stonyhurst": True,
        },
        target,
    )

    assert result["date"] == target
    assert result["effective_date"] == target
    assert result["magnetogram_date"] == target
    assert result["local_file"] == local_files
    assert result["selection"] is selection
    np.testing.assert_array_equal(result["Br_linear"], Br_linear)
    assert captured == {
        "read_resize": True,
        "use_interpolation": True,
        "effective_date": target,
        "rotation_resize": True,
    }


@pytest.mark.parametrize(
    ("module_name", "processing_name", "processing_result"),
    [
        (
            "coconut_tools.magnetogram.sph_filtering",
            "project_and_reconstruct",
            lambda Br: (Br, np.array([1.0])),
        ),
        (
            "coconut_tools.magnetogram.NLD_implicit_method",
            "filter_radial_field",
            lambda Br: (Br, 1.0),
        ),
        (
            "coconut_tools.magnetogram.Yaroslavsky_filter",
            "filter_radial_field_weighted",
            lambda Br: Br,
        ),
    ],
)
def test_custom_magnetogram_skips_download_and_interpolation(
    monkeypatch,
    tmp_path,
    module_name,
    processing_name,
    processing_result,
):
    module = importlib.import_module(module_name)
    custom_path = tmp_path / "custom.fits"
    Br = np.arange(8.0).reshape(2, 4)
    theta = np.array([0.25, 0.75])
    phi = np.linspace(0.0, 2.0 * np.pi, Br.shape[1], endpoint=False)
    Theta, Phi = np.meshgrid(theta, phi, indexing="ij")
    captured = {}

    def fail_acquisition(*args, **kwargs):
        pytest.fail("A custom magnetogram must not enter an acquisition path.")

    monkeypatch.setattr(module, "generate_output_and_map_names", fail_acquisition)
    monkeypatch.setattr(
        module,
        "generate_output_and_interpolation_map_names",
        fail_acquisition,
    )
    monkeypatch.setattr(module, "read_interpolated_magnetogram", fail_acquisition)

    def fake_read(path, map_type, adapt_map, resize=False):
        captured.update(
            {
                "path": path,
                "map_type": map_type,
                "adapt_map": adapt_map,
                "resize": resize,
            }
        )
        return Br, Theta, Phi

    monkeypatch.setattr(module, "read_magnetogram", fake_read)
    monkeypatch.setattr(
        module,
        "apply_configured_longitude_rotation",
        lambda Br_in, Br_linear, *args, **kwargs: (Br_in, Br_linear, None),
    )
    monkeypatch.setattr(
        module,
        processing_name,
        lambda Br_in, *args, **kwargs: processing_result(Br_in),
    )

    results = module.process_config(
        {
            "date": DATE,
            "custom_magnetogram": custom_path,
            "interpolation": True,
            "resize": True,
            "rotate_to_stonyhurst": False,
            "write_map": False,
            "show_map": False,
        }
    )
    result = results[0]

    assert len(results) == 1
    assert captured == {
        "path": str(custom_path),
        "map_type": "custom",
        "adapt_map": 6,
        "resize": True,
    }
    assert result["local_file"] == str(custom_path)
    assert result["selection"] is None
    assert result["Br_linear"] is None


@pytest.mark.parametrize(
    ("module_name", "filter_name", "filter_result"),
    [
        (
            "coconut_tools.magnetogram.Yaroslavsky_filter",
            "filter_radial_field_weighted",
            lambda Br: Br,
        ),
        (
            "coconut_tools.magnetogram.NLD_implicit_method",
            "filter_radial_field",
            lambda Br: (Br, 0.5),
        ),
    ],
)
@pytest.mark.parametrize("map_type", ["HMI_hourly", "HMI_fdt"])
def test_filtered_pipelines_forward_resize_for_hmi_interpolation(
    monkeypatch,
    tmp_path,
    module_name,
    filter_name,
    filter_result,
    map_type,
):
    module = importlib.import_module(module_name)
    target = datetime(2026, 7, 1, 7, 15)
    local_files = [str(tmp_path / f"{map_type.lower()}_{index}.fits") for index in range(4)]
    selection = object()
    Br = np.arange(8.0).reshape(2, 4)
    Br_linear = Br + 1.0
    theta = np.array([0.25, 0.75])
    phi = np.linspace(0.0, 2.0 * np.pi, Br.shape[1], endpoint=False)
    Theta, Phi = np.meshgrid(theta, phi, indexing="ij")
    captured = {}

    monkeypatch.setattr(
        module,
        "generate_output_and_interpolation_map_names",
        lambda *args, **kwargs: (str(tmp_path / "map.dat"), local_files, selection),
    )
    monkeypatch.setattr(
        module,
        "generate_output_and_map_names",
        lambda *args, **kwargs: pytest.fail(
            f"{map_type} interpolation must not use the single-map path"
        ),
    )

    def fake_read_interpolated(*args, **kwargs):
        captured["read_resize"] = kwargs["resize"]
        return Br, Theta, Phi, Br_linear

    monkeypatch.setattr(module, "read_interpolated_magnetogram", fake_read_interpolated)
    monkeypatch.setattr(module, "magnetogram_effective_date", lambda *args, **kwargs: target)
    monkeypatch.setattr(module, "magnetogram_display_date", lambda *args, **kwargs: target)

    def fake_rotation(Br_in, Br_linear_in, *args, **kwargs):
        captured["use_interpolation"] = args[3]
        captured["rotation_resize"] = kwargs["resize"]
        return Br_in, Br_linear_in, 42.0

    monkeypatch.setattr(module, "apply_configured_longitude_rotation", fake_rotation)
    monkeypatch.setattr(module, filter_name, lambda Br_in, *args, **kwargs: filter_result(Br_in))

    result = module.process_magnetogram_date(
        {
            "date": target.isoformat(),
            "map_type": map_type,
            "output_dir": str(tmp_path),
            "interpolation": True,
            "resize": True,
            "write_map": False,
            "show_map": False,
            "rotate_to_stonyhurst": True,
        },
        target,
    )

    assert result["local_file"] == local_files
    assert result["selection"] is selection
    assert result["rotation_angle"] == pytest.approx(42.0)
    assert captured == {
        "read_resize": True,
        "use_interpolation": True,
        "rotation_resize": True,
    }


def _write_dat_and_png(outdir, name, Br_input, Br_output, Theta, Phi):
    from coconut_tools.magnetogram.io.writers import write_bc_file
    from coconut_tools.magnetogram.visualization.plotting import plot_maps

    dat_path = outdir / f"{name}.dat"
    png_path = outdir / f"{name}.png"

    write_bc_file(str(dat_path), Br_output, Theta[:, 0], Phi[0, :], r_st=1.0)
    plot_maps(
        Br_input,
        Br_output,
        Theta[:, 0],
        Phi[0, :],
        MAP_TYPE,
        "sinlat",
        output_path=str(png_path),
        date=DATE,
    )

    _assert_artifact(dat_path)
    _assert_artifact(png_path)


def _run_three_filters(outdir, prefix, Br, Theta, Phi):
    from coconut_tools.magnetogram.NLD_implicit_method import filter_radial_field
    from coconut_tools.magnetogram.Yaroslavsky_filter import filter_radial_field_weighted
    from coconut_tools.magnetogram.processing.flux_balance import correct_net_flux
    from coconut_tools.magnetogram.processing.spherical_harmonics import (
        project_and_reconstruct,
    )

    Br, Theta, Phi = _center_crop(Br, Theta, Phi)
    Br = correct_net_flux(Br, Theta[:, 0], Phi[0, :], method="surface_mean")

    Br_sph, coefbr = project_and_reconstruct(Br, Theta, Phi, lmax=3, amp=1, alpha=0)
    assert Br_sph.shape == Br.shape
    assert coefbr.size > 0
    _write_dat_and_png(outdir, f"{prefix}_sph", Br, Br_sph, Theta, Phi)

    Br_nld, timestep = filter_radial_field(
        Br,
        Phi[0, :],
        Theta[:, 0],
        iterations=1,
        tau=0.5,
        apply_gaussian=True,
        gaussian_sigma=1.0,
        dx_override=1.0,
        dy_override=1.0,
    )
    assert Br_nld.shape == Br.shape
    assert timestep == 0.5
    _write_dat_and_png(outdir, f"{prefix}_nld", Br, Br_nld, Theta, Phi)

    Br_yaroslavsky = filter_radial_field_weighted(
        Br,
        Phi[0, :],
        Theta[:, 0],
        alpha_factor=1.1,
        Rn=1,
        sig=0.0,
    )
    assert Br_yaroslavsky.shape == Br.shape
    _write_dat_and_png(outdir, f"{prefix}_yaroslavsky", Br, Br_yaroslavsky, Theta, Phi)


@pytest.mark.bigdata
def test_real_gong_single_magnetogram_filters_and_outputs():
    os.environ.setdefault("MPLBACKEND", "Agg")

    outdir = Path(__file__).parent / "_outputs"
    workdir = outdir / "magnetogram" / "single"
    workdir.mkdir(parents=True, exist_ok=True)

    from coconut_tools.magnetogram.io.downloads import (
        generate_output_and_map_names,
        magnetogram_display_date,
        magnetogram_effective_date,
    )
    from coconut_tools.magnetogram.io.readers import read_magnetogram
    from coconut_tools.magnetogram.processing.longitude import (
        apply_configured_longitude_rotation,
    )

    output_name, local_file = generate_output_and_map_names(
        DATE,
        MAP_TYPE,
        str(workdir),
        method_used="sph",
    )

    assert output_name == str(workdir / "map_gong_sph.dat")
    _assert_artifact(Path(local_file))

    Br, Theta, Phi = read_magnetogram(local_file, MAP_TYPE)
    assert Br.ndim == 2
    assert Br.shape == Theta.shape == Phi.shape
    assert np.isfinite(Br).all()

    effective_date = magnetogram_effective_date(local_file, MAP_TYPE, DATE)
    assert effective_date == magnetogram_display_date(local_file, MAP_TYPE, DATE)

    Br, _, rotation_angle = apply_configured_longitude_rotation(
        Br,
        None,
        local_file,
        MAP_TYPE,
        DATE,
        use_interpolation=False,
        rotate_to_stonyhurst=True,
        effective_date=effective_date,
    )
    assert rotation_angle is not None

    _run_three_filters(workdir, "real_gong_single", Br, Theta, Phi)


@pytest.mark.bigdata
def test_real_gong_temporal_interpolation_filters_and_outputs():
    os.environ.setdefault("MPLBACKEND", "Agg")

    outdir = Path(__file__).parent / "_outputs"
    workdir = outdir / "magnetogram" / "temporal"
    rawdir = outdir / "magnetogram" / "raw"
    workdir.mkdir(parents=True, exist_ok=True)
    rawdir.mkdir(parents=True, exist_ok=True)

    from coconut_tools.magnetogram.io.downloads import (
        generate_output_and_interpolation_map_names,
        magnetogram_effective_date,
    )
    from coconut_tools.magnetogram.io.readers import read_interpolated_magnetogram
    from coconut_tools.magnetogram.processing.longitude import (
        apply_configured_longitude_rotation,
    )

    output_name, local_files, selection = generate_output_and_interpolation_map_names(
        DATE,
        MAP_TYPE,
        str(workdir),
        method_used="sph",
        download_dir=str(rawdir),
    )

    assert output_name == str(workdir / "map_gong_sph.dat")
    assert len(local_files) == 4
    for local_file in local_files:
        _assert_artifact(Path(local_file))

    assert selection.target_date == datetime.fromisoformat(DATE)
    assert selection.coef_before + selection.coef_after == pytest.approx(1.0)

    Br, Theta, Phi, Br_linear = read_interpolated_magnetogram(
        local_files,
        MAP_TYPE,
        selection,
        interpolation_order=2,
    )
    assert Br.shape == Theta.shape == Phi.shape == Br_linear.shape
    assert np.isfinite(Br).all()
    assert magnetogram_effective_date(
        local_files[0],
        MAP_TYPE,
        DATE,
        interpolated=True,
    ) == datetime.fromisoformat(DATE)

    Br, Br_linear, rotation_angle = apply_configured_longitude_rotation(
        Br,
        Br_linear,
        local_files,
        MAP_TYPE,
        DATE,
        use_interpolation=True,
        rotate_to_stonyhurst=True,
        effective_date=DATE,
    )
    assert Br_linear.shape == Br.shape
    assert rotation_angle is not None

    _run_three_filters(workdir, "real_gong_temporal", Br, Theta, Phi)

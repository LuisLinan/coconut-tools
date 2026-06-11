from datetime import datetime

import numpy as np
import pytest

from coconut_tools.magnetogram import sph_filtering
from coconut_tools.magnetogram.sph_filtering import magnetogram_display_date


TARGET_DATE = datetime(2020, 12, 7, 15, 0)


@pytest.mark.parametrize(
    ("file_name", "map_type", "expected"),
    [
        (
            "mrzqs201207t1504c2238_181.fits.gz",
            "GONG",
            datetime(2020, 12, 7, 15, 4),
        ),
        (
            "adapt40311_044012_202012071400_i00012600n1.fts.gz",
            "ADAPT",
            datetime(2020, 12, 7, 14, 0),
        ),
    ],
)
def test_magnetogram_display_date_uses_observation_date(file_name, map_type, expected):
    assert magnetogram_display_date(file_name, map_type, TARGET_DATE) == expected


@pytest.mark.parametrize("map_type", ["HMI_small", "HMI_polfil", "WSO"])
def test_magnetogram_display_date_uses_target_date(map_type):
    assert magnetogram_display_date("magnetogram", map_type, TARGET_DATE) == TARGET_DATE


@pytest.mark.parametrize("map_type", ["GONG", "ADAPT"])
def test_magnetogram_display_date_uses_target_date_for_interpolation(map_type):
    assert (
        magnetogram_display_date(
            "magnetogram",
            map_type,
            TARGET_DATE,
            interpolated=True,
        )
        == TARGET_DATE
    )


def test_process_magnetogram_date_plots_selected_gong_observation_date(
    monkeypatch,
    tmp_path,
):
    observed = {}
    Br = np.zeros((2, 4))
    Theta = np.zeros_like(Br)
    Phi = np.zeros_like(Br)

    monkeypatch.setattr(
        sph_filtering,
        "generate_output_and_map_names",
        lambda *args, **kwargs: (
            str(tmp_path / "boundary.dat"),
            str(tmp_path / "mrzqs201207t1504c2238_181.fits.gz"),
        ),
    )
    monkeypatch.setattr(
        sph_filtering,
        "read_magnetogram",
        lambda *args, **kwargs: (Br, Theta, Phi),
    )
    monkeypatch.setattr(
        sph_filtering,
        "project_and_reconstruct",
        lambda *args, **kwargs: (Br, np.array([])),
    )
    monkeypatch.setattr(
        sph_filtering,
        "plot_maps",
        lambda *args, **kwargs: observed.update(date=kwargs["date"]),
    )

    result = sph_filtering.process_magnetogram_date(
        {
            "map_type": "GONG",
            "interpolation": False,
            "rotate_to_stonyhurst": False,
            "write_map": False,
            "show_map": True,
            "output_dir": str(tmp_path),
        },
        TARGET_DATE,
    )

    expected = datetime(2020, 12, 7, 15, 4)
    assert observed["date"] == expected
    assert result["magnetogram_date"] == expected

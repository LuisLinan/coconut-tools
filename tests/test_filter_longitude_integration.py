from datetime import datetime

import numpy as np
import pytest

from coconut_tools.magnetogram import NLD_implicit_method
from coconut_tools.magnetogram import Yaroslavsky_filter


TARGET_DATE = datetime(2020, 12, 7, 15, 0)
MAGNETOGRAM_DATE = datetime(2020, 12, 7, 15, 4)


@pytest.mark.parametrize(
    ("module", "filter_name"),
    [
        (NLD_implicit_method, "filter_radial_field"),
        (Yaroslavsky_filter, "filter_radial_field_weighted"),
    ],
)
def test_filter_pipeline_applies_shared_longitude_processing(
    monkeypatch,
    tmp_path,
    module,
    filter_name,
):
    Br = np.arange(8, dtype=float).reshape(2, 4)
    Theta = np.zeros_like(Br)
    Phi = np.zeros_like(Br)
    rotated = Br + 100.0
    captured = {}

    monkeypatch.setattr(
        module,
        "generate_output_and_map_names",
        lambda *args, **kwargs: (
            str(tmp_path / "boundary.dat"),
            str(tmp_path / "mrzqs201207t1504c2238_181.fits.gz"),
        ),
    )
    monkeypatch.setattr(
        module,
        "read_magnetogram",
        lambda *args, **kwargs: (Br.copy(), Theta, Phi),
    )
    monkeypatch.setattr(
        module,
        "magnetogram_display_date",
        lambda *args, **kwargs: MAGNETOGRAM_DATE,
    )

    def fake_longitude_rotation(*args, **kwargs):
        captured["rotate_to_stonyhurst"] = args[-1]
        return rotated.copy(), None, 60.0

    monkeypatch.setattr(module, "apply_configured_longitude_rotation", fake_longitude_rotation)

    if module is NLD_implicit_method:
        monkeypatch.setattr(
            module,
            filter_name,
            lambda filtered_br, *args, **kwargs: (captured.setdefault("Br", filtered_br), 1.0),
        )
    else:
        monkeypatch.setattr(
            module,
            filter_name,
            lambda filtered_br, *args, **kwargs: captured.setdefault("Br", filtered_br),
        )

    result = module.process_magnetogram_date(
        {
            "map_type": "GONG",
            "interpolation": False,
            "write_map": False,
            "show_map": False,
            "output_dir": str(tmp_path),
        },
        TARGET_DATE,
    )

    np.testing.assert_array_equal(captured["Br"], rotated)
    assert captured["rotate_to_stonyhurst"] is True
    assert result["rotation_angle"] == 60.0
    assert result["magnetogram_date"] == MAGNETOGRAM_DATE

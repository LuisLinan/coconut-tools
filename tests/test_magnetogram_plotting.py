import numpy as np
import pytest

from coconut_tools.magnetogram.sph_filtering import (
    _colorbar_extend,
    _plot_magnetogram_axis,
    _symmetric_color_limit,
)


class RecordingAxis:
    def __init__(self):
        self.calls = []

    def imshow(self, *args, **kwargs):
        self.calls.append(("imshow", args, kwargs))
        return object()

    def pcolormesh(self, *args, **kwargs):
        self.calls.append(("pcolormesh", args, kwargs))
        return object()


def test_symmetric_color_limit_uses_robust_percentile():
    values = np.concatenate([np.full(99, 2.0), [1000.0]])

    limit = _symmetric_color_limit(values)

    assert limit == pytest.approx(np.percentile(np.abs(values), 99.0))
    assert 2.0 < limit < 1000.0


def test_symmetric_color_limit_handles_empty_or_zero_data():
    values = np.array([0.0, np.nan, np.inf, -np.inf])

    assert _symmetric_color_limit(values) == 1.0


def test_colorbar_extend_reports_clipped_symmetric_limits():
    values = np.array([-20.0, -5.0, 0.0, 3.0, 15.0])

    assert _colorbar_extend(values, 10.0) == 'both'
    assert _colorbar_extend(values, 20.0) == 'neither'


def test_native_sine_latitude_grid_uses_imshow():
    axis = RecordingAxis()
    values = np.arange(12.0).reshape(3, 4)
    longitude = np.array([0.0, 90.0, 180.0, 270.0])
    latitude = np.array([90.0, 0.0, -90.0])

    _, ylabel = _plot_magnetogram_axis(
        axis,
        values,
        longitude,
        latitude,
        "sinlat",
        limit=10.0,
    )

    assert ylabel == "Sine Latitude"
    assert [call[0] for call in axis.calls] == ["imshow"]
    args, kwargs = axis.calls[0][1:]
    np.testing.assert_array_equal(args[0], values[::-1])
    assert kwargs["extent"] == pytest.approx([0.0, 270.0, -1.0, 1.0])


def test_true_latitude_grid_uses_pcolormesh_coordinates():
    axis = RecordingAxis()
    values = np.arange(12.0).reshape(3, 4)
    longitude = np.array([0.0, 90.0, 180.0, 270.0])
    latitude = np.array([90.0, 30.0, -90.0])

    _, ylabel = _plot_magnetogram_axis(
        axis,
        values,
        longitude,
        latitude,
        "lat",
        limit=10.0,
    )

    assert ylabel == "Latitude"
    assert [call[0] for call in axis.calls] == ["pcolormesh"]
    args, kwargs = axis.calls[0][1:]
    np.testing.assert_array_equal(args[0], longitude)
    np.testing.assert_array_equal(args[1], latitude)
    np.testing.assert_array_equal(args[2], values)
    assert kwargs["shading"] == "auto"

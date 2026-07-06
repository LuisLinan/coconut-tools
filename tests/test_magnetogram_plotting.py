import numpy as np
import pytest

from coconut_tools.magnetogram.sph_filtering import (
    _colorbar_extend,
    _symmetric_color_limit,
)


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

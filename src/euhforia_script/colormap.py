# This file is part of EUHFORIA.
#
# Copyright 2016, 2017 Jens Pomoell
#
# EUHFORIA is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# EUHFORIA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EUHFORIA. If not, see <http://www.gnu.org/licenses/>.


"""EUHFORIA color maps
"""

import matplotlib.colors

# from matplotlib.colors import LinearSegmentedColormap


_lime = (0.55, 0.85, 0.15)
_lemon = (0.95, 0.95, 0.25)
_orange = (1.00, 0.70, 0.00)
_grapefruit = (1.00, 0.30, 0.18)

#
# Defines the "citrus" colormap
#
_colors_citrus = [
    (0.00, (0.95, 0.00, 1.00)),  # purple
    (0.07, (0.00, 0.25, 1.00)),  # dark blue
    (0.14, (0.00, 0.85, 1.00)),  # light blue/green
    (0.21, _lime),  # lime
    (0.28, _lemon),  # lemon
    (0.35, _orange),  # orange
    (0.43, _grapefruit),  # grapfruit
    (0.50, (1.00, 1.00, 1.00)),  # white
    (0.66, (0.50, 0.66, 0.66)),  # blue/gray
    (0.83, (0.20, 0.33, 0.33)),  # dark blue/gray
    (1.00, (0.00, 0.00, 0.00)),
]  # black

_colors_sunset = [
    (0.00, (0.20, 0.00, 0.30)),  # deep purple
    (0.08, (0.40, 0.00, 0.60)),  # purple
    (0.16, (0.60, 0.10, 0.70)),  # magenta
    (0.24, (0.80, 0.20, 0.60)),  # pink
    (0.32, (0.95, 0.40, 0.40)),  # salmon
    (0.40, (1.00, 0.65, 0.20)),  # orange
    (0.48, (1.00, 0.85, 0.40)),  # light orange
    (0.55, (1.00, 1.00, 1.00)),  # white
    (0.70, (0.70, 0.80, 0.85)),  # light blue gray
    (0.85, (0.35, 0.45, 0.55)),  # dark blue gray
    (1.00, (0.00, 0.00, 0.00)),  # black
]

citrus = matplotlib.colors.LinearSegmentedColormap.from_list("citrus", _colors_citrus)


_colors_citrus_low = [
    (0.00, (0.95, 0.00, 1.00)),  # purple
    (0.14, (0.00, 0.25, 1.00)),  # dark blue
    (0.29, (0.00, 0.85, 1.00)),  # light blue/green
    (0.43, _lime),  # lime
    (0.57, _lemon),  # lemon
    (0.71, _orange),  # orange
    (0.86, _grapefruit),  # grapefruit
    (1.0, (1.00, 1.00, 1.00)),
]

citrus_low = matplotlib.colors.LinearSegmentedColormap.from_list("citrus_low", _colors_citrus_low)

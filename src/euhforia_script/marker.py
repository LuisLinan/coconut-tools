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


"""EUHFORIA marker style definitions
"""

#
# Defines plot marker styles for heliospheric objects.
# Keep both ``Marker`` and ``marker`` keys for backward compatibility with
# older plotting code and example notebooks.
#

_BASE_STYLE = {
    "Mercury": {"marker": "o", "color": (254.0 / 255.0, 240.0 / 255.0, 196.0 / 255.0)},
    "Venus": {"marker": "o", "color": (213.0 / 255.0, 144.0 / 255.0, 47.0 / 255.0)},
    "Earth": {"marker": "o", "color": (0.0 / 255.0, 117.0 / 255.0, 195.0 / 255.0)},
    "Mars": {"marker": "o", "color": (201.0 / 255.0, 34.0 / 255.0, 31.0 / 255.0)},
    "Jupiter": {"marker": "o", "color": (231.0 / 255.0, 196.0 / 255.0, 195.0 / 255.0)},
    "Saturn": {"marker": "o", "color": (200.0 / 255.0, 164.0 / 255.0, 100.0 / 255.0)},
    "Uranus": {"marker": "o", "color": (199.0 / 255.0, 240.0 / 255.0, 243.0 / 255.0)},
    "Neptune": {"marker": "o", "color": (57.0 / 255.0, 68.0 / 255.0, 216.0 / 255.0)},
    "Pluto": {"marker": "o", "color": (231.0 / 255.0, 221.0 / 255.0, 210.0 / 255.0)},
    "STA": {"marker": "s", "color": "r"},
    "STB": {"marker": "s", "color": "b"},
}

style = {
    name: {"marker": attributes["marker"], "Marker": attributes["marker"], "color": attributes["color"]}
    for name, attributes in _BASE_STYLE.items()
}

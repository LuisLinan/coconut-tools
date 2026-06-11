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
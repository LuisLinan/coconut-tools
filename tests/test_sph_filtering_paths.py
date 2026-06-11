from coconut_tools.magnetogram.sph_filtering import resolve_figure_path


DATE = "2020-12-07T15:00:00"


def test_resolve_figure_path_from_directory(tmp_path):
    assert resolve_figure_path(str(tmp_path), "../", "GONG", DATE) == str(
        tmp_path / "gong_20201207150000.png"
    )


def test_resolve_figure_path_replaces_empty_png_name(tmp_path):
    empty_png_name = str(tmp_path / ".png")

    assert resolve_figure_path(empty_png_name, "../", "GONG", DATE) == str(
        tmp_path / "gong_20201207150000.png"
    )


def test_resolve_figure_path_adds_timestamp_for_series(tmp_path):
    figure = str(tmp_path / "gong.png")

    assert resolve_figure_path(
        figure,
        "../",
        "GONG",
        DATE,
        use_unique_name=True,
    ) == str(tmp_path / "gong_20201207150000.png")

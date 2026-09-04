from pathlib import Path


def test_can_import_package():
    import importlib

    module = importlib.import_module("coconut_tools")
    assert module is not None


def test_magnetogram_root_contains_only_public_launch_modules():
    from coconut_tools.magnetogram import sph_filtering

    package_dir = Path(sph_filtering.__file__).parent
    root_modules = {path.name for path in package_dir.glob("*.py")}

    assert root_modules == {
        "NLD_implicit_method.py",
        "Yaroslavsky_filter.py",
        "__init__.py",
        "sph_filtering.py",
    }

# tests/test_imports.py
def test_can_import_package():
    import importlib
    module = importlib.import_module("coconut_tools")
    assert module is not None
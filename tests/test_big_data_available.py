from pathlib import Path
import pytest

@pytest.mark.bigdata
def test_bigdata_files_exist(big_vtu_path: Path, big_cfmesh_path: Path):
    """Ensure that the large VTU and CFmesh files are downloaded and not empty."""
    # Existence
    assert big_vtu_path.exists(), "VTU file was not downloaded"
    assert big_cfmesh_path.exists(), "CFmesh file was not downloaded"

    # Non-empty
    assert big_vtu_path.stat().st_size > 0, "VTU file is empty"
    assert big_cfmesh_path.stat().st_size > 0, "CFmesh file is empty"

    # Sanity check on approximate size
    assert big_vtu_path.stat().st_size > 100_000_000, "VTU file seems too small (<100 MB)"
    assert big_cfmesh_path.stat().st_size > 400_000_000, "CFmesh file seems too small (<400 MB)"

from pathlib import Path
from datetime import datetime
import pytest

@pytest.mark.bigdata
def test_create_boundary_dat(big_cfmesh_path: Path):
    """Test that create_boundary_fromcfmesh produces a non-empty .dat file."""
    # Persistent output directory (not cleaned by pytest)
    outdir = Path(__file__).parent / "_outputs"
    outdir.mkdir(exist_ok=True)

    out_dat = outdir / "test.dat"

    # Import the function under test
    from coconut_tools.toheliosphere.create_dat import create_boundary_fromcfmesh

    # Parameters
    when = datetime.strptime("2024-04-09T05:04:00", "%Y-%m-%dT%H:%M:%S")
    R = 21.5
    ntheta = 180
    nphi = 360
    dr = 0.01

    # Call the function
    res = create_boundary_fromcfmesh(
        str(big_cfmesh_path),
        when,
        R,
        ntheta,
        nphi,
        dr,
        str(out_dat),
        full_output=True,
    )

    # Checks
    assert out_dat.exists(), ".dat file was not created"
    assert out_dat.stat().st_size > 0, ".dat file is empty"

    # Optional: if the function returns a path, validate it as well
    if isinstance(res, (str, Path)):
        res_path = Path(res)
        assert res_path.exists(), "Returned path does not exist"
        assert res_path.stat().st_size > 0, "Returned file is empty"

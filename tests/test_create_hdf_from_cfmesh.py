from pathlib import Path
import pytest

@pytest.mark.bigdata
def test_create_hdf_from_cfmesh(big_cfmesh_path: Path):
    """
    Integration test for create_hdf_from_cfmesh:
    - uses the cached CFmesh dataset folder
    - writes outputs to tests/_outputs/cfmesh
    - asserts that at least one non-empty HDF file is produced
    """
    # Output directory (persistent; we do NOT auto-clean this folder)
    outdir = Path(__file__).parent / "_outputs"
    outdir.mkdir(exist_ok=True)

    # Inputs
    satellite_cartesian = {"STA": [-10.0, 0.0, 0.0]}
    # The function expects a mapping of case_name -> folder; use the cache folder
    cfmesh_glob = str(big_cfmesh_path.parent / "*.CFmesh")
    cfmesh_cases = {"CFMESH_A": cfmesh_glob}


    # Conservative timings (adjust if your API expects different units)
    delta_t = 60.0   # e.g., 60 seconds between samples
    time_step = 1.0  # multiplier or step index, depending on your implementation

    # Import and call the function under test
    from coconut_tools.postprocessing.create_hdf import create_hdf_from_cfmesh

    result = create_hdf_from_cfmesh(
        satellite_cartesian,
        cfmesh_cases,
        outdir,
        delta_t,
        time_step,
    )

    # Assert that at least one HDF file exists and is non-empty
    produced = list(outdir.rglob("*.hdf")) + list(outdir.rglob("*.h5")) + list(outdir.rglob("*.hdf5"))
    assert produced, "No HDF files were produced in the output directory."

    for p in produced:
        assert p.exists(), f"Missing output file: {p}"
        assert p.stat().st_size > 0, f"Output file is empty: {p}"

    # Optional: if the function returns a path or list of paths, validate them too
    if isinstance(result, (str, Path)):
        rp = Path(result)
        assert rp.exists() and rp.stat().st_size > 0, "Returned HDF path does not exist or is empty."
    elif isinstance(result, (list, tuple)) and result:
        for rp in result:
            rp = Path(rp)
            assert rp.exists() and rp.stat().st_size > 0, f"Returned HDF path is invalid or empty: {rp}"

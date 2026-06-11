from pathlib import Path
import os
import pytest

@pytest.mark.bigdata
def test_compare_tomography_with_simulation(big_vtu_path: Path):
    """
    Integration test for tomography pipeline:
    - downloads tomography .dat file if missing
    - runs compare_tomography_with_simulation on a VTU + DAT
    - asserts that the output PDF exists and is non-empty
    """
    # Persistent output directory
    outdir = Path(__file__).parent / "_outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    # Inputs
    date = "20190704"
    altitude = "5-0"
    datfile = outdir / f"tomo_sta_cor2a_{date}_{altitude}.dat"

    # Import the functions under test
    from coconut_tools.plot.tomography import download_tomography_file, compare_tomography_with_simulation

    # Ensure datfile is available
    if not datfile.exists():
        download_tomography_file(date, altitude, str(outdir))

    # Use the cached VTU (big file)
    vtufile = str(big_vtu_path)  # corona-mhd_0.vtu from _bigdata_cache
    output_pdf = outdir / "tomo_5.pdf"

    # Run the comparison
    compare_tomography_with_simulation(vtufile, str(datfile), str(output_pdf))

    # Check output PDF
    assert output_pdf.exists(), "Tomography comparison PDF was not created"
    assert output_pdf.stat().st_size > 0, "Tomography comparison PDF is empty"

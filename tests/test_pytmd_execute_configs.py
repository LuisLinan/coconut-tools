from pathlib import Path
import time
import pytest

@pytest.mark.bigdata
def test_pytmd_execute_configs_creates_cfmesh(big_cfmesh_path: Path):
    """
    Integration test for the pyTDM pipeline:
      - create a configuration with myconfig.createconfig
      - run init_TDm.execute_configs(cfmesh=True)
      - ensure at least one .CFmesh file is generated and non-empty
    All I/O is confined to tests/_outputs/pytmd/.
    """
    # Base folders under tests/_outputs
    out_base = Path(__file__).parent / "_outputs" / "pytmd"
    out_base.mkdir(parents=True, exist_ok=True)

    # Where to save the generated config(s)
    config_dir = out_base / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Where pyTDM should write its outputs (CFmesh, logs, etc.)
    tdm_work_dir = out_base / "tdm_work"
    tdm_work_dir.mkdir(parents=True, exist_ok=True)

    # The input dataset folder: use the cache dir that holds corona.CFmesh (and other inputs)
    bigdata_dir = big_cfmesh_path.parent

    # Record start time to later filter newly produced files
    t0 = time.time()

    # 1) Create the configuration
    from coconut_tools.pyTDM.core_td.myconfig import createconfig

    createconfig(
        path_file=str(bigdata_dir),     # input data folder (cache)
        path_tdm=str(tdm_work_dir),     # working/output folder for pyTDM
        path_save=str(config_dir),      # where to save the config file(s)
        name="Test",
        theta=1.57, phi=3.14, alpha=0,
        D=0.15, A=0.10, R=0.3, delta=0.01,
        zeta=5, case_tdm="first", geometry="spherical", solver="COCONUT",
        flux="Tdm", nb_proc=72, nb_r=200, nb_th=200, nb_phi=200, eps=0.01,
        nfr=100, xc=0.5, xh=0.5, hh_fr=120, ll_fr=35, F_flx=20,
    )

    # Sanity: ensure at least one config file exists in config_dir
    cfg_files = list(config_dir.rglob("*"))
    assert cfg_files, "No configuration files were created in the config directory."

    # 2) Execute configs (ask to generate CFmesh)
    from coconut_tools.pyTDM.init_TDm import execute_configs

    execute_configs(path=str(config_dir), cfmesh=True)

    # 3) Assert at least one .CFmesh was produced after t0 and is non-empty
    produced = [p for p in tdm_work_dir.rglob("*.CFmesh") if p.stat().st_mtime >= t0]
    # If your tool writes .CFmesh into a sibling/subdir, broaden the search to out_base:
    if not produced:
        produced = [p for p in out_base.rglob("*.CFmesh") if p.stat().st_mtime >= t0]

    assert produced, "No .CFmesh file was produced by execute_configs(cfmesh=True)."

    for p in produced:
        assert p.exists(), f"Missing output file: {p}"
        assert p.stat().st_size > 0, f"Output .CFmesh is empty: {p}"

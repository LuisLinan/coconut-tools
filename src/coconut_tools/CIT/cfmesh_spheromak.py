"""Inject a spheromak into a COCONUT CFmesh while preserving file structure."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice, zip_longest
from pathlib import Path
import configparser
import re

import numpy as np
import pyvista as pv
from pyevtk.hl import gridToVTK
from scipy.constants import astronomical_unit
from scipy.spatial import cKDTree
from vtk import VTK_WEDGE

from coconut_tools.CIT.spheromak_model import LocalLFFSpheromak, LocalSpheromakParameters
from coconut_tools.logger_config import setup_logger

logger = setup_logger(__name__)

SOLAR_RADIUS_M = astronomical_unit / 215.0
PROTON_MASS_KG = 1.67e-27
BOLTZMANN_SI = 1.380649e-23
NUMBER_DENSITY_FACTOR = 2.0

DENSITY_CODE_TO_KG_M3 = 1.67e-13
VELOCITY_CODE_TO_M_S = 480248.0
MAGNETIC_CODE_TO_T = 2.2e-4
PRESSURE_CODE_TO_PA = 0.03851

STATE_VALUE_COUNT = 9
STATE_FLOAT_FORMAT = "{:.16e}"
FLOAT_RTOL = 1.0e-12
FLOAT_ATOL = 1.0e-12


@dataclass(frozen=True)
class CFmeshSections:
    """Important section locations and counts inside a CFmesh file.

    Args:
        nb_eq: Number of state values per line.
        nb_nodes: Number of nodes in the mesh.
        nb_states: Number of state lines in the file.
        nb_elements: Number of elements in the mesh.
        element_start_line: Line index of the first connectivity row.
        node_start_line: Line index of the first node row.
        state_start_line: Line index of the first state row.
        state_end_line: Line index of the first `!END` after the state block.
        total_lines: Total number of lines in the file.
    """

    nb_eq: int
    nb_nodes: int
    nb_states: int
    nb_elements: int
    element_start_line: int
    node_start_line: int
    state_start_line: int
    state_end_line: int
    total_lines: int


@dataclass(frozen=True)
class CFmeshData:
    """Numerical CFmesh content required by the injection workflow.

    Args:
        coordinates_rsun: Node coordinates in solar radii.
        connectivity: Six-node prism connectivity for each cell.
        centers_rsun: Cell centers in solar radii.
        states_code: State block in CFmesh code units.
    """

    coordinates_rsun: np.ndarray
    connectivity: np.ndarray
    centers_rsun: np.ndarray
    states_code: np.ndarray


@dataclass(frozen=True)
class SpheromakInsertionConfig:
    """Configuration driving the CFmesh injection workflow.

    Args:
        input_cfmesh: Path to the input CFmesh.
        output_dir: Directory where outputs are written.
        case_name: Prefix used for generated files.
        lat_deg: Eruption latitude in degrees.
        lon_deg: Eruption longitude in degrees.
        radius_rsun: Spheromak radius in solar radii.
        speed_km_s: Bulk speed in kilometers per second.
        mass_density_kg_m3: Uniform mass density inside the spheromak, or `None`
            for automatic ambient scaling.
        temperature_k: Uniform temperature inside the spheromak, or `None`
            for automatic ambient scaling.
        helicity_sign: Helicity sign, typically `+1` or `-1`.
        tilt_deg: Tilt angle in degrees.
        toroidal_flux_wb: Total toroidal magnetic flux in Weber.
        center_radius_rsun: Explicit center radius in solar radii.
        density_factor: Ambient density multiplier when density is automatic.
        temperature_factor: Ambient temperature multiplier when temperature is automatic.
        write_vtu_before: Whether to export the pre-injection VTU.
        write_vtu_after: Whether to export the post-injection VTU.
        write_vts_before: Whether to export the pre-injection structured VTS.
        write_vts_after: Whether to export the post-injection structured VTS.
        vts_nb_r: Radial resolution of the structured VTS.
        vts_nb_theta: Colatitude resolution of the structured VTS.
        vts_nb_phi: Longitude resolution of the structured VTS.
        vts_eps: Angular offset to avoid the poles in the structured VTS.
    """

    input_cfmesh: Path
    output_dir: Path
    case_name: str
    lat_deg: float
    lon_deg: float
    radius_rsun: float
    speed_km_s: float
    mass_density_kg_m3: float | None
    temperature_k: float | None
    helicity_sign: int
    tilt_deg: float
    toroidal_flux_wb: float
    center_radius_rsun: float
    density_factor: float
    temperature_factor: float
    write_vtu_before: bool
    write_vtu_after: bool
    write_vts_before: bool
    write_vts_after: bool
    vts_nb_r: int
    vts_nb_theta: int
    vts_nb_phi: int
    vts_eps: float

    @classmethod
    def from_ini(cls, config_path: str | Path) -> "SpheromakInsertionConfig":
        """Load the workflow configuration from an INI file.

        Args:
            config_path: Path to the configuration file.

        Returns:
            Parsed configuration instance.
        """
        parser = configparser.ConfigParser()
        config_path = Path(config_path)
        if not parser.read(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        return cls(
            input_cfmesh=Path(parser["Paths"]["input_cfmesh"]).expanduser(),
            output_dir=Path(parser["Paths"]["output_dir"]).expanduser(),
            case_name=parser["Paths"]["case_name"].strip(),
            lat_deg=float(parser["Spheromak"]["lat_deg"]),
            lon_deg=float(parser["Spheromak"]["lon_deg"]),
            radius_rsun=float(parser["Spheromak"]["radius_rsun"]),
            speed_km_s=float(parser["Spheromak"]["speed_km_s"]),
            mass_density_kg_m3=_parse_optional_float(parser["Spheromak"]["mass_density_kg_m3"]),
            temperature_k=_parse_optional_float(parser["Spheromak"]["temperature_k"]),
            helicity_sign=int(parser["Spheromak"]["helicity_sign"]),
            tilt_deg=float(parser["Spheromak"]["tilt_deg"]),
            toroidal_flux_wb=float(parser["Spheromak"]["toroidal_flux_wb"]),
            center_radius_rsun=float(parser["Placement"]["center_radius_rsun"]),
            density_factor=float(parser["Plasma"]["density_factor"]),
            temperature_factor=float(parser["Plasma"]["temperature_factor"]),
            write_vtu_before=parser["Visualization"].getboolean("write_vtu_before"),
            write_vtu_after=parser["Visualization"].getboolean("write_vtu_after"),
            write_vts_before=parser["Visualization"].getboolean("write_vts_before"),
            write_vts_after=parser["Visualization"].getboolean("write_vts_after", fallback=False),
            vts_nb_r=int(parser["Visualization"]["vts_nb_r"]),
            vts_nb_theta=int(parser["Visualization"]["vts_nb_theta"]),
            vts_nb_phi=int(parser["Visualization"]["vts_nb_phi"]),
            vts_eps=float(parser["Visualization"]["vts_eps"]),
        )

    @property
    def case_output_dir(self) -> Path:
        """Return the output directory specific to the case."""
        return self.output_dir / self.case_name

    @property
    def output_cfmesh_path(self) -> Path:
        """Return the generated CFmesh path."""
        return self.case_output_dir / f"{self.case_name}_spheromak.CFmesh"

    @property
    def before_vtu_path(self) -> Path:
        """Return the pre-injection VTU path."""
        return self.case_output_dir / f"{self.case_name}_before_volume.vtu"

    @property
    def after_vtu_path(self) -> Path:
        """Return the post-injection VTU path."""
        return self.case_output_dir / f"{self.case_name}_after_volume.vtu"

    @property
    def before_vts_path(self) -> Path:
        """Return the pre-injection VTS path."""
        return self.case_output_dir / f"{self.case_name}_before_structured.vts"

    @property
    def after_vts_path(self) -> Path:
        """Return the post-injection VTS path."""
        return self.case_output_dir / f"{self.case_name}_after_structured.vts"

    @property
    def radius_m(self) -> float:
        """Return the spheromak radius in meters."""
        return self.radius_rsun * SOLAR_RADIUS_M

    @property
    def speed_m_s(self) -> float:
        """Return the bulk speed in meters per second."""
        return self.speed_km_s * 1.0e3

    @property
    def direction_vector(self) -> np.ndarray:
        """Return the radial propagation direction."""
        lat_rad = np.deg2rad(self.lat_deg)
        lon_rad = np.deg2rad(self.lon_deg)
        return np.array(
            [
                np.cos(lat_rad) * np.cos(lon_rad),
                np.cos(lat_rad) * np.sin(lon_rad),
                np.sin(lat_rad),
            ],
            dtype=np.float64,
        )

    def center_position_m(self) -> np.ndarray:
        """Return the center position used for the current injection.

        Returns:
            Cartesian center coordinates in meters.
        """
        return self.center_radius_rsun * SOLAR_RADIUS_M * self.direction_vector


@dataclass(frozen=True)
class SpheromakInjectionResult:
    """Result of the injection workflow.

    Args:
        output_cfmesh: Path to the generated CFmesh.
        before_vtu: Path to the pre-injection VTU, if written.
        after_vtu: Path to the post-injection VTU, if written.
        before_vts: Path to the pre-injection VTS, if written.
        after_vts: Path to the post-injection VTS, if written.
        modified_cell_count: Number of modified state lines.
    """

    output_cfmesh: Path
    before_vtu: Path | None
    after_vtu: Path | None
    before_vts: Path | None
    after_vts: Path | None
    modified_cell_count: int


def create_example_config(config_path: str | Path) -> Path:
    """Create a ready-to-edit example configuration file.

    Args:
        config_path: Destination path of the INI file.

    Returns:
        Path to the created configuration file.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    parser = configparser.ConfigParser()
    parser["Paths"] = {
        "input_cfmesh": "tests/local_test/corona.CFmesh",
        "output_dir": "tests/_outputs/cit",
        "case_name": "spheromak_test",
    }
    parser["Spheromak"] = {
        "lat_deg": "0.0",
        "lon_deg": "0.0",
        "radius_rsun": "5.0",
        "speed_km_s": "800.0",
        "mass_density_kg_m3": "auto",
        "temperature_k": "auto",
        "helicity_sign": "1",
        "tilt_deg": "0.0",
        "toroidal_flux_wb": "1.0e14",
    }
    parser["Placement"] = {
        "center_radius_rsun": "10.0",
    }
    parser["Plasma"] = {
        "density_factor": "1.5",
        "temperature_factor": "1.5",
    }
    parser["Visualization"] = {
        "write_vtu_before": "true",
        "write_vtu_after": "true",
        "write_vts_before": "true",
        "write_vts_after": "true",
        "vts_nb_r": "200",
        "vts_nb_theta": "200",
        "vts_nb_phi": "200",
        "vts_eps": "0.01",
    }

    with config_path.open("w", encoding="utf-8", newline="\n") as handle:
        parser.write(handle)

    logger.info("Example configuration written to %s", config_path)
    return config_path


def apply_spheromak_to_cfmesh(config_path: str | Path) -> SpheromakInjectionResult:
    """Inject a spheromak into a CFmesh and write validation outputs.

    Args:
        config_path: Path to the INI configuration file.

    Returns:
        Result dataclass containing generated paths and the modified cell count.
    """
    config = SpheromakInsertionConfig.from_ini(config_path)
    config.case_output_dir.mkdir(parents=True, exist_ok=True)

    sections = scan_cfmesh_sections(config.input_cfmesh)
    logger.info("Scanning complete: %s states, %s elements", sections.nb_states, sections.nb_elements)

    cfmesh_data = load_cfmesh_geometry_and_states(config.input_cfmesh, sections)
    logger.info("Loaded geometry and state block from %s", config.input_cfmesh)

    center_m = config.center_position_m()
    centers_m = cfmesh_data.centers_rsun * SOLAR_RADIUS_M
    distances_m = np.linalg.norm(centers_m - center_m, axis=1)
    modified_mask = distances_m <= config.radius_m
    modified_count = int(np.count_nonzero(modified_mask))
    if modified_count == 0:
        raise ValueError("The configured spheromak does not intersect any CFmesh cell centers.")

    _validate_unit_round_trip(cfmesh_data.states_code, modified_mask)

    if config.write_vtu_before:
        write_unstructured_vtu(
            config.before_vtu_path,
            cfmesh_data.coordinates_rsun,
            cfmesh_data.connectivity,
            cfmesh_data.states_code,
            modified_mask,
        )

    if config.write_vts_before:
        write_structured_vts(
            config.before_vts_path,
            cfmesh_data.centers_rsun,
            cfmesh_data.states_code,
            modified_mask,
            config,
        )

    ambient_density_kg_m3, ambient_temperature_k = estimate_ambient_plasma(
        distances_m=distances_m,
        modified_mask=modified_mask,
        states_code=cfmesh_data.states_code,
        radius_m=config.radius_m,
    )

    target_density_kg_m3 = config.mass_density_kg_m3
    if target_density_kg_m3 is None:
        target_density_kg_m3 = ambient_density_kg_m3 * config.density_factor

    target_temperature_k = config.temperature_k
    if target_temperature_k is None:
        target_temperature_k = ambient_temperature_k * config.temperature_factor

    target_pressure_pa = mass_density_and_temperature_to_pressure(
        mass_density_kg_m3=target_density_kg_m3,
        temperature_k=target_temperature_k,
    )

    _print_injected_spheromak_characteristics(
        config=config,
        target_density_kg_m3=target_density_kg_m3,
        target_temperature_k=target_temperature_k,
    )

    spheromak = LocalLFFSpheromak(
        LocalSpheromakParameters(
            radius_m=config.radius_m,
            lat_deg=config.lat_deg,
            lon_deg=config.lon_deg,
            tilt_deg=config.tilt_deg,
            helicity_sign=float(config.helicity_sign),
            toroidal_flux_wb=config.toroidal_flux_wb,
            center_m=center_m,
        )
    )

    bulk_velocity_m_s = spheromak.bulk_velocity_vector(config.speed_m_s)
    magnetic_field_t = spheromak.magnetic_field_cartesian(centers_m[modified_mask])

    cfmesh_data.states_code[modified_mask, 0] = target_density_kg_m3 / DENSITY_CODE_TO_KG_M3
    cfmesh_data.states_code[modified_mask, 1:4] = bulk_velocity_m_s / VELOCITY_CODE_TO_M_S
    cfmesh_data.states_code[modified_mask, 4:7] = magnetic_field_t / MAGNETIC_CODE_TO_T
    cfmesh_data.states_code[modified_mask, 7] = target_pressure_pa / PRESSURE_CODE_TO_PA

    write_modified_cfmesh(
        input_cfmesh=config.input_cfmesh,
        output_cfmesh=config.output_cfmesh_path,
        sections=sections,
        modified_mask=modified_mask,
        modified_states_code=cfmesh_data.states_code,
    )
    validate_generated_cfmesh(
        input_cfmesh=config.input_cfmesh,
        output_cfmesh=config.output_cfmesh_path,
        modified_mask=modified_mask,
        source_sections=sections,
    )

    if config.write_vtu_after:
        write_unstructured_vtu(
            config.after_vtu_path,
            cfmesh_data.coordinates_rsun,
            cfmesh_data.connectivity,
            cfmesh_data.states_code,
            modified_mask,
        )

    if config.write_vts_after:
        write_structured_vts(
            config.after_vts_path,
            cfmesh_data.centers_rsun,
            cfmesh_data.states_code,
            modified_mask,
            config,
        )

    logger.info("Injection complete: %s cells modified", modified_count)
    return SpheromakInjectionResult(
        output_cfmesh=config.output_cfmesh_path,
        before_vtu=config.before_vtu_path if config.write_vtu_before else None,
        after_vtu=config.after_vtu_path if config.write_vtu_after else None,
        before_vts=config.before_vts_path if config.write_vts_before else None,
        after_vts=config.after_vts_path if config.write_vts_after else None,
        modified_cell_count=modified_count,
    )


def _print_injected_spheromak_characteristics(
    config: SpheromakInsertionConfig,
    target_density_kg_m3: float,
    target_temperature_k: float,
) -> None:
    """Print the physical characteristics of the injected spheromak."""
    print("Injected spheromak characteristics:")
    print(
        "Lat      Lon       Radius  Speed   Density   Temp.  "
        "Hel. sign   Tilt angle  Flux"
    )
    print(
        "#        [deg HEEQ] [deg HEEQ] [RSun]  [km/s]  [kg/m^3]  [K]      "
        "[+- 1]     [deg]      [Wb]"
    )
    print(
        f"{config.lat_deg:7.2f}  "
        f"{config.lon_deg:8.2f}  "
        f"{config.radius_rsun:6.3f}  "
        f"{config.speed_km_s:6.1f}  "
        f"{target_density_kg_m3:.3e}  "
        f"{target_temperature_k:.3e}  "
        f"{config.helicity_sign:9d}  "
        f"{config.tilt_deg:10.2f}  "
        f"{config.toroidal_flux_wb:.3e}"
    )


def scan_cfmesh_sections(input_cfmesh: str | Path) -> CFmeshSections:
    """Scan a CFmesh file and return its critical section boundaries.

    Args:
        input_cfmesh: Path to the CFmesh file.

    Returns:
        Dataclass describing the block boundaries.
    """
    input_cfmesh = Path(input_cfmesh)
    nb_eq = nb_nodes = nb_states = nb_elements = 0
    element_start_line = node_start_line = state_start_line = -1
    state_end_line = -1
    total_lines = 0

    nb_eq_pattern = re.compile(r"!NB_EQ\s+(\d+)")
    nb_nodes_pattern = re.compile(r"!NB_NODES\s+(\d+)")
    nb_states_pattern = re.compile(r"!NB_STATES\s+(\d+)")
    nb_elements_pattern = re.compile(r"!NB_ELEM\s+(\d+)")

    with input_cfmesh.open("r", encoding="utf-8", newline="") as handle:
        for line_idx, line in enumerate(handle):
            total_lines = line_idx + 1
            if line.startswith("!NB_EQ "):
                nb_eq = int(nb_eq_pattern.search(line).group(1))
            elif line.startswith("!NB_NODES "):
                nb_nodes = int(nb_nodes_pattern.search(line).group(1))
            elif line.startswith("!NB_STATES "):
                nb_states = int(nb_states_pattern.search(line).group(1))
            elif line.startswith("!NB_ELEM "):
                nb_elements = int(nb_elements_pattern.search(line).group(1))
            elif line.startswith("!LIST_ELEM"):
                element_start_line = line_idx + 1
            elif line.startswith("!LIST_NODE"):
                node_start_line = line_idx + 1
            elif line.startswith("!LIST_STATE 1"):
                state_start_line = line_idx + 1
            elif state_start_line >= 0 and state_end_line < 0 and line.startswith("!END"):
                state_end_line = line_idx

    if nb_eq != STATE_VALUE_COUNT:
        raise ValueError(f"Unsupported CFmesh state width: expected 9, found {nb_eq}.")
    if min(element_start_line, node_start_line, state_start_line, state_end_line) < 0:
        raise ValueError(f"Failed to identify the expected CFmesh sections in {input_cfmesh}.")

    return CFmeshSections(
        nb_eq=nb_eq,
        nb_nodes=nb_nodes,
        nb_states=nb_states,
        nb_elements=nb_elements,
        element_start_line=element_start_line,
        node_start_line=node_start_line,
        state_start_line=state_start_line,
        state_end_line=state_end_line,
        total_lines=total_lines,
    )


def load_cfmesh_geometry_and_states(
    input_cfmesh: str | Path,
    sections: CFmeshSections,
) -> CFmeshData:
    """Load cell centers and the state block from a CFmesh file.

    Args:
        input_cfmesh: Path to the CFmesh file.
        sections: Section boundaries previously scanned.

    Returns:
        Parsed CFmesh numerical content.
    """
    connectivity = _loadtxt_block(
        input_cfmesh,
        start_line=sections.element_start_line,
        row_count=sections.nb_elements,
        dtype=np.int32,
        usecols=(0, 1, 2, 3, 4, 5),
    )
    coordinates = _loadtxt_block(
        input_cfmesh,
        start_line=sections.node_start_line,
        row_count=sections.nb_nodes,
        dtype=np.float64,
        usecols=(0, 1, 2),
    )
    states_code = _loadtxt_block(
        input_cfmesh,
        start_line=sections.state_start_line,
        row_count=sections.nb_states,
        dtype=np.float64,
        usecols=tuple(range(STATE_VALUE_COUNT)),
    )

    center_x = sum(coordinates[connectivity[:, i], 0] for i in range(6)) / 6.0
    center_y = sum(coordinates[connectivity[:, i], 1] for i in range(6)) / 6.0
    center_z = sum(coordinates[connectivity[:, i], 2] for i in range(6)) / 6.0
    centers_rsun = np.column_stack((center_x, center_y, center_z))

    return CFmeshData(
        coordinates_rsun=coordinates,
        connectivity=connectivity,
        centers_rsun=centers_rsun,
        states_code=states_code,
    )


def estimate_ambient_plasma(
    distances_m: np.ndarray,
    modified_mask: np.ndarray,
    states_code: np.ndarray,
    radius_m: float,
) -> tuple[float, float]:
    """Estimate ambient density and temperature around the insertion region.

    Args:
        distances_m: Distance to the spheromak center for each cell center.
        modified_mask: Mask of cells located inside the spheromak.
        states_code: State block in CFmesh code units.
        radius_m: Spheromak radius in meters.

    Returns:
        Tuple `(ambient_density_kg_m3, ambient_temperature_k)`.
    """
    outside_mask = ~modified_mask
    shell_mask = outside_mask & (distances_m <= 1.5 * radius_m)
    candidate_mask = shell_mask

    if not np.any(candidate_mask):
        candidate_mask = outside_mask
    if not np.any(candidate_mask):
        candidate_mask = np.ones_like(modified_mask, dtype=bool)

    density_kg_m3 = states_code[:, 0] * DENSITY_CODE_TO_KG_M3
    temperature_k = code_pressure_and_density_to_temperature(states_code[:, 0], states_code[:, 7])

    ambient_density_kg_m3 = float(np.median(density_kg_m3[candidate_mask]))
    ambient_temperature_k = float(np.median(temperature_k[candidate_mask]))
    return ambient_density_kg_m3, ambient_temperature_k


def mass_density_and_temperature_to_pressure(
    mass_density_kg_m3: float,
    temperature_k: float,
) -> float:
    """Convert mass density and temperature to gas pressure.

    Args:
        mass_density_kg_m3: Mass density in kilograms per cubic meter.
        temperature_k: Temperature in Kelvin.

    Returns:
        Pressure in Pascal.
    """
    number_density = mass_density_kg_m3 / PROTON_MASS_KG
    return number_density * NUMBER_DENSITY_FACTOR * BOLTZMANN_SI * temperature_k


def code_pressure_and_density_to_temperature(
    density_code: np.ndarray,
    pressure_code: np.ndarray,
) -> np.ndarray:
    """Convert CFmesh density/pressure code units into Kelvin.

    Args:
        density_code: Density in CFmesh code units.
        pressure_code: Pressure in CFmesh code units.

    Returns:
        Temperature in Kelvin.
    """
    mass_density = density_code * DENSITY_CODE_TO_KG_M3
    number_density = mass_density / PROTON_MASS_KG
    pressure_pa = pressure_code * PRESSURE_CODE_TO_PA
    denominator = np.maximum(number_density * NUMBER_DENSITY_FACTOR * BOLTZMANN_SI, 1.0e-30)
    return pressure_pa / denominator


def write_modified_cfmesh(
    input_cfmesh: str | Path,
    output_cfmesh: str | Path,
    sections: CFmeshSections,
    modified_mask: np.ndarray,
    modified_states_code: np.ndarray,
) -> None:
    """Write the new CFmesh while preserving every untouched line verbatim.

    Args:
        input_cfmesh: Path to the original CFmesh.
        output_cfmesh: Destination path for the modified CFmesh.
        sections: Pre-scanned CFmesh section boundaries.
        modified_mask: Boolean mask of modified state rows.
        modified_states_code: Final state block in CFmesh code units.
    """
    input_cfmesh = Path(input_cfmesh)
    output_cfmesh = Path(output_cfmesh)
    output_cfmesh.parent.mkdir(parents=True, exist_ok=True)
    newline = detect_newline(input_cfmesh)

    with input_cfmesh.open("r", encoding="utf-8", newline="") as source, output_cfmesh.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as target:
        state_idx = 0
        for line_idx, line in enumerate(source):
            if line_idx < sections.state_start_line or line_idx >= sections.state_end_line:
                target.write(line)
                continue

            if modified_mask[state_idx]:
                target.write(format_state_row(modified_states_code[state_idx], newline=newline))
            else:
                target.write(line)
            state_idx += 1


def validate_generated_cfmesh(
    input_cfmesh: str | Path,
    output_cfmesh: str | Path,
    modified_mask: np.ndarray,
    source_sections: CFmeshSections,
) -> None:
    """Validate that the generated CFmesh preserved the original structure.

    Args:
        input_cfmesh: Path to the original CFmesh.
        output_cfmesh: Path to the generated CFmesh.
        modified_mask: Mask of modified state rows.
        source_sections: Pre-scanned sections of the original file.
    """
    output_sections = scan_cfmesh_sections(output_cfmesh)
    if output_sections != source_sections:
        raise ValueError(
            "The generated CFmesh structure differs from the input structure: "
            f"{source_sections} != {output_sections}"
        )

    with Path(input_cfmesh).open("r", encoding="utf-8", newline="") as source, Path(output_cfmesh).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as generated:
        for line_idx, (source_line, generated_line) in enumerate(zip_longest(source, generated, fillvalue=None)):
            if source_line is None or generated_line is None:
                raise ValueError("The input and output CFmesh files do not have the same number of lines.")

            if line_idx < source_sections.state_start_line or line_idx >= source_sections.state_end_line:
                if source_line != generated_line:
                    raise ValueError(f"Unexpected non-state difference detected at line {line_idx + 1}.")
                continue

            state_idx = line_idx - source_sections.state_start_line
            if not modified_mask[state_idx] and source_line != generated_line:
                raise ValueError(
                    f"An untouched state line changed unexpectedly at line {line_idx + 1}."
                )
            if modified_mask[state_idx]:
                validate_state_line(generated_line)


def write_unstructured_vtu(
    output_path: str | Path,
    coordinates_rsun: np.ndarray,
    connectivity: np.ndarray,
    states_code: np.ndarray,
    modified_mask: np.ndarray,
) -> Path:
    """Write a volumetric VTU using the original prism connectivity.

    Args:
        output_path: Destination VTU path.
        coordinates_rsun: Node coordinates in solar radii.
        connectivity: Six-node prism connectivity.
        states_code: State block in CFmesh code units.
        modified_mask: Mask of cells inside the spheromak region.

    Returns:
        Written VTU path with geometry in solar radii and fields in physical units.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cell_count = connectivity.shape[0]
    cells = np.empty((cell_count, 7), dtype=np.int64)
    cells[:, 0] = 6
    cells[:, 1:] = connectivity.astype(np.int64)
    celltypes = np.full(cell_count, VTK_WEDGE, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells.ravel(), celltypes, coordinates_rsun)
    fields = extract_physical_fields(states_code)
    for name, value in fields.items():
        grid.cell_data[name] = value
    grid.cell_data["B_cart"] = np.column_stack((fields["Bx"], fields["By"], fields["Bz"]))
    grid.cell_data["V_cart"] = np.column_stack((fields["Vx"], fields["Vy"], fields["Vz"]))
    grid.cell_data["inside_spheromak"] = modified_mask.astype(np.uint8)
    grid.save(str(output_path))
    return output_path


def write_structured_vts(
    output_path: str | Path,
    centers_rsun: np.ndarray,
    states_code: np.ndarray,
    modified_mask: np.ndarray,
    config: SpheromakInsertionConfig,
) -> Path:
    """Write a structured VTS representation sampled from the original CFmesh.

    Args:
        output_path: Destination VTS path.
        centers_rsun: Cell centers in solar radii.
        states_code: State block in CFmesh code units.
        modified_mask: Mask of cells inside the spheromak region.
        config: Workflow configuration.

    Returns:
        Written VTS path with geometry in meters and fields in physical units.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    radii = np.linalg.norm(centers_rsun, axis=1)
    r_min = float(np.min(radii))
    r_max = float(np.max(radii))
    r_axis = np.linspace(r_min, r_max, config.vts_nb_r)
    theta_axis = np.linspace(config.vts_eps, np.pi - config.vts_eps, config.vts_nb_theta)
    phi_axis = np.linspace(config.vts_eps, 2.0 * np.pi - config.vts_eps, config.vts_nb_phi)

    grid_r, grid_theta, grid_phi = np.meshgrid(r_axis, theta_axis, phi_axis, indexing="ij")
    grid_x_rsun = grid_r * np.sin(grid_theta) * np.cos(grid_phi)
    grid_y_rsun = grid_r * np.sin(grid_theta) * np.sin(grid_phi)
    grid_z_rsun = grid_r * np.cos(grid_theta)

    sample_points = np.column_stack((grid_x_rsun.ravel(), grid_y_rsun.ravel(), grid_z_rsun.ravel()))
    nearest_idx = cKDTree(centers_rsun).query(sample_points)[1]

    grid_x = grid_x_rsun * SOLAR_RADIUS_M
    grid_y = grid_y_rsun * SOLAR_RADIUS_M
    grid_z = grid_z_rsun * SOLAR_RADIUS_M

    fields = extract_physical_fields(states_code)
    bx = np.asfortranarray(fields["Bx"][nearest_idx].reshape(grid_x.shape))
    by = np.asfortranarray(fields["By"][nearest_idx].reshape(grid_x.shape))
    bz = np.asfortranarray(fields["Bz"][nearest_idx].reshape(grid_x.shape))
    vx = np.asfortranarray(fields["Vx"][nearest_idx].reshape(grid_x.shape))
    vy = np.asfortranarray(fields["Vy"][nearest_idx].reshape(grid_x.shape))
    vz = np.asfortranarray(fields["Vz"][nearest_idx].reshape(grid_x.shape))
    rho = np.asfortranarray(fields["rho"][nearest_idx].reshape(grid_x.shape))
    pressure = np.asfortranarray(fields["P"][nearest_idx].reshape(grid_x.shape))
    temperature = np.asfortranarray(fields["T"][nearest_idx].reshape(grid_x.shape))
    inside = np.asfortranarray(modified_mask.astype(np.uint8)[nearest_idx].reshape(grid_x.shape))

    point_data = {
        "Bx": bx,
        "By": by,
        "Bz": bz,
        "Vx": vx,
        "Vy": vy,
        "Vz": vz,
        "rho": rho,
        "P": pressure,
        "T": temperature,
        "inside_spheromak": inside,
        "B_cart": (bx, by, bz),
        "V_cart": (vx, vy, vz),
    }

    base_path = output_path.with_suffix("")
    gridToVTK(str(base_path), grid_x, grid_y, grid_z, pointData=point_data)
    return base_path.with_suffix(".vts")


def extract_physical_fields(states_code: np.ndarray) -> dict[str, np.ndarray]:
    """Convert a CFmesh state block from code units to physical units.

    Args:
        states_code: State block in CFmesh code units.

    Returns:
        Mapping of field name to physical array.
    """
    density_kg_m3 = states_code[:, 0] * DENSITY_CODE_TO_KG_M3
    pressure_pa = states_code[:, 7] * PRESSURE_CODE_TO_PA
    temperature_k = code_pressure_and_density_to_temperature(states_code[:, 0], states_code[:, 7])
    return {
        "rho": density_kg_m3,
        "Vx": states_code[:, 1] * VELOCITY_CODE_TO_M_S,
        "Vy": states_code[:, 2] * VELOCITY_CODE_TO_M_S,
        "Vz": states_code[:, 3] * VELOCITY_CODE_TO_M_S,
        "Bx": states_code[:, 4] * MAGNETIC_CODE_TO_T,
        "By": states_code[:, 5] * MAGNETIC_CODE_TO_T,
        "Bz": states_code[:, 6] * MAGNETIC_CODE_TO_T,
        "P": pressure_pa,
        "T": temperature_k,
    }


def format_state_row(values: np.ndarray, newline: str) -> str:
    """Format a state row using the repository CFmesh scientific notation.

    Args:
        values: One state row with nine floating-point values.
        newline: Newline sequence to append.

    Returns:
        Formatted line.
    """
    if values.shape != (STATE_VALUE_COUNT,):
        raise ValueError(f"Expected a flat row of {STATE_VALUE_COUNT} values, got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Cannot write non-finite values to CFmesh.")
    return " ".join(STATE_FLOAT_FORMAT.format(value) for value in values) + newline


def validate_state_line(line: str) -> None:
    """Ensure that a modified state line keeps the expected CFmesh shape.

    Args:
        line: State line to validate.
    """
    values = np.fromstring(line.strip(), sep=" ")
    if values.size != STATE_VALUE_COUNT:
        raise ValueError(f"Expected {STATE_VALUE_COUNT} state values, found {values.size}: {line!r}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Encountered non-finite values in a modified state line.")


def detect_newline(file_path: str | Path) -> str:
    """Detect the newline convention of a text file.

    Args:
        file_path: Path to the file.

    Returns:
        Detected newline sequence, defaulting to `\\n`.
    """
    sample = Path(file_path).read_bytes()[:4096]
    if b"\r\n" in sample:
        return "\r\n"
    return "\n"


def _loadtxt_block(
    input_cfmesh: str | Path,
    start_line: int,
    row_count: int,
    dtype: np.dtype,
    usecols: tuple[int, ...],
) -> np.ndarray:
    """Load a numerical block from the CFmesh without reading the whole file into memory.

    Args:
        input_cfmesh: Path to the CFmesh file.
        start_line: First line index of the block.
        row_count: Number of rows to read.
        dtype: Target NumPy dtype.
        usecols: Columns to read.

    Returns:
        NumPy array containing the selected block.
    """
    with Path(input_cfmesh).open("r", encoding="utf-8", newline="") as handle:
        return np.loadtxt(
            islice(handle, start_line, start_line + row_count),
            dtype=dtype,
            usecols=usecols,
            ndmin=2,
        )


def _validate_unit_round_trip(states_code: np.ndarray, modified_mask: np.ndarray) -> None:
    """Check that unit conversions are internally consistent on untouched rows.

    Args:
        states_code: State block in CFmesh code units.
        modified_mask: Mask of modified rows.
    """
    untouched = np.flatnonzero(~modified_mask)
    if untouched.size == 0:
        untouched = np.arange(states_code.shape[0])

    sample_idx = untouched[: min(16, untouched.size)]
    sample = states_code[sample_idx]

    round_trip = np.empty((sample.shape[0], 8), dtype=np.float64)
    round_trip[:, 0] = (sample[:, 0] * DENSITY_CODE_TO_KG_M3) / DENSITY_CODE_TO_KG_M3
    round_trip[:, 1:4] = (sample[:, 1:4] * VELOCITY_CODE_TO_M_S) / VELOCITY_CODE_TO_M_S
    round_trip[:, 4:7] = (sample[:, 4:7] * MAGNETIC_CODE_TO_T) / MAGNETIC_CODE_TO_T
    round_trip[:, 7] = (sample[:, 7] * PRESSURE_CODE_TO_PA) / PRESSURE_CODE_TO_PA

    if not np.allclose(sample[:, :8], round_trip, rtol=FLOAT_RTOL, atol=FLOAT_ATOL):
        raise ValueError("CFmesh unit conversions are not round-trip consistent.")


def _parse_optional_float(value: str) -> float | None:
    """Parse a float value that may be set to `auto`.

    Args:
        value: Raw value from the configuration.

    Returns:
        Parsed float or `None`.
    """
    text = value.strip().lower()
    if text in {"", "auto", "none"}:
        return None
    return float(text)


if __name__ == "__main__":
    CONFIG_PATH = Path("tests/_outputs/cit/spheromak_example.ini")
    WRITE_EXAMPLE_CONFIG = True

    if WRITE_EXAMPLE_CONFIG:
        create_example_config(CONFIG_PATH)
    else:
        apply_spheromak_to_cfmesh(CONFIG_PATH)

    CONFIG_PATH = Path("tests/_outputs/cit/spheromak_example.ini")
    WRITE_EXAMPLE_CONFIG = False

    if WRITE_EXAMPLE_CONFIG:
        create_example_config(CONFIG_PATH)
    else:
        apply_spheromak_to_cfmesh(CONFIG_PATH)

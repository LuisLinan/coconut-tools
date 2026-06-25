"""Small playground for experiments with downloaded magnetograms."""

from coconut_tools.tools.logger_config import setup_logger
from coconut_tools.magnetogram.magnetogram_download import generate_output_and_map_names


logger = setup_logger(__name__)


if __name__ == "__main__":
    date = "2020-12-07T15:00:00"
    output_dir = r"C:\Users\luisl\Desktop\testmagnetogram"
    map_types = [
        "WSO",
        "GONG_mrzqs",
        "GONG_mrbqs",
        "GONG_mrbqj",
        "GONG_mrmqs",
        "GONG_mrnqs",
        "ADAPT",
        "HMI_small",
        "HMI_polfil",
    ]

    for map_type in map_types:
        try:
            output_name, local_file = generate_output_and_map_names(
                date=date,
                map_type=map_type,
                output_dir=output_dir,
                lmax=None,
            )
            logger.info(
                "Downloaded %s magnetogram to %s; boundary output would be %s",
                map_type,
                local_file,
                output_name,
            )
        except Exception as exc:
            logger.exception("Failed to download %s magnetogram: %s", map_type, exc)

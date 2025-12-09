from os.path import dirname, join
import logging
import sys

# Allow running as script while preserving package imports
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, dirname(dirname(__file__)))
    __package__ = "FLiESANN"

from ECOv002_calval_tables import load_times_locations
from GEOS5FP import GEOS5FP
from FLiESANN import GEOS5FP_INPUTS

logger = logging.getLogger(__name__)

def generate_FLiESANN_GEOS5FP_inputs(
        filename: str = None,
        update_package_data: bool = True) -> None:
    logger.info("Generating FLiES-ANN GEOS-5 FP input table:")

    for item in GEOS5FP_INPUTS:
        logger.info(f"  - {item}")

    # Load sample times and locations
    targets_df = load_times_locations()

    # Create GEOS5FP connection
    GEOS5FP_connection = GEOS5FP()

    # Query for FLiESANN GEOS5FP input variables
    results_df = GEOS5FP_connection.query(
        target_variables=GEOS5FP_INPUTS,
        targets_df=targets_df
    )

    if update_package_data:
        if filename is None:
            filename = join(dirname(__file__), "ECOv002_calval_FLiESANN_GEOS5FP_inputs.csv")

        results_df.to_csv(filename, index=False)

    return results_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_FLiESANN_GEOS5FP_inputs()

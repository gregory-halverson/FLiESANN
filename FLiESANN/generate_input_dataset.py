from typing import Union, List
import sys
from pathlib import Path

# Add parent directory to path to enable proper imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from os.path import abspath, dirname, join

import pandas as pd
from GEOS5FP import GEOS5FP
from ECOv002_calval_tables import load_calval_table
from FLiESANN import generate_FLiESANN_inputs_table, process_FLiESANN_table

import logging

from FLiESANN import load_ECOv002_calval_FLiESANN_inputs

logger = logging.getLogger(__name__)

# Load the calibration/validation table
def generate_input_dataset(
        inputs_df: Union[pd.DataFrame, str] = None,
        regenerate_variables: List[str] = None,
        max_rows: int = None) -> pd.DataFrame:
    if regenerate_variables is not None:
        logger.info(f"Regenerating FLiESANN inputs: {', '.join(regenerate_variables)}")
        inputs_df = load_ECOv002_calval_FLiESANN_inputs

        for var in regenerate_variables:
            if var in inputs_df.columns:
                inputs_df = inputs_df.drop(columns=[var])
    else:
        logger.info("Generating BESS-JPL input dataset from ECOv002 cal/val FLiESANN inputs")
        inputs_df = load_calval_table()

    # Limit to subset for testing if specified
    if max_rows is not None:
        logger.info(f"Limiting to first {max_rows} rows for testing")
        inputs_df = inputs_df.head(max_rows)

    # Ensure `time_UTC` is in datetime format
    inputs_df['time_UTC'] = pd.to_datetime(inputs_df['time_UTC'])
    
    # Rename/add columns to match what generate_FLiES_inputs_table expects
    if 'Site ID' in inputs_df.columns and 'tower' not in inputs_df.columns:
        inputs_df['tower'] = inputs_df['Site ID']
    if 'Lat' in inputs_df.columns and 'lat' not in inputs_df.columns:
        inputs_df['lat'] = inputs_df['Lat']
    if 'Long' in inputs_df.columns and 'lon' not in inputs_df.columns:
        inputs_df['lon'] = inputs_df['Long']
    if 'elevation_m' in inputs_df.columns and 'elevation_km' not in inputs_df.columns:
        inputs_df['elevation_km'] = inputs_df['elevation_m'] / 1000.0

    # Initialize connection for GEOS5FP data
    GEOS5FP_connection = GEOS5FP()

    # Generate FLiES inputs table with atmospheric parameters from GEOS-5 FP
    inputs_df = generate_FLiESANN_inputs_table(
        inputs_df,
        GEOS5FP_connection=GEOS5FP_connection
    )

    # Process with BESS-JPL model
    outputs_df = process_FLiESANN_table(inputs_df)

    inputs_filename = join(abspath(dirname(__file__)), "ECOv002-cal-val-BESS-JPL-inputs.csv")
    outputs_filename = join(abspath(dirname(__file__)), "ECOv002-cal-val-BESS-JPL-outputs.csv")

    # Save the input dataset to a CSV file
    inputs_df.to_csv(inputs_filename, index=False)

    # Save the processed results to a CSV file
    outputs_df.to_csv(outputs_filename, index=False)

    logger.info(f"Processed {len(outputs_df)} records from the full cal/val dataset")
    logger.info(f"input dataset: {inputs_filename}")
    logger.info(f"output dataset: {outputs_filename}")

    return inputs_df

if __name__ == "__main__":
    import sys
    # Check if --test flag is provided
    max_rows = 5 if "--test" in sys.argv else None
    generate_input_dataset(max_rows=max_rows)

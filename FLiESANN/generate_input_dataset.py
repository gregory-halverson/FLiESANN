# Import necessary libraries
import sys
from pathlib import Path

# Add parent directory to path to enable proper imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from GEOS5FP import GEOS5FP
from ECOv002_calval_tables import load_calval_table
from FLiESANN import generate_FLiES_inputs_table

# Load the calibration/validation table
def main():
    calval_df = load_calval_table()

    # Ensure `time_UTC` is in datetime format
    calval_df['time_UTC'] = pd.to_datetime(calval_df['time_UTC'])

    # Initialize connection for GEOS5FP data
    GEOS5FP_connection = GEOS5FP()

    # Generate FLiES inputs table with atmospheric parameters from GEOS-5 FP
    results_df = generate_FLiES_inputs_table(
        calval_df,
        GEOS5FP_connection=GEOS5FP_connection
    )

    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    output_path = script_dir / "ECOv002-cal-val-FLiESANN-inputs.csv"
    
    # Save the processed results to a CSV file
    results_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()




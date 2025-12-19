"""
Quick test with minimal data to verify the broadcasting fix works.
"""
import pandas as pd
from GEOS5FP import GEOS5FP
from NASADEM import NASADEMConnection
from ECOv002_calval_tables import load_calval_table
from FLiESANN import process_FLiESANN_table

# Load the calibration/validation table
calval_df = load_calval_table()

# Ensure `time_UTC` is in datetime format
calval_df['time_UTC'] = pd.to_datetime(calval_df['time_UTC'])

# Take only first 2 rows to speed up testing
calval_df = calval_df.head(2)

print(f"Testing with {len(calval_df)} rows:")
print(calval_df[['time_UTC', 'geometry']].to_string())

# Initialize connections for GEOS5FP and NASADEM data
GEOS5FP_connection = GEOS5FP(download_directory="GEOS5FP_download")
NASADEM_connection = NASADEMConnection(download_directory="NASADEM_download")

# Process the filtered dataset
try:
    results_df = process_FLiESANN_table(
        calval_df,
        GEOS5FP_connection=GEOS5FP_connection,
        NASADEM_connection=NASADEM_connection,
        row_wise=False
    )
    print("\n✓ Success! Processing completed without errors.")
    print(f"Results shape: {results_df.shape}")
    print(f"Results columns: {results_df.columns.tolist()}")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

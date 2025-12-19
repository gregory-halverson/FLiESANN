#!/usr/bin/env python
"""
Test script to verify broadcasting fixes
"""
import numpy as np
import pandas as pd
from GEOS5FP import GEOS5FP
from NASADEM import NASADEMConnection
from ECOv002_calval_tables import load_calval_table
from FLiESANN import process_FLiESANN_table

# Load the calibration/validation table
calval_df = load_calval_table()

# Ensure `time_UTC` is in datetime format
calval_df['time_UTC'] = pd.to_datetime(calval_df['time_UTC'])

# Create a `date_UTC` column by extracting the date from `time_UTC`
calval_df['date_UTC'] = calval_df['time_UTC'].dt.date

# Filter the dataset to only include the first date
first_date = calval_df['date_UTC'].min()
calval_df = calval_df[calval_df['date_UTC'] == first_date]

print(f"Processing {len(calval_df)} rows for date {first_date}")
print(f"Columns: {calval_df.columns.tolist()}")
print(f"Albedo shape: {calval_df.albedo.values.shape}")

# Initialize connections for GEOS5FP and NASADEM data
GEOS5FP_connection = GEOS5FP()
NASADEM_connection = NASADEMConnection()

try:
    # Process the filtered dataset
    results_df = process_FLiESANN_table(
        calval_df,
        GEOS5FP_connection=GEOS5FP_connection,
        NASADEM_connection=NASADEM_connection
    )
    
    print("\n✓ SUCCESS! No broadcasting errors occurred.")
    print(f"Results shape: {results_df.shape}")
    print(f"Results columns: {results_df.columns.tolist()[:10]}...")
    
except ValueError as e:
    if "broadcast" in str(e):
        print(f"\n✗ BROADCASTING ERROR: {e}")
        import traceback
        traceback.print_exc()
    else:
        raise
except Exception as e:
    print(f"\n✗ OTHER ERROR: {e}")
    import traceback
    traceback.print_exc()

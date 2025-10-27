#!/usr/bin/env python3
"""
Test script to verify atmospheric default approach with a small subset of data
"""
import pandas as pd
from GEOS5FP import GEOS5FP
from NASADEM import NASADEMConnection
from ECOv002_calval_tables import load_calval_table
from FLiESANN import process_FLiESANN_table

# Load just a small subset of data
calval_df = load_calval_table()
print(f"Original dataset shape: {calval_df.shape}")

# Take just the first 5 rows for testing
test_df = calval_df.head(5).copy()
print(f"Test dataset shape: {test_df.shape}")

# Ensure time_UTC is in datetime format
test_df['time_UTC'] = pd.to_datetime(test_df['time_UTC'])

# Add default atmospheric parameters to match reference data
test_df['COT'] = 0.0  # Clear sky conditions  
test_df['AOT'] = 0.0  # No aerosols
test_df['vapor_gccm'] = 0.0  # No water vapor
test_df['ozone_cm'] = 0.3  # Constant ozone level

print("Added atmospheric defaults:")
print(test_df[['COT', 'AOT', 'vapor_gccm', 'ozone_cm']].head())

# Test processing with a small subset
print("\nProcessing subset with atmospheric defaults...")

GEOS5FP_connection = GEOS5FP(download_directory="GEOS5FP_download")
NASADEM_connection = NASADEMConnection(download_directory="NASADEM_download")

try:
    results_df = process_FLiESANN_table(
        test_df,  
        GEOS5FP_connection=GEOS5FP_connection,
        NASADEM_connection=NASADEM_connection       
    )
    
    print(f"Successfully processed {len(results_df)} rows")
    
    # Check for NaN values in output
    output_cols = ['SWin_Wm2', 'UV_Wm2', 'PAR_Wm2', 'NIR_Wm2']
    available_output_cols = [col for col in output_cols if col in results_df.columns]
    
    if available_output_cols:
        print("\nOutput summary:")
        for col in available_output_cols:
            nan_count = results_df[col].isna().sum()
            print(f"{col}: {nan_count}/{len(results_df)} NaN values")
            if nan_count == 0:  # If no NaN values, show some sample values
                print(f"  Sample values: {results_df[col].head(3).tolist()}")
    else:
        print("Output columns not found, showing all columns:")
        print(list(results_df.columns))
        
except Exception as e:
    print(f"Error during processing: {e}")
    import traceback
    traceback.print_exc()
#!/usr/bin/env python3
"""
Test FLiES processing with minimal example
"""
import pandas as pd
from ECOv002_calval_tables import load_calval_table

# Load just one row for testing
print("Loading calval data...")
calval_df = load_calval_table()
test_df = calval_df.head(1).copy()  # Just one row
print(f"Test dataset shape: {test_df.shape}")

# Ensure time_UTC is in datetime format
test_df['time_UTC'] = pd.to_datetime(test_df['time_UTC'])

# Add default atmospheric parameters that match reference data
test_df['COT'] = 0.0  
test_df['AOT'] = 0.0  
test_df['vapor_gccm'] = 0.0  
test_df['ozone_cm'] = 0.3  

print("Sample row with atmospheric defaults:")
print("time_UTC:", test_df['time_UTC'].iloc[0])
print("geometry:", test_df['geometry'].iloc[0])  
print("albedo:", test_df['albedo'].iloc[0])
print("COT:", test_df['COT'].iloc[0])
print("AOT:", test_df['AOT'].iloc[0])
print("vapor_gccm:", test_df['vapor_gccm'].iloc[0])
print("ozone_cm:", test_df['ozone_cm'].iloc[0])

# Try to process with FLiES - but use None connections to avoid downloads
print("\nImporting FLiES processing...")
from FLiESANN import process_FLiESANN_table

print("Processing 1 row with None connections to avoid data downloads...")

try:
    # Use None for both connections to avoid downloading external data
    results_df = process_FLiESANN_table(
        test_df,
        GEOS5FP_connection=None,
        NASADEM_connection=None      
    )
    
    print(f"Successfully processed! Results shape: {results_df.shape}")
    
    # Check for key output columns
    output_cols = ['SWin_Wm2', 'UV_Wm2', 'PAR_Wm2', 'NIR_Wm2']
    for col in output_cols:
        if col in results_df.columns:
            val = results_df[col].iloc[0]
            print(f"{col}: {val} (type: {type(val)})")
            
except Exception as e:
    print(f"Error during processing: {e}")
    import traceback
    traceback.print_exc()
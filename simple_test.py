#!/usr/bin/env python3
"""
Simple test without external data connections
"""
import pandas as pd
from ECOv002_calval_tables import load_calval_table

# Load just a small subset of data
print("Loading calval data...")
calval_df = load_calval_table()
print(f"Original dataset shape: {calval_df.shape}")

# Take just the first 2 rows for testing
test_df = calval_df.head(2).copy()
print(f"Test dataset shape: {test_df.shape}")

# Check what columns we have
print("Columns in test data:")
print(list(test_df.columns))

# Add atmospheric defaults that match reference data
test_df['COT'] = 0.0  
test_df['AOT'] = 0.0  
test_df['vapor_gccm'] = 0.0  
test_df['ozone_cm'] = 0.3  

print("\nAdded atmospheric parameters:")
print(test_df[['COT', 'AOT', 'vapor_gccm', 'ozone_cm']])

print("Test completed successfully!")
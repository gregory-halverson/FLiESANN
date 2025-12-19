"""
Debug script to understand the DataFrame structure from GEOS5FP.
"""
import pandas as pd
from GEOS5FP import GEOS5FP
from ECOv002_calval_tables import load_calval_table
from rasters import MultiPoint

# Load data
calval_df = load_calval_table()
calval_df['time_UTC'] = pd.to_datetime(calval_df['time_UTC'])
calval_df = calval_df.head(2)

print("Input data:")
print(calval_df[['time_UTC', 'geometry']].to_string())

# Create MultiPoint
geometries = [row.geometry for _, row in calval_df.iterrows()]
multi_point = MultiPoint(geometries)
times_UTC = pd.to_datetime(calval_df.time_UTC)

# Initialize GEOS5FP
geos5fp = GEOS5FP(download_directory="GEOS5FP_download")

# Retrieve COT
print("\n--- Retrieving COT ---")
COT = geos5fp.COT(time_UTC=times_UTC, geometry=multi_point, resampling="cubic")

print(f"\nCOT type: {type(COT)}")
print(f"COT shape: {COT.shape}")
print(f"\nCOT DataFrame:")
print(COT)
print(f"\nCOT columns: {COT.columns.tolist() if isinstance(COT, pd.DataFrame) else 'N/A'}")
print(f"COT index: {COT.index.tolist() if isinstance(COT, pd.DataFrame) else 'N/A'}")

if isinstance(COT, pd.DataFrame):
    print(f"\nCOT.values type: {type(COT.values)}")
    print(f"COT.values shape: {COT.values.shape}")
    print(f"COT.values:\n{COT.values}")

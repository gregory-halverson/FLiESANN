#!/usr/bin/env python
"""
Debug script to understand GEOS5FP data shapes
"""
import pandas as pd
from GEOS5FP import GEOS5FP
from ECOv002_calval_tables import load_calval_table
import shapely

# Load the calibration/validation table
calval_df = load_calval_table()

# Ensure `time_UTC` is in datetime format
calval_df['time_UTC'] = pd.to_datetime(calval_df['time_UTC'])

# Create a `date_UTC` column
calval_df['date_UTC'] = calval_df['time_UTC'].dt.date

# Filter to first date
first_date = calval_df['date_UTC'].min()
calval_df = calval_df[calval_df['date_UTC'] == first_date]

print(f"Input data: {len(calval_df)} rows")
print(f"Times: {calval_df['time_UTC'].tolist()}")
print(f"Geometries: {calval_df['geometry'].tolist()}")

# Create MultiPoint geometry
geometries = calval_df.geometry.values
times_UTC = calval_df.time_UTC.values

# Create MultiPoint
from shapely.geometry import MultiPoint
multi_point = MultiPoint(geometries)

print(f"\nMultiPoint has {len(multi_point.geoms)} points")

# Initialize GEOS5FP
GEOS5FP_connection = GEOS5FP()

# Retrieve COT
print("\nRetrieving COT...")
COT = GEOS5FP_connection.COT(
    time_UTC=times_UTC,
    geometry=multi_point,
    resampling="nearest"
)

print(f"COT type: {type(COT)}")
if isinstance(COT, pd.DataFrame):
    print(f"COT DataFrame shape: {COT.shape}")
    print(f"COT DataFrame columns: {COT.columns.tolist()}")
    print(f"COT DataFrame index: {COT.index.tolist()}")
    print(f"COT DataFrame:\n{COT}")
    print(f"\nCOT.iloc[:, 0] shape: {COT.iloc[:, 0].shape}")
    print(f"COT.iloc[:, 0] values: {COT.iloc[:, 0].values}")
else:
    print(f"COT shape: {COT.shape if hasattr(COT, 'shape') else 'N/A'}")
    print(f"COT: {COT}")

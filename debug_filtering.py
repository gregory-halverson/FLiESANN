#!/usr/bin/env python
"""
Debug the DataFrame filtering logic
"""
import pandas as pd
import numpy as np
from GEOS5FP import GEOS5FP
from ECOv002_calval_tables import load_calval_table
from shapely.geometry import MultiPoint

# Load data
calval_df = load_calval_table()
calval_df['time_UTC'] = pd.to_datetime(calval_df['time_UTC'])
calval_df['date_UTC'] = calval_df['time_UTC'].dt.date
first_date = calval_df['date_UTC'].min()
calval_df = calval_df[calval_df['date_UTC'] == first_date]

print(f"Input: {len(calval_df)} rows")
print(f"Times: {calval_df['time_UTC'].tolist()}")

geometries = calval_df.geometry.values
times_UTC = calval_df.time_UTC.values
multi_point = MultiPoint(geometries)

print(f"\nMultiPoint: {len(multi_point.geoms)} points")
print(f"Time array: {len(times_UTC)} times")

# Retrieve COT
GEOS5FP_connection = GEOS5FP()
COT = GEOS5FP_connection.COT(time_UTC=times_UTC, geometry=multi_point, resampling="nearest")

print(f"\nCOT DataFrame shape: {COT.shape}")
print(f"COT DataFrame:\n{COT}")

# Test filtering logic
input_coords = [(geom.x, geom.y) for geom in multi_point.geoms]
time_UTC_array = times_UTC

print(f"\nInput coords: {input_coords}")
print(f"Time array: {time_UTC_array}")

unique_times = sorted(set(pd.to_datetime(time_UTC_array)))
print(f"\nUnique times: {unique_times}")

time_to_index = {t: i for i, t in enumerate(unique_times)}
print(f"Time to index: {time_to_index}")

selected_rows = []
for i, (coord, time_val) in enumerate(zip(input_coords, time_UTC_array)):
    time_val = pd.to_datetime(time_val)
    time_idx = time_to_index[time_val]
    row_idx = time_idx * len(input_coords) + i
    print(f"\nInput row {i}: coord={coord}, time={time_val}")
    print(f"  time_idx={time_idx}, row_idx={row_idx}")
    if row_idx < len(COT):
        print(f"  Selected DataFrame row {row_idx}: {COT.iloc[row_idx].tolist()}")
        selected_rows.append(row_idx)

print(f"\nSelected rows: {selected_rows}")
print(f"Expected: [0, 3] (first location at first time, second location at second time)")

#!/usr/bin/env python
"""
Test the actual filtering function
"""
import pandas as pd
import numpy as np
from GEOS5FP import GEOS5FP
from ECOv002_calval_tables import load_calval_table
from shapely.geometry import MultiPoint
import shapely
import rasters as rt

# Load data
calval_df = load_calval_table()
calval_df['time_UTC'] = pd.to_datetime(calval_df['time_UTC'])
calval_df['date_UTC'] = calval_df['time_UTC'].dt.date
first_date = calval_df['date_UTC'].min()
calval_df = calval_df[calval_df['date_UTC'] == first_date]

geometries = calval_df.geometry.values
times_UTC = calval_df.time_UTC.values
multi_point = MultiPoint(geometries)

# Retrieve COT
GEOS5FP_connection = GEOS5FP()
COT = GEOS5FP_connection.COT(time_UTC=times_UTC, geometry=multi_point, resampling="nearest")

print(f"COT DataFrame shape: {COT.shape}")
print(f"COT DataFrame:\n{COT}\n")

# Copy the filtering function
def filter_dataframe_to_location_time_pairs(df, geometry, time_UTC):
    """Filter DataFrame returned from GEOS5FP to match original location-time pairs"""
    print(f"filter_dataframe_to_location_time_pairs called")
    print(f"  df type: {type(df)}, shape: {df.shape if isinstance(df, pd.DataFrame) else 'N/A'}")
    print(f"  geometry type: {type(geometry)}")
    print(f"  geometry has geoms: {hasattr(geometry, 'geoms')}")
    if hasattr(geometry, 'geoms'):
        print(f"  geometry.geoms length: {len(geometry.geoms)}")
    
    if not isinstance(df, pd.DataFrame) or len(df) == len(geometry.geoms):
        print(f"  Condition 1: not DataFrame or len(df)==len(geoms)")
        print(f"    isinstance(df, pd.DataFrame): {isinstance(df, pd.DataFrame)}")
        if isinstance(df, pd.DataFrame) and hasattr(geometry, 'geoms'):
            print(f"    len(df): {len(df)}, len(geometry.geoms): {len(geometry.geoms)}")
            print(f"    len(df) == len(geometry.geoms): {len(df) == len(geometry.geoms)}")
        print(f"  Returning df unchanged")
        return df
    
    # Get coordinates of input geometry points
    if isinstance(geometry, (shapely.geometry.MultiPoint, rt.MultiPoint)):
        input_coords = [(geom.x, geom.y) for geom in geometry.geoms]
    else:
        print(f"  Not a MultiPoint, returning df")
        return df
    
    print(f"  input_coords: {input_coords}")
    
    # Convert time_UTC to array if it's a single value
    if not hasattr(time_UTC, '__len__'):
        time_UTC_array = [time_UTC] * len(input_coords)
    else:
        time_UTC_array = time_UTC
    
    print(f"  time_UTC_array length: {len(time_UTC_array)}")
    
    unique_times = sorted(set(pd.to_datetime(time_UTC_array)))
    time_to_index = {t: i for i, t in enumerate(unique_times)}
    
    selected_rows = []
    for i, (coord, time_val) in enumerate(zip(input_coords, time_UTC_array)):
        time_val = pd.to_datetime(time_val)
        time_idx = time_to_index[time_val]
        row_idx = time_idx * len(input_coords) + i
        if row_idx < len(df):
            selected_rows.append(row_idx)
    
    print(f"  selected_rows: {selected_rows}")
    print(f"  len(selected_rows): {len(selected_rows)}, len(input_coords): {len(input_coords)}")
    
    if len(selected_rows) == len(input_coords):
        result = df.iloc[selected_rows, 0].values.astype(np.float32)
        print(f"  Filtering succeeded, returning {len(result)} values: {result}")
        return result
    else:
        result = df.iloc[:, 0].values.astype(np.float32)
        print(f"  Fallback: returning {len(result)} values: {result}")
        return result

# Test the function
result = filter_dataframe_to_location_time_pairs(COT, multi_point, times_UTC)
print(f"\nFinal result type: {type(result)}")
print(f"Final result shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
print(f"Final result: {result}")

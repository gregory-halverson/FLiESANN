"""
Debug script to understand the shapes being created.
"""
import pandas as pd
import numpy as np
from GEOS5FP import GEOS5FP
from NASADEM import NASADEMConnection
from ECOv002_calval_tables import load_calval_table
from rasters import MultiPoint

# Load the calibration/validation table
calval_df = load_calval_table()

# Ensure `time_UTC` is in datetime format
calval_df['time_UTC'] = pd.to_datetime(calval_df['time_UTC'])

# Take only first 2 rows
calval_df = calval_df.head(2)

print(f"Testing with {len(calval_df)} rows:")
print(calval_df[['time_UTC', 'geometry']].to_string())

# Initialize connections
GEOS5FP_connection = GEOS5FP(download_directory="GEOS5FP_download")
NASADEM_connection = NASADEMConnection(download_directory="NASADEM_download")

# Create MultiPoint geometry from the dataframe
geometries = [row.geometry for _, row in calval_df.iterrows()]
multi_point = MultiPoint(geometries)
print(f"\nMultiPoint has {len(multi_point.geoms)} points")

# Get times
times_UTC = pd.to_datetime(calval_df.time_UTC)
print(f"Times: {times_UTC.tolist()}")

# Test retrieval of one variable
print("\n--- Testing COT retrieval ---")
COT = GEOS5FP_connection.COT(
    time_UTC=times_UTC,
    geometry=multi_point,
    resampling="cubic"
)
print(f"COT type: {type(COT)}")
print(f"COT shape: {COT.shape if hasattr(COT, 'shape') else 'N/A'}")
if hasattr(COT, 'shape'):
    print(f"COT values shape: {COT.shape}")
    print(f"COT min/max: {np.min(COT):.3f} / {np.max(COT):.3f}")

print("\n--- Testing KG_climate retrieval ---")
from koppengeiger import load_koppen_geiger
KG_climate = load_koppen_geiger(geometry=multi_point)
print(f"KG_climate type: {type(KG_climate)}")
print(f"KG_climate shape: {KG_climate.shape if hasattr(KG_climate, 'shape') else 'N/A'}")
if hasattr(KG_climate, 'shape'):
    print(f"KG_climate values: {KG_climate}")

print("\n--- Testing elevation retrieval ---")
elevation_km = NASADEM_connection.elevation_km(geometry=multi_point)
print(f"elevation_km type: {type(elevation_km)}")
print(f"elevation_km shape: {elevation_km.shape if hasattr(elevation_km, 'shape') else 'N/A'}")
if hasattr(elevation_km, 'shape'):
    print(f"elevation_km values: {elevation_km}")

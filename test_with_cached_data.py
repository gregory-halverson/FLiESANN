"""
Test with already downloaded GEOS5FP data to bypass network issues
"""
import pandas as pd
import numpy as np
from GEOS5FP import GEOS5FP
from NASADEM import NASADEMConnection
from ECOv002_calval_tables import load_calval_table
from rasters import MultiPoint
from koppengeiger import load_koppen_geiger

# Load data
calval_df = load_calval_table()
calval_df['time_UTC'] = pd.to_datetime(calval_df['time_UTC'])
calval_df = calval_df.head(2)

print(f"Testing with {len(calval_df)} rows")
geometries = [row.geometry for _, row in calval_df.iterrows()]
multi_point = MultiPoint(geometries)
times_UTC = pd.to_datetime(calval_df.time_UTC)

# Manually create fake data to test broadcasting without network access
print("\n--- Simulating GEOS5FP DataFrame structure ---")
# Simulate what GEOS5FP returns: DataFrame with 4 rows (2 times × 2 points) and 5 columns
fake_COT = pd.DataFrame({
    'COT': [0.1, 0.2, 0.3, 0.4],
    'lat': [35.0, 41.0, 35.0, 41.0],
    'lon': [-76.0, -80.0, -76.0, -80.0],
    'lat_used': [35.0, 41.0, 35.0, 41.0],
    'lon_used': [-76.0, -80.0, -76.0, -80.0]
})

print(f"Fake COT DataFrame shape: {fake_COT.shape}")
print(fake_COT)

# Get KG_climate
print("\n--- Getting KG_climate ---")
KG_climate = load_koppen_geiger(geometry=multi_point)
print(f"KG_climate type: {type(KG_climate)}")
print(f"KG_climate shape: {KG_climate.shape}")
print(f"KG_climate values: {KG_climate}")

# Test the conversion logic
print("\n--- Testing DataFrame conversion ---")
import pandas as pd
if isinstance(fake_COT, pd.DataFrame):
    print("COT is DataFrame, converting...")
    COT_array = fake_COT.iloc[:, 0].values.astype(np.float32)
    print(f"COT array shape after conversion: {COT_array.shape}")
    print(f"COT array values: {COT_array}")

# Test broadcasting
print("\n--- Testing broadcasting ---")
from FLiESANN.ensure_array import ensure_array

try:
    KG_broadcast = ensure_array(KG_climate, COT_array.shape)
    print(f"✓ Success! KG_climate broadcast to shape: {KG_broadcast.shape}")
    print(f"KG_broadcast values: {KG_broadcast}")
except Exception as e:
    print(f"✗ Error: {e}")

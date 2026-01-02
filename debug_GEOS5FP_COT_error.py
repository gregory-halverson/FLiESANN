from datetime import datetime
import numpy as np
from GEOS5FP import GEOS5FP
from rasters import Point  # Use rasters.Point instead of shapely.geometry.Point
import traceback  # Import traceback for detailed error output

# Simulate the inputs
geometry = Point(-118.2437, 34.0522)  # Los Angeles coordinates (longitude, latitude)
time_UTC = datetime(2025, 12, 29, 12, 0, 0)  # Replace with the actual time used
resampling = "cubic"  # Replace with the actual resampling method used

# Initialize the GEOS5FP connection
GEOS5FP_connection = GEOS5FP()

# Retrieve COT
COT = GEOS5FP_connection.COT(
    time_UTC=time_UTC,
    geometry=geometry,
    resampling=resampling
)

# Debugging: Check the type of COT
print(f"Type of COT: {type(COT)}")

# Attempt to clip COT to reproduce the error
try:
    COT = np.clip(COT, 0, None)  # Replace with the actual clipping logic
except Exception as e:
    print("Error during clipping:")
    traceback.print_exc()  # Print the full traceback
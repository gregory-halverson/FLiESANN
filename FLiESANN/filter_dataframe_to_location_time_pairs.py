import numpy as np
import pandas as pd
import shapely
import rasters as rt


def filter_dataframe_to_location_time_pairs(df, geometry, time_UTC):
    """Filter DataFrame or 2D array returned from GEOS5FP to match original location-time pairs"""
    
    # Handle DataFrame
    if isinstance(df, pd.DataFrame):
        if len(df) == len(geometry.geoms):
            return df
            
        # Extract first column (data values) from DataFrame
        data_array = df.iloc[:, 0].values.astype(np.float32)
    # Handle 2D numpy array
    elif isinstance(df, np.ndarray) and len(df.shape) == 2:
        if df.shape[0] == (len(geometry.geoms) if hasattr(geometry, 'geoms') else 0):
            return df
        
        # Extract first column if multiple columns
        if df.shape[1] > 1:
            data_array = df[:, 0].astype(np.float32)
        else:
            data_array = df.flatten().astype(np.float32)
    # Handle 1D array or scalar
    else:
        return df
    
    # Get coordinates of input geometry points
    if isinstance(geometry, (shapely.geometry.MultiPoint, rt.MultiPoint)):
        input_coords = [(geom.x, geom.y) for geom in geometry.geoms]
    else:
        return df
    
    # Convert time_UTC to array if it's a single value
    if not hasattr(time_UTC, '__len__'):
        time_UTC_array = [time_UTC] * len(input_coords)
    else:
        time_UTC_array = time_UTC
    
    # For each input location-time pair, find the matching DataFrame row
    # GEOS5FP processes unique times and returns data for all locations at each time
    # We need to select only the rows that match our specific location-time pairs
    
    # Create a mapping of (lat, lon, time_index) -> row index
    # The DataFrame contains all locations for each unique time
    unique_times = sorted(set(pd.to_datetime(time_UTC_array)))
    time_to_index = {t: i for i, t in enumerate(unique_times)}
    
    selected_rows = []
    for i, (coord, time_val) in enumerate(zip(input_coords, time_UTC_array)):
        time_val = pd.to_datetime(time_val)
        time_idx = time_to_index[time_val]
        # Row index = time_idx * num_locations + location_idx
        row_idx = time_idx * len(input_coords) + i
        if row_idx < len(data_array):
            selected_rows.append(row_idx)
    
    if len(selected_rows) == len(input_coords):
        return data_array[selected_rows].astype(np.float32)
    else:
        # Fallback: return all rows if filtering fails
        return data_array.astype(np.float32)

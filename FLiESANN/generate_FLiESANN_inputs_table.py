import logging

import numpy as np
import pandas as pd
import rasters as rt
from dateutil import parser
from pandas import DataFrame
from rasters import MultiPoint, WGS84
from shapely.geometry import Point
from GEOS5FP import GEOS5FP
from NASADEM import NASADEMConnection
from .retrieve_FLiESANN_inputs import retrieve_FLiESANN_inputs

logger = logging.getLogger(__name__)

def generate_FLiESANN_inputs_table(
        input_df: DataFrame,
        GEOS5FP_connection: GEOS5FP = None,
        NASADEM_connection: NASADEMConnection = None) -> DataFrame:
    """
    Generates a DataFrame of FLiES inputs by retrieving atmospheric and static data.
    
    This is a simple wrapper around retrieve_FLiESANN_inputs that handles DataFrame
    input/output and geometry parsing.

    Parameters:
    input_df (pd.DataFrame): A DataFrame containing the following columns:
        - time_UTC (str or datetime): Time in UTC.
        - geometry (str or shapely.geometry.Point) or (lat, lon): Spatial coordinates. 
          If "geometry" is a string, it should be in WKT format (e.g., "POINT (lon lat)").
        - albedo (float, optional): Surface albedo.
        - COT (float, optional): Cloud optical thickness.
        - AOT (float, optional): Aerosol optical thickness.
        - vapor_gccm (float, optional): Water vapor in grams per cubic centimeter.
        - ozone_cm (float, optional): Ozone concentration in centimeters.
        - elevation_m (float, optional): Elevation in meters.
        - SZA_deg (float, optional): Solar zenith angle in degrees.
        - KG or KG_climate (str, optional): Köppen-Geiger climate classification.
        - SWin_Wm2 (float, optional): Shortwave incoming solar radiation.
        - day_of_year (float, optional): Day of year.
        - hour_of_day (float, optional): Hour of day.
    GEOS5FP_connection (GEOS5FP, optional): Connection object for GEOS-5 FP data.
    NASADEM_connection (NASADEMConnection, optional): Connection object for NASADEM data.

    Returns:
    pd.DataFrame: A DataFrame with the same structure as the input, but with additional columns:
        - albedo: Surface albedo array
        - COT: Cloud optical thickness array
        - AOT: Aerosol optical thickness array
        - vapor_gccm: Water vapor array
        - ozone_cm: Ozone concentration array
        - elevation_m: Elevation in meters array
        - elevation_km: Elevation in kilometers array
        - KG_climate: Climate classification array
        - SZA_deg: Solar zenith angle array
        - SWin_Wm2: Shortwave incoming radiation array
        - day_of_year: Day of year array
        - atype: Aerosol type array
        - ctype: Cloud type array

    Raises:
    KeyError: If required columns ("geometry" or "lat" and "lon") are missing.
    """
    def ensure_geometry(row):
        if "geometry" in row:
            if isinstance(row.geometry, str):
                s = row.geometry.strip()
                if s.startswith("POINT"):
                    coords = s.replace("POINT", "").replace("(", "").replace(")", "").strip().split()
                    return Point(float(coords[0]), float(coords[1]))
                elif "," in s:
                    coords = [float(c) for c in s.split(",")]
                    return Point(coords[0], coords[1])
                else:
                    coords = [float(c) for c in s.split()]
                    return Point(coords[0], coords[1])
        return row.geometry

    logger.info("started generating FLiES input table")

    # Ensure geometry column is properly formatted
    input_df = input_df.copy()
    input_df["geometry"] = input_df.apply(ensure_geometry, axis=1)

    # Prepare output DataFrame
    output_df = input_df.copy()
    
    # Prepare geometries
    if "geometry" in input_df.columns:
        geometries = MultiPoint([(geom.x, geom.y) for geom in input_df.geometry], crs=WGS84)
    elif "lat" in input_df.columns and "lon" in input_df.columns:
        geometries = MultiPoint([(lon, lat) for lon, lat in zip(input_df.lon, input_df.lat)], crs=WGS84)
    else:
        raise KeyError("Input DataFrame must contain either 'geometry' or both 'lat' and 'lon' columns.")
    
    # Convert time column to datetime
    times_UTC = pd.to_datetime(input_df.time_UTC)
    
    logger.info(f"generating inputs for {len(input_df)} rows")

    # Helper function to get column values or None if column doesn't exist
    def get_column_or_none(df, col_name, default_col_name=None):
        if col_name in df.columns:
            return df[col_name].values
        elif default_col_name and default_col_name in df.columns:
            return df[default_col_name].values
        else:
            return None

    # Retrieve all inputs at once using vectorized retrieve_FLiESANN_inputs call
    FLiES_inputs = retrieve_FLiESANN_inputs(
        geometry=geometries,
        time_UTC=times_UTC,
        albedo=get_column_or_none(input_df, "albedo"),
        COT=get_column_or_none(input_df, "COT"),
        AOT=get_column_or_none(input_df, "AOT"),
        vapor_gccm=get_column_or_none(input_df, "vapor_gccm"),
        ozone_cm=get_column_or_none(input_df, "ozone_cm"),
        elevation_m=get_column_or_none(input_df, "elevation_m"),
        SZA_deg=get_column_or_none(input_df, "SZA_deg", "SZA"),
        KG_climate=get_column_or_none(input_df, "KG_climate", "KG"),
        SWin_Wm2=get_column_or_none(input_df, "SWin_Wm2"),
        day_of_year=get_column_or_none(input_df, "day_of_year"),
        hour_of_day=get_column_or_none(input_df, "hour_of_day"),
        GEOS5FP_connection=GEOS5FP_connection,
        NASADEM_connection=NASADEM_connection
    )

    # Add retrieved inputs to the output DataFrame
    for key, values in FLiES_inputs.items():
        # Skip values with mismatched lengths
        if hasattr(values, '__len__') and not isinstance(values, str):
            if len(values) != len(output_df):
                continue
            # Extract scalar values from single-element arrays
            # This prevents pandas from storing array representations as strings
            if isinstance(values, np.ndarray):
                # Convert array to list of scalars
                values = [v.item() if isinstance(v, np.ndarray) and v.size == 1 else v for v in values]
        output_df[key] = values

    logger.info("completed generating FLiES input table")

    return output_df

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
from .process_FLiESANN import FLiESANN

logger = logging.getLogger(__name__)

def process_FLiESANN_table(
        input_df: DataFrame,
        GEOS5FP_connection: GEOS5FP = None,
        NASADEM_connection: NASADEMConnection = None,
        row_wise: bool = False) -> DataFrame:
    """
    Processes a DataFrame of FLiES inputs and returns a DataFrame with FLiES outputs.

    Parameters:
    input_df (pd.DataFrame): A DataFrame containing the following columns:
        - time_UTC (str or datetime): Time in UTC.
        - geometry (str or shapely.geometry.Point) or (lat, lon): Spatial coordinates. If "geometry" is a string, it should be in WKT format (e.g., "POINT (lon lat)").
        - doy (int, optional): Day of the year. If not provided, it will be derived from "time_UTC".
        - albedo (float): Surface albedo.
        - COT (float, optional): Cloud optical thickness.
        - AOT (float, optional): Aerosol optical thickness.
        - vapor_gccm (float): Water vapor in grams per cubic centimeter.
        - ozone_cm (float): Ozone concentration in centimeters.
        - elevation_m (float): Elevation in meters.
        - SZA (float, optional): Solar zenith angle in degrees.
        - KG or KG_climate (str): Köppen-Geiger climate classification.
    GEOS5FP_connection (GEOS5FP, optional): Connection object for GEOS-5 FP data.
    NASADEM_connection (NASADEMConnection, optional): Connection object for NASADEM data.
    row_wise (bool, optional): If True (default), processes each row individually. If False, 
        attempts vectorized processing when possible for better performance.

    Returns:
    pd.DataFrame: A DataFrame with the same structure as the input, but with additional columns:
        - SWin_Wm2: Shortwave incoming solar radiation at the bottom of the atmosphere.
        - SWin_TOA_Wm2: Shortwave incoming solar radiation at the top of the atmosphere.
        - UV_Wm2: Ultraviolet radiation.
        - PAR_Wm2: Photosynthetically active radiation (visible).
        - NIR_Wm2: Near-infrared radiation.
        - PAR_diffuse_Wm2: Diffuse visible radiation.
        - NIR_diffuse_Wm2: Diffuse near-infrared radiation.
        - PAR_direct_Wm2: Direct visible radiation.
        - NIR_direct_Wm2: Direct near-infrared radiation.
        - atmospheric_transmittance: Total atmospheric transmittance.
        - UV_proportion: Proportion of ultraviolet radiation.
        - PAR_proportion: Proportion of visible radiation.
        - NIR_proportion: Proportion of near-infrared radiation.
        - UV_diffuse_fraction: Diffuse fraction of ultraviolet radiation.
        - PAR_diffuse_fraction: Diffuse fraction of visible radiation.
        - NIR_diffuse_fraction: Diffuse fraction of near-infrared radiation.

    Raises:
    KeyError: If required columns ("geometry" or "lat" and "lon", "KG_climate" or "KG") are missing.
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

    logger.info("started processing FLiES input table")

    # Ensure geometry column is properly formatted
    input_df = input_df.copy()
    input_df["geometry"] = input_df.apply(ensure_geometry, axis=1)

    # Prepare output DataFrame
    output_df = input_df.copy()

    if row_wise:
        # Process each row individually (original behavior)
        logger.info("processing table row-wise")
        results = []
        for _, row in input_df.iterrows():
            if "geometry" in row:
                geometry = rt.Point((row.geometry.x, row.geometry.y), crs=WGS84)
            elif "lat" in row and "lon" in row:
                geometry = rt.Point((row.lon, row.lat), crs=WGS84)
            else:
                raise KeyError("Input DataFrame must contain either 'geometry' or both 'lat' and 'lon' columns.")

            time_UTC = pd.to_datetime(row.time_UTC)
            doy = row.doy if "doy" in row else time_UTC.timetuple().tm_yday

            logger.info(f"processing row with time_UTC: {time_UTC}, geometry: {geometry}")

            FLiES_results = FLiESANN(
                geometry=geometry,
                time_UTC=time_UTC,
                albedo=row.albedo.item() if hasattr(row.albedo, 'item') else row.albedo,
                COT=row.get("COT").item() if row.get("COT") is not None and hasattr(row.get("COT"), 'item') else row.get("COT"),
                AOT=row.get("AOT").item() if row.get("AOT") is not None and hasattr(row.get("AOT"), 'item') else row.get("AOT"),
                vapor_gccm=row.get("vapor_gccm").item() if row.get("vapor_gccm") is not None and hasattr(row.get("vapor_gccm"), 'item') else row.get("vapor_gccm"),
                ozone_cm=row.get("ozone_cm").item() if row.get("ozone_cm") is not None and hasattr(row.get("ozone_cm"), 'item') else row.get("ozone_cm"),
                elevation_m=row.get("elevation_m").item() if row.get("elevation_m") is not None and hasattr(row.get("elevation_m"), 'item') else row.get("elevation_m"),
                SZA_deg=row.get("SZA").item() if row.get("SZA") is not None and hasattr(row.get("SZA"), 'item') else row.get("SZA"),
                KG_climate=row.get("KG_climate", row.get("KG")).item() if row.get("KG_climate", row.get("KG")) is not None and hasattr(row.get("KG_climate", row.get("KG")), 'item') else row.get("KG_climate", row.get("KG")),
                GEOS5FP_connection=GEOS5FP_connection,
                NASADEM_connection=NASADEM_connection
            )

            results.append(FLiES_results)

        # Combine results into the output DataFrame
        for key in results[0].keys():
            output_df[key] = [result[key] for result in results]
    else:
        # Vectorized processing for better performance
        logger.info("processing table in vectorized mode")
        
        # Prepare geometries
        if "geometry" in input_df.columns:
            geometries = MultiPoint([(geom.x, geom.y) for geom in input_df.geometry], crs=WGS84)
        elif "lat" in input_df.columns and "lon" in input_df.columns:
            geometries = MultiPoint([(lon, lat) for lon, lat in zip(input_df.lon, input_df.lat)], crs=WGS84)
        else:
            raise KeyError("Input DataFrame must contain either 'geometry' or both 'lat' and 'lon' columns.")
        
        # Convert time column to datetime
        times_UTC = pd.to_datetime(input_df.time_UTC)
        
        logger.info(f"processing {len(input_df)} rows in vectorized mode")

        # Helper function to get column values or None if column doesn't exist
        def get_column_or_none(df, col_name, default_col_name=None):
            if col_name in df.columns:
                return df[col_name].values
            elif default_col_name and default_col_name in df.columns:
                return df[default_col_name].values
            else:
                return None

        # Process all rows at once using vectorized FLiESANN call
        FLiES_results = FLiESANN(
            geometry=geometries,
            time_UTC=times_UTC,
            albedo=input_df.albedo.values,
            COT=get_column_or_none(input_df, "COT"),
            AOT=get_column_or_none(input_df, "AOT"),
            vapor_gccm=get_column_or_none(input_df, "vapor_gccm"),
            ozone_cm=get_column_or_none(input_df, "ozone_cm"),
            elevation_m=get_column_or_none(input_df, "elevation_m"),
            SZA_deg=get_column_or_none(input_df, "SZA"),
            KG_climate=get_column_or_none(input_df, "KG_climate", "KG"),
            GEOS5FP_connection=GEOS5FP_connection,
            NASADEM_connection=NASADEM_connection
        )

        # Add results to the output DataFrame
        for key, values in FLiES_results.items():
            output_df[key] = values

    logger.info("completed processing FLiES input table")

    return output_df

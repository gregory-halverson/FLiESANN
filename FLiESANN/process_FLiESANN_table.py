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
        NASADEM_connection: NASADEMConnection = None) -> DataFrame:
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
        - elevation_km (float): Elevation in kilometers.
        - SZA (float, optional): Solar zenith angle in degrees.
        - KG or KG_climate (str): Köppen-Geiger climate classification.

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

    # Iterate over rows and process each individually
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
            albedo=row.albedo,
            COT=row.get("COT"),
            AOT=row.get("AOT"),
            vapor_gccm=row.get("vapor_gccm"),
            ozone_cm=row.get("ozone_cm"),
            elevation_km=row.get("elevation_km"),
            SZA=row.get("SZA"),
            KG_climate=row.get("KG_climate", row.get("KG")),
            GEOS5FP_connection=GEOS5FP_connection,
            NASADEM_connection=NASADEM_connection
        )

        results.append(FLiES_results)

    # Combine results into the output DataFrame
    for key in results[0].keys():
        output_df[key] = [result[key] for result in results]

    logger.info("completed processing FLiES input table")

    return output_df

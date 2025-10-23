import logging

import numpy as np
import pandas as pd
import rasters as rt
from dateutil import parser
from pandas import DataFrame
from rasters import MultiPoint, WGS84
from shapely.geometry import Point

from .process_FLiESANN import FLiESANN

logger = logging.getLogger(__name__)

def process_FLiESANN_table(input_df: DataFrame) -> DataFrame:
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
    
    def ensure_geometry(df):
        if "geometry" in df:
            if isinstance(df.geometry.iloc[0], str):
                def parse_geom(s):
                    s = s.strip()
                    if s.startswith("POINT"):
                        coords = s.replace("POINT", "").replace("(", "").replace(")", "").strip().split()
                        return Point(float(coords[0]), float(coords[1]))
                    elif "," in s:
                        coords = [float(c) for c in s.split(",")]
                        return Point(coords[0], coords[1])
                    else:
                        coords = [float(c) for c in s.split()]
                        return Point(coords[0], coords[1])
                df = df.copy()
                df['geometry'] = df['geometry'].apply(parse_geom)
        return df

    input_df = ensure_geometry(input_df)

    logger.info("started extracting geometry from FLiES input table")

    if "geometry" in input_df:
        # Convert Point objects to coordinate tuples for MultiPoint
        if hasattr(input_df.geometry.iloc[0], "x") and hasattr(input_df.geometry.iloc[0], "y"):
            coords = [(pt.x, pt.y) for pt in input_df.geometry]
            geometry = MultiPoint(coords, crs=WGS84)
        else:
            geometry = MultiPoint(input_df.geometry, crs=WGS84)
    elif "lat" in input_df and "lon" in input_df:
        lat = np.array(input_df.lat).astype(np.float64)
        lon = np.array(input_df.lon).astype(np.float64)
        geometry = MultiPoint(x=lon, y=lat, crs=WGS84)
    else:
        raise KeyError("Input DataFrame must contain either 'geometry' or both 'lat' and 'lon' columns.")

    logger.info("completed extracting geometry from FLiES input table")

    logger.info("started extracting time from FLiES input table")
    time_UTC = pd.to_datetime(input_df.time_UTC).tolist()
    logger.info("completed extracting time from FLiES input table")

    # Extract day of year from time_UTC if not provided
    if "doy" in input_df:
        doy = np.array(input_df.doy).astype(np.float64)
    else:
        doy = np.array([t.timetuple().tm_yday for t in time_UTC]).astype(np.float64)

    # Extract required FLiES parameters
    albedo = np.array(input_df.albedo).astype(np.float64)
    
    if "COT" in input_df:
        COT = np.array(input_df.COT).astype(np.float64)
    else:
        COT = None
    
    if "AOT" in input_df:
        AOT = np.array(input_df.AOT).astype(np.float64)
    else:
        AOT = None

    if "vapor_gccm" in input_df:
        vapor_gccm = np.array(input_df.vapor_gccm).astype(np.float64)
    else:
        vapor_gccm = None
    
    if "ozone_cm" in input_df:
        ozone_cm = np.array(input_df.ozone_cm).astype(np.float64)
    else:
        ozone_cm = None

    if "elevation_km" in input_df:
        elevation_km = np.array(input_df.elevation_km).astype(np.float64)
    else:
        elevation_km = None

    if "SZA" in input_df:
        SZA = np.array(input_df.SZA).astype(np.float64)
    else:
        SZA = None

    # Handle Köppen-Geiger climate classification
    if "KG_climate" in input_df:
        KG_climate = np.array(input_df.KG_climate)
    elif "KG" in input_df:
        KG_climate = np.array(input_df.KG)
    else:
        KG_climate = None
    
    FLiES_results = FLiESANN(
        geometry=geometry,
        time_UTC=time_UTC,
        albedo=albedo,
        COT=COT,
        AOT=AOT,
        vapor_gccm=vapor_gccm,
        ozone_cm=ozone_cm,
        elevation_km=elevation_km,
        SZA=SZA,
        KG_climate=KG_climate
    )

    output_df = input_df.copy()

    for key, value in FLiES_results.items():
        output_df[key] = value

    return output_df

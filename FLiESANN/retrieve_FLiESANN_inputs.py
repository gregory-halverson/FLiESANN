from typing import Union
from datetime import datetime
import numpy as np
import pandas as pd
import rasters as rt
from rasters import Raster, RasterGeometry
from GEOS5FP import GEOS5FP
from solar_apparent_time import calculate_solar_day_of_year, calculate_solar_hour_of_day
from sun_angles import calculate_SZA_from_DOY_and_hour
from NASADEM import NASADEMConnection, NASADEM
import shapely

from .ensure_array import ensure_array
from .retrieve_FLiESANN_static_inputs import retrieve_FLiESANN_static_inputs
from .retrieve_FLiESANN_GEOS5FP_inputs import retrieve_FLiESANN_GEOS5FP_inputs
from .filter_dataframe_to_location_time_pairs import filter_dataframe_to_location_time_pairs
from .determine_atype import determine_atype
from .determine_ctype import determine_ctype
from .constants import *


def retrieve_FLiESANN_inputs(
        albedo: Union[Raster, np.ndarray, float] = None,
        COT: Union[Raster, np.ndarray, float] = None,
        AOT: Union[Raster, np.ndarray, float] = None,
        vapor_gccm: Union[Raster, np.ndarray, float] = None,
        ozone_cm: Union[Raster, np.ndarray, float] = None,
        elevation_m: Union[Raster, np.ndarray, float] = None,
        SZA_deg: Union[Raster, np.ndarray, float] = None,
        KG_climate: Union[Raster, np.ndarray, int] = None,
        SWin_Wm2: Union[Raster, np.ndarray, float] = None,
        geometry: Union[RasterGeometry, shapely.geometry.Point, rt.Point, shapely.geometry.MultiPoint, rt.MultiPoint] = None,
        time_UTC: datetime = None,
        day_of_year: Union[Raster, np.ndarray, float] = None,
        hour_of_day: Union[Raster, np.ndarray, float] = None,
        GEOS5FP_connection: GEOS5FP = None,
        NASADEM_connection: NASADEMConnection = None,
        resampling: str = DEFAULT_RESAMPLING,
        zero_COT_correction: bool = ZERO_COT_CORRECTION,
        offline_mode: bool = False) -> dict:
    """
    Retrieve and prepare all input arrays for FLiESANN inference.
    
    This function handles:
    - Shape determination from geometry
    - Static input retrieval (elevation, climate)
    - GEOS-5 FP atmospheric input retrieval (COT, AOT, vapor, ozone)
    - DataFrame to array conversion and filtering
    - Array shape normalization
    - Aerosol and cloud type determination
    
    Args:
        albedo: Surface broadband albedo (0.3-5.0 μm)
        COT: Cloud optical thickness
        AOT: Aerosol optical thickness
        vapor_gccm: Water vapor in grams per square centimeter
        ozone_cm: Ozone concentration in centimeters
        elevation_m: Elevation in meters
        SZA_deg: Solar zenith angle in degrees
        KG_climate: Köppen-Geiger climate classification
        SWin_Wm2: Shortwave incoming solar radiation
        geometry: Spatial geometry (RasterGeometry, Point, or MultiPoint)
        time_UTC: UTC time for the calculation
        day_of_year: Day of the year
        hour_of_day: Hour of the day
        GEOS5FP_connection: Connection to GEOS-5 FP data
        NASADEM_connection: Connection to NASADEM data
        resampling: Resampling method for raster data
        zero_COT_correction: Flag to apply zero COT correction
        
    Returns:
        dict: Dictionary containing all prepared input arrays with keys:
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
    """
    # Determine shape for array operations - include MultiPoint for vectorized processing
    if isinstance(geometry, (Raster, np.ndarray)):
        shape = geometry.shape
    elif isinstance(geometry, (shapely.geometry.MultiPoint, rt.MultiPoint)):
        shape = (len(geometry.geoms),) if hasattr(geometry, 'geoms') else (len(geometry),)
    else:
        shape = None

    albedo = ensure_array(albedo, shape)
    SWin_Wm2 = ensure_array(SWin_Wm2, shape)
    day_of_year = ensure_array(day_of_year, shape)
    hour_of_day = ensure_array(hour_of_day, shape)
    SZA_deg = ensure_array(SZA_deg, shape)

    # Retrieve static inputs (elevation and climate)
    static_inputs = retrieve_FLiESANN_static_inputs(
        elevation_m=elevation_m,
        KG_climate=KG_climate,
        geometry=geometry,
        NASADEM_connection=NASADEM_connection,
        resampling=resampling
    )
    
    # Extract retrieved values
    elevation_m = static_inputs["elevation_m"]
    elevation_km = static_inputs["elevation_km"]
    KG_climate = static_inputs["KG_climate"]

    # Retrieve GEOS-5 FP atmospheric inputs
    GEOS5FP_inputs = retrieve_FLiESANN_GEOS5FP_inputs(
        COT=COT,
        AOT=AOT,
        vapor_gccm=vapor_gccm,
        ozone_cm=ozone_cm,
        geometry=geometry,
        time_UTC=time_UTC,
        GEOS5FP_connection=GEOS5FP_connection,
        resampling=resampling,
        zero_COT_correction=zero_COT_correction,
        offline_mode=offline_mode
    )
    
    # Extract retrieved values
    COT = GEOS5FP_inputs["COT"]
    AOT = GEOS5FP_inputs["AOT"]
    vapor_gccm = GEOS5FP_inputs["vapor_gccm"]
    ozone_cm = GEOS5FP_inputs["ozone_cm"]
    
    # Convert DataFrames to arrays first (if they are DataFrames)
    # This is necessary because GEOS5FP returns DataFrames for time-series data
    # When processing multiple location-time pairs, GEOS5FP returns a DataFrame with
    # rows for each unique time at each location, creating a Cartesian product.
    # We need to filter to match only the original location-time pairs.
    if isinstance(COT, pd.DataFrame) or (isinstance(COT, np.ndarray) and len(COT.shape) == 2):
        COT = filter_dataframe_to_location_time_pairs(COT, geometry, time_UTC)
    if isinstance(AOT, pd.DataFrame) or (isinstance(AOT, np.ndarray) and len(AOT.shape) == 2):
        AOT = filter_dataframe_to_location_time_pairs(AOT, geometry, time_UTC)
    if isinstance(vapor_gccm, pd.DataFrame) or (isinstance(vapor_gccm, np.ndarray) and len(vapor_gccm.shape) == 2):
        vapor_gccm = filter_dataframe_to_location_time_pairs(vapor_gccm, geometry, time_UTC)
    if isinstance(ozone_cm, pd.DataFrame) or (isinstance(ozone_cm, np.ndarray) and len(ozone_cm.shape) == 2):
        ozone_cm = filter_dataframe_to_location_time_pairs(ozone_cm, geometry, time_UTC)
    
    # Update shape based on actual retrieved data arrays (after DataFrame conversion)
    # For MultiPoint geometries, use the original shape (number of points)
    # For raster geometries, use the shape of retrieved data
    if isinstance(geometry, (shapely.geometry.MultiPoint, rt.MultiPoint)):
        actual_shape = (len(geometry.geoms),) if hasattr(geometry, 'geoms') else shape
    elif hasattr(COT, 'shape') and not isinstance(COT, pd.DataFrame):
        actual_shape = COT.shape
    elif hasattr(AOT, 'shape') and not isinstance(AOT, pd.DataFrame):
        actual_shape = AOT.shape
    elif hasattr(vapor_gccm, 'shape') and not isinstance(vapor_gccm, pd.DataFrame):
        actual_shape = vapor_gccm.shape
    elif hasattr(ozone_cm, 'shape') and not isinstance(ozone_cm, pd.DataFrame):
        actual_shape = ozone_cm.shape
    else:
        actual_shape = shape
    
    # Ensure arrays have correct shape
    KG_climate = ensure_array(KG_climate, actual_shape) if not isinstance(KG_climate, int) else KG_climate
    COT = ensure_array(COT, actual_shape)
    AOT = ensure_array(AOT, actual_shape)
    vapor_gccm = ensure_array(vapor_gccm, actual_shape)
    ozone_cm = ensure_array(ozone_cm, actual_shape)
    elevation_km = ensure_array(elevation_km, actual_shape)
    elevation_m = ensure_array(elevation_m, actual_shape)
    albedo = ensure_array(albedo, actual_shape)
    SZA_deg = ensure_array(SZA_deg, actual_shape)
    day_of_year = ensure_array(day_of_year, actual_shape)
    SWin_Wm2 = ensure_array(SWin_Wm2, actual_shape)

    # determine aerosol/cloud types
    atype = determine_atype(KG_climate, COT)  # Determine aerosol type
    ctype = determine_ctype(KG_climate, COT)  # Determine cloud type
    
    # Ensure atype and ctype match actual_shape
    atype = ensure_array(atype, actual_shape)
    ctype = ensure_array(ctype, actual_shape)
    
    return {
        "albedo": albedo,
        "COT": COT,
        "AOT": AOT,
        "vapor_gccm": vapor_gccm,
        "ozone_cm": ozone_cm,
        "elevation_m": elevation_m,
        "elevation_km": elevation_km,
        "KG_climate": KG_climate,
        "SZA_deg": SZA_deg,
        "SWin_Wm2": SWin_Wm2,
        "day_of_year": day_of_year,
        "atype": atype,
        "ctype": ctype
    }

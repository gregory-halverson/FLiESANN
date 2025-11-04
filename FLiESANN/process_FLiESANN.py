from typing import Union
from time import process_time
from datetime import datetime
import numpy as np
import rasters as rt
from rasters import Raster, RasterGeometry
from GEOS5FP import GEOS5FP
from solar_apparent_time import solar_day_of_year_for_area, solar_hour_of_day_for_area
from solar_apparent_time import calculate_solar_day_of_year, calculate_solar_hour_of_day
from sun_angles import calculate_SZA_from_DOY_and_hour
from koppengeiger import load_koppen_geiger
from NASADEM import NASADEM, NASADEMConnection
import shapely

from .constants import *
from .colors import *
from .determine_atype import determine_atype
from .determine_ctype import determine_ctype
from .run_FLiESANN_inference import run_FLiESANN_inference

def FLiESANN(
        albedo: Union[Raster, np.ndarray, float],
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
        NASADEM_connection: NASADEMConnection = NASADEM,
        resampling: str = "cubic",
        ANN_model=None,
        model_filename: str = MODEL_FILENAME,
        split_atypes_ctypes: bool = SPLIT_ATYPES_CTYPES,
        zero_COT_correction: bool = ZERO_COT_CORRECTION) -> dict:
    """
    Processes Forest Light Environmental Simulator (FLiES) calculations using an 
    artificial neural network (ANN) emulator.

    This function estimates radiative transfer components such as total transmittance, 
    diffuse and direct radiation in different spectral bands (UV, visible, near-infrared) 
    based on various atmospheric and environmental parameters.

    Args:
        albedo (Union[Raster, np.ndarray]): Surface albedo.
        COT (Union[Raster, np.ndarray], optional): Cloud optical thickness. Defaults to None.
        AOT (Union[Raster, np.ndarray], optional): Aerosol optical thickness. Defaults to None.
        vapor_gccm (Union[Raster, np.ndarray], optional): Water vapor in grams per square centimeter. Defaults to None.
        ozone_cm (Union[Raster, np.ndarray], optional): Ozone concentration in centimeters. Defaults to None.
        elevation_m (Union[Raster, np.ndarray], optional): Elevation in meters. Defaults to None.
        SZA (Union[Raster, np.ndarray], optional): Solar zenith angle. Defaults to None.
        KG_climate (Union[Raster, np.ndarray], optional): Köppen-Geiger climate classification. Defaults to None.
        SWin_Wm2 (Union[Raster, np.ndarray], optional): Shortwave incoming solar radiation at the bottom of the atmosphere. Defaults to None.
        geometry (RasterGeometry, optional): RasterGeometry object defining the spatial extent and resolution. Defaults to None.
        time_UTC (datetime, optional): UTC time for the calculation. Defaults to None.
        day_of_year (Union[Raster, np.ndarray], optional): Day of the year. Defaults to None.
        hour_of_day (Union[Raster, np.ndarray], optional): Hour of the day. Defaults to None.
        GEOS5FP_connection (GEOS5FP, optional): Connection to GEOS-5 FP data. Defaults to None.
        NASADEM_connection (NASADEMConnection, optional): Connection to NASADEM data. Defaults to NASADEM.
        resampling (str, optional): Resampling method for raster data. Defaults to "cubic".
        ANN_model (optional): Pre-loaded ANN model object. Defaults to None.
        model_filename (str, optional): Filename of the ANN model to load. Defaults to MODEL_FILENAME.
        split_atypes_ctypes (bool, optional): Flag for handling aerosol and cloud types separately. Defaults to SPLIT_ATYPES_CTYPES.
        zero_COT_correction (bool, optional): Flag to apply zero COT correction. Defaults to ZERO_COT_CORRECTION.

    Returns:
        dict: A dictionary containing the calculated radiative transfer components as Raster objects or np.ndarrays, including:
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
            - UV_proportion: Proportion of UV radiation.
            - PAR_proportion: Proportion of visible radiation.
            - NIR_proportion: Proportion of near-infrared radiation.
            - UV_diffuse_fraction: Diffuse fraction of UV radiation.
            - PAR_diffuse_fraction: Diffuse fraction of visible radiation.
            - NIR_diffuse_fraction: Diffuse fraction of near-infrared radiation.

    Raises:
        ValueError: If required time or geometry parameters are not provided.
    """
    results = {}

    def ensure_array(value, shape=None):
        """Ensure the input is an array, converting scalar values if necessary."""
        if isinstance(value, (int, float)):
            return np.full(shape, value, dtype=np.float32) if shape else np.array(value, dtype=np.float32)
        elif value is None:
            return None
        elif isinstance(value, np.ndarray):
            # Convert object arrays with None values to float arrays with NaN
            if value.dtype == object:
                # Replace None with NaN and convert to float32
                value_copy = value.copy()
                value_copy[value_copy == None] = np.nan
                return value_copy.astype(np.float32)
            else:
                return value.astype(np.float32)
        else:
            # For other types (like lists), convert to array and then ensure float32
            arr = np.array(value)
            if arr.dtype == object:
                arr[arr == None] = np.nan
                return arr.astype(np.float32)
            else:
                return arr.astype(np.float32)

    if geometry is not None and not isinstance(geometry, RasterGeometry) and not isinstance(geometry, (shapely.geometry.Point, rt.Point, shapely.geometry.MultiPoint, rt.MultiPoint)):
        raise TypeError(f"geometry must be a RasterGeometry, Point, MultiPoint or None, not {type(geometry)}")

    if geometry is None and isinstance(albedo, Raster):
        geometry = albedo.geometry

    if (day_of_year is None or hour_of_day is None) and time_UTC is not None and geometry is not None:
        day_of_year = calculate_solar_day_of_year(time_UTC=time_UTC, geometry=geometry)
        hour_of_day = calculate_solar_hour_of_day(time_UTC=time_UTC, geometry=geometry)

    if time_UTC is None and day_of_year is None and hour_of_day is None:
        raise ValueError("no time given between time_UTC, day_of_year, and hour_of_day")

    if GEOS5FP_connection is None:
        GEOS5FP_connection = GEOS5FP()

    ## FIXME need to fetch default values for parameters: COT, AOT, vapor_gccm, ozone_cm, elevation_km, SZA, KG_climate 

    # Determine shape for array operations - include MultiPoint for vectorized processing
    if isinstance(geometry, (Raster, np.ndarray)):
        shape = geometry.shape
    elif isinstance(geometry, (shapely.geometry.MultiPoint, rt.MultiPoint)):
        shape = (len(geometry.geoms),) if hasattr(geometry, 'geoms') else (len(geometry),)
    else:
        shape = None

    albedo = ensure_array(albedo, shape)

    results["albedo"] = albedo
    
    SWin_Wm2 = ensure_array(SWin_Wm2, shape)
    day_of_year = ensure_array(day_of_year, shape)
    hour_of_day = ensure_array(hour_of_day, shape)

    if SZA_deg is None and geometry is not None:
        SZA_deg = calculate_SZA_from_DOY_and_hour(
            lat=geometry.lat,
            lon=geometry.lon,
            DOY=day_of_year,
            hour=hour_of_day
        )

    if SZA_deg is None:
        raise ValueError("solar zenith angle or geometry and time must be given")

    results["SZA_deg"] = SZA_deg

    SZA_deg = ensure_array(SZA_deg, shape)

    if KG_climate is None and geometry is not None:
        KG_climate = load_koppen_geiger(geometry=geometry)

    if KG_climate is None:
        raise ValueError("Koppen Geieger climate classification or geometry must be given")

    results["KG_climate"] = KG_climate

    KG_climate = ensure_array(KG_climate, shape) if not isinstance(KG_climate, int) else KG_climate

    if zero_COT_correction:
        COT = np.zeros(albedo.shape, dtype=np.float32)
    elif COT is None and geometry is not None and time_UTC is not None:
        COT = GEOS5FP_connection.COT(
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling
        )
    
    if COT is None:
        raise ValueError("cloud optical thickness or geometry and time must be given")

    results["COT"] = COT

    COT = ensure_array(COT, shape)
    
    if AOT is None and geometry is not None and time_UTC is not None:
        AOT = GEOS5FP_connection.AOT(
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling
        )

    if AOT is None:
        raise ValueError("aerosol optical thickness or geometry and time must be given")

    results["AOT"] = AOT

    AOT = ensure_array(AOT, shape)

    if vapor_gccm is None and geometry is not None and time_UTC is not None:
        vapor_gccm = GEOS5FP_connection.vapor_gccm(
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling
        )

    if vapor_gccm is None:
        raise ValueError("water vapor or geometry and time must be given")

    results["vapor_gccm"] = vapor_gccm

    vapor_gccm = ensure_array(vapor_gccm, shape)

    if ozone_cm is None and geometry is not None and time_UTC is not None:
        ozone_cm = GEOS5FP_connection.ozone_cm(
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling
        )

    if ozone_cm is None:
        raise ValueError("ozone concentration or geometry and time must be given")

    results["ozone_cm"] = ozone_cm

    ozone_cm = ensure_array(ozone_cm, shape)

    if elevation_m is not None:
        elevation_km = elevation_m / 1000.0

    if elevation_km is None and geometry is not None:
        elevation_km = NASADEM.elevation_km(geometry=geometry)

    if elevation_km is None:
        raise ValueError("elevation or geometry must be given")

    results["elevation_m"] = elevation_m

    elevation_km = ensure_array(elevation_km, shape)

    # Preprocess COT and determine aerosol/cloud types
    COT = np.clip(COT, 0, None)  # Ensure COT is non-negative
    COT = rt.where(COT < 0.001, 0, COT)  # Set very small COT values to 0
    atype = determine_atype(KG_climate, COT)  # Determine aerosol type
    ctype = determine_ctype(KG_climate, COT)  # Determine cloud type

    # Run ANN inference to get initial radiative transfer parameters
    prediction_start_time = process_time()
    
    inference_results = run_FLiESANN_inference(
        atype=atype,
        ctype=ctype,
        COT=COT,
        AOT=AOT,
        vapor_gccm=vapor_gccm,
        ozone_cm=ozone_cm,
        albedo=albedo,
        elevation_m=elevation_m,
        SZA=SZA_deg,
        ANN_model=ANN_model,
        model_filename=model_filename,
        split_atypes_ctypes=split_atypes_ctypes
    )

    results.update(inference_results)

    prediction_end_time = process_time()
    prediction_duration = prediction_end_time - prediction_start_time

    # Extract individual components from the results dictionary
    atmospheric_transmittance = results["atmospheric_transmittance"]
    UV_proportion = results["UV_proportion"]
    PAR_proportion = results["PAR_proportion"]
    NIR_proportion = results["NIR_proportion"]
    UV_diffuse_fraction = results["UV_diffuse_fraction"]
    PAR_diffuse_fraction = results["PAR_diffuse_fraction"]
    NIR_diffuse_fraction = results["NIR_diffuse_fraction"]

    ## Correction for diffuse PAR
    COT = rt.where(COT == 0.0, np.nan, COT)
    COT = rt.where(np.isfinite(COT), COT, np.nan)
    x = np.log(COT)
    p1 = 0.05088
    p2 = 0.04909
    p3 = 0.5017
    corr = np.array(p1 * x * x + p2 * x + p3)
    corr[np.logical_or(np.isnan(corr), corr > 1.0)] = 1.0
    PAR_diffuse_fraction = PAR_diffuse_fraction * corr * 0.915

    ## Radiation components
    if SWin_Wm2 is None:
        dr = 1.0 + 0.033 * np.cos(2 * np.pi / 365.0 * day_of_year)  # Earth-sun distance correction factor
        SWin_TOA_Wm2 = 1333.6 * dr * np.cos(SZA_deg * np.pi / 180.0)  # Extraterrestrial radiation
        SWin_TOA_Wm2 = rt.where(SZA_deg > 90.0, 0, SWin_TOA_Wm2)  # Set Ra to 0 when the sun is below the horizon
    
    SWin_Wm2 = SWin_TOA_Wm2 * atmospheric_transmittance  # scale top-of-atmosphere shortwave radiation to bottom-of-atmosphere

    # Calculate ultraviolet radiation (UV) in W/m² by scaling the total shortwave incoming radiation (SWin_Wm2)
    # with the proportion of UV radiation (UV_proportion). UV radiation is a small fraction of the solar spectrum.
    UV_Wm2 = SWin_Wm2 * UV_proportion

    # Calculate photosynthetically active radiation (PAR) in W/m², which represents the visible portion of the solar spectrum.
    # This is derived by scaling the total shortwave incoming radiation (SWin_Wm2) with the proportion of visible radiation (PAR_proportion).
    PAR_Wm2 = SWin_Wm2 * PAR_proportion

    # Calculate near-infrared radiation (NIR) in W/m², which represents the portion of the solar spectrum beyond visible light.
    # This is derived by scaling the total shortwave incoming radiation (SWin_Wm2) with the proportion of NIR radiation (NIR_proportion).
    NIR_Wm2 = SWin_Wm2 * NIR_proportion

    # Calculate diffuse visible radiation (PAR_diffuse_Wm2) in W/m² by scaling the total visible radiation (PAR_Wm2)
    # with the diffuse fraction of visible radiation (PAR_diffuse_fraction). The np.clip function ensures the value
    # remains within the range [0, PAR_Wm2]. Diffuse radiation is scattered sunlight that reaches the surface indirectly.
    PAR_diffuse_Wm2 = np.clip(PAR_Wm2 * PAR_diffuse_fraction, 0, PAR_Wm2)

    # Calculate diffuse near-infrared radiation (NIR_diffuse_Wm2) in W/m² by scaling the total NIR radiation (NIR_Wm2)
    # with the diffuse fraction of NIR radiation (NIR_diffuse_fraction). The np.clip function ensures the value
    # remains within the range [0, NIR_Wm2].
    NIR_diffuse_Wm2 = np.clip(NIR_Wm2 * NIR_diffuse_fraction, 0, NIR_Wm2)

    # Calculate direct visible radiation (PAR_direct_Wm2) in W/m² by subtracting the diffuse visible radiation (PAR_diffuse_Wm2)
    # from the total visible radiation (PAR_Wm2). The np.clip function ensures the value remains within the range [0, PAR_Wm2].
    # Direct radiation is sunlight that reaches the surface without being scattered.
    PAR_direct_Wm2 = np.clip(PAR_Wm2 - PAR_diffuse_Wm2, 0, PAR_Wm2)

    # Calculate direct near-infrared radiation (NIR_direct_Wm2) in W/m² by subtracting the diffuse NIR radiation (NIR_diffuse_Wm2)
    # from the total NIR radiation (NIR_Wm2). The np.clip function ensures the value remains within the range [0, NIR_Wm2].
    NIR_direct_Wm2 = np.clip(NIR_Wm2 - NIR_diffuse_Wm2, 0, NIR_Wm2)

    if isinstance(geometry, RasterGeometry):
        SWin_Wm2 = rt.Raster(SWin_Wm2, geometry=geometry)
        SWin_TOA_Wm2 = rt.Raster(SWin_TOA_Wm2, geometry=geometry)
        UV_Wm2 = rt.Raster(UV_Wm2, geometry=geometry)
        PAR_Wm2 = rt.Raster(PAR_Wm2, geometry=geometry)
        NIR_Wm2 = rt.Raster(NIR_Wm2, geometry=geometry)
        PAR_diffuse_Wm2 = rt.Raster(PAR_diffuse_Wm2, geometry=geometry)
        NIR_diffuse_Wm2 = rt.Raster(NIR_diffuse_Wm2, geometry=geometry)
        PAR_direct_Wm2 = rt.Raster(PAR_direct_Wm2, geometry=geometry)
        NIR_direct_Wm2 = rt.Raster(NIR_direct_Wm2, geometry=geometry)

    if isinstance(UV_Wm2, Raster):
        UV_Wm2.cmap = UV_CMAP

    # Update the results dictionary with new items instead of replacing it
    results.update({
        "SWin_Wm2": SWin_Wm2,
        "SWin_TOA_Wm2": SWin_TOA_Wm2,
        "UV_Wm2": UV_Wm2,
        "PAR_Wm2": PAR_Wm2,
        "NIR_Wm2": NIR_Wm2,
        "PAR_diffuse_Wm2": PAR_diffuse_Wm2,
        "NIR_diffuse_Wm2": NIR_diffuse_Wm2,
        "PAR_direct_Wm2": PAR_direct_Wm2,
        "NIR_direct_Wm2": NIR_direct_Wm2,
        "atmospheric_transmittance": atmospheric_transmittance,
        "UV_proportion": UV_proportion,
        "PAR_proportion": PAR_proportion,
        "NIR_proportion": NIR_proportion,
        "UV_diffuse_fraction": UV_diffuse_fraction,
        "PAR_diffuse_fraction": PAR_diffuse_fraction,
        "NIR_diffuse_fraction": NIR_diffuse_fraction
    })

    # Convert results to Raster objects if raster geometry is given
    if isinstance(geometry, RasterGeometry):
        for key in results.keys():
            results[key] = rt.Raster(results[key], geometry=geometry)

    return results

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
from .retrieve_FLiESANN_inputs import retrieve_FLiESANN_inputs
from .ensure_array import ensure_array
from .partition_spectral_albedo_with_NDVI import partition_spectral_albedo_with_NDVI

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
        NDVI: Union[Raster, np.ndarray, float] = None,
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
        zero_COT_correction: bool = ZERO_COT_CORRECTION,
        offline_mode: bool = False) -> dict:
    """
    Processes Forest Light Environmental Simulator (FLiES) calculations using an 
    artificial neural network (ANN) emulator.

    This function estimates radiative transfer components such as total transmittance, 
    diffuse and direct radiation in different spectral bands (UV, visible, near-infrared) 
    based on various atmospheric and environmental parameters.

    Args:
        albedo (Union[Raster, np.ndarray]): Surface broadband albedo (0.3-5.0 μm).
        COT (Union[Raster, np.ndarray], optional): Cloud optical thickness. Defaults to None.
        AOT (Union[Raster, np.ndarray], optional): Aerosol optical thickness. Defaults to None.
        vapor_gccm (Union[Raster, np.ndarray], optional): Water vapor in grams per square centimeter. Defaults to None.
        ozone_cm (Union[Raster, np.ndarray], optional): Ozone concentration in centimeters. Defaults to None.
        elevation_m (Union[Raster, np.ndarray], optional): Elevation in meters. Defaults to None.
        SZA (Union[Raster, np.ndarray], optional): Solar zenith angle. Defaults to None.
        KG_climate (Union[Raster, np.ndarray], optional): Köppen-Geiger climate classification. Defaults to None.
        SWin_Wm2 (Union[Raster, np.ndarray], optional): Shortwave incoming solar radiation at the bottom of the atmosphere. Defaults to None.
        NDVI (Union[Raster, np.ndarray], optional): Normalized Difference Vegetation Index (-1 to 1). When provided, enables
            spectral partitioning of albedo into PAR and NIR components based on vegetation properties (Liang 2001,
            Schaaf et al. 2002). If None, spectral albedos are assumed equal to broadband albedo. Defaults to None.
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
            - SWout_Wm2: Shortwave outgoing (reflected) solar radiation.
            - UV_Wm2: Ultraviolet radiation.
            - PAR_Wm2: Photosynthetically active radiation (visible).
            - NIR_Wm2: Near-infrared radiation.
            - PAR_diffuse_Wm2: Diffuse visible radiation.
            - NIR_diffuse_Wm2: Diffuse near-infrared radiation.
            - PAR_direct_Wm2: Direct visible radiation.
            - NIR_direct_Wm2: Direct near-infrared radiation.
            - PAR_reflected_Wm2: Reflected photosynthetically active radiation.
            - NIR_reflected_Wm2: Reflected near-infrared radiation.
            - PAR_albedo: PAR spectral albedo. If NDVI provided, calculated using vegetation-specific partitioning
              (Liang 2001, Schaaf et al. 2002); otherwise assumes uniform spectral reflectance.
            - NIR_albedo: NIR spectral albedo. If NDVI provided, calculated using vegetation-specific partitioning;
              otherwise assumes uniform spectral reflectance.
            - atmospheric_transmittance: Total atmospheric transmittance.
            - UV_proportion: Proportion of UV radiation.
            - PAR_proportion: Proportion of visible radiation.
            - NIR_proportion: Proportion of near-infrared radiation.
            - UV_diffuse_fraction: Diffuse fraction of UV radiation.
            - PAR_diffuse_fraction: Diffuse fraction of visible radiation.
            - NIR_diffuse_fraction: Diffuse fraction of near-infrared radiation.
            - NDVI: (only if provided as input) Normalized Difference Vegetation Index.

    Raises:
        ValueError: If required time or geometry parameters are not provided.
    """
    results = {}

    if geometry is not None and not isinstance(geometry, RasterGeometry) and not isinstance(geometry, (shapely.geometry.Point, rt.Point, shapely.geometry.MultiPoint, rt.MultiPoint)):
        raise TypeError(f"geometry must be a RasterGeometry, Point, MultiPoint or None, not {type(geometry)}")

    if geometry is None and isinstance(albedo, Raster):
        geometry = albedo.geometry

    if (day_of_year is None or hour_of_day is None) and time_UTC is not None and geometry is not None:
        day_of_year = calculate_solar_day_of_year(time_UTC=time_UTC, geometry=geometry)
        hour_of_day = calculate_solar_hour_of_day(time_UTC=time_UTC, geometry=geometry)

    if time_UTC is None and day_of_year is None and hour_of_day is None:
        raise ValueError("no time given between time_UTC, day_of_year, and hour_of_day")

    if SZA_deg is None and geometry is not None:
        SZA_deg = calculate_SZA_from_DOY_and_hour(
            lat=geometry.lat,
            lon=geometry.lon,
            DOY=day_of_year,
            hour=hour_of_day
        )

    if SZA_deg is None:
        raise ValueError("solar zenith angle or geometry and time must be given")

    # Retrieve and prepare all input arrays
    inputs = retrieve_FLiESANN_inputs(
        albedo=albedo,
        COT=COT,
        AOT=AOT,
        vapor_gccm=vapor_gccm,
        ozone_cm=ozone_cm,
        elevation_m=elevation_m,
        SZA_deg=SZA_deg,
        KG_climate=KG_climate,
        SWin_Wm2=SWin_Wm2,
        geometry=geometry,
        time_UTC=time_UTC,
        day_of_year=day_of_year,
        hour_of_day=hour_of_day,
        GEOS5FP_connection=GEOS5FP_connection,
        NASADEM_connection=NASADEM_connection,
        resampling=resampling,
        zero_COT_correction=zero_COT_correction,
        offline_mode=offline_mode
    )
    
    # Extract prepared inputs
    albedo = inputs["albedo"]
    COT = inputs["COT"]
    AOT = inputs["AOT"]
    vapor_gccm = inputs["vapor_gccm"]
    ozone_cm = inputs["ozone_cm"]
    elevation_m = inputs["elevation_m"]
    elevation_km = inputs["elevation_km"]
    KG_climate = inputs["KG_climate"]
    SZA_deg = inputs["SZA_deg"]
    SWin_Wm2 = inputs["SWin_Wm2"]
    day_of_year = inputs["day_of_year"]
    atype = inputs["atype"]
    ctype = inputs["ctype"]
    
    # Store key inputs in results
    results["albedo"] = albedo
    results["SZA_deg"] = SZA_deg
    results["elevation_m"] = elevation_m
    results["KG_climate"] = KG_climate
    results["COT"] = COT
    results["AOT"] = AOT
    results["vapor_gccm"] = vapor_gccm
    results["ozone_cm"] = ozone_cm

    # Run ANN inference to get initial radiative transfer parameters
    prediction_start_time = process_time()
    
    FLiESANN_inference_results = run_FLiESANN_inference(
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

    results.update(FLiESANN_inference_results)

    # Record the end time for performance monitoring
    prediction_end_time = process_time()
    
    # Calculate total time taken for the ANN inference in seconds
    prediction_duration = prediction_end_time - prediction_start_time

    # Extract individual components from the results dictionary
    # Fraction of incoming solar radiation that reaches the surface after atmospheric attenuation (0-1) [previously: tm]
    atmospheric_transmittance = results["atmospheric_transmittance"]
    # Proportion of total solar radiation in the ultraviolet range (280-400 nm) (0-1) [previously: puv]
    UV_proportion = results["UV_proportion"]
    # Proportion of total solar radiation in the photosynthetically active range (400-700 nm) (0-1) [previously: pvis]
    PAR_proportion = results["PAR_proportion"]
    # Proportion of total solar radiation in the near-infrared range (700-3000 nm) (0-1) [previously: pnir]
    NIR_proportion = results["NIR_proportion"]
    # Fraction of UV radiation that is diffuse (scattered) rather than direct (0-1) [previously: fduv]
    UV_diffuse_fraction = results["UV_diffuse_fraction"]
    # Fraction of PAR radiation that is diffuse (scattered) rather than direct (0-1) [previously: fdvis]
    PAR_diffuse_fraction = results["PAR_diffuse_fraction"]
    # Fraction of NIR radiation that is diffuse (scattered) rather than direct (0-1) [previously: fdnir]
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
    # with the proportion of UV radiation (UV_proportion). UV radiation is a small fraction of the solar spectrum. [previously: UV]
    UV_Wm2 = SWin_Wm2 * UV_proportion

    # Calculate photosynthetically active radiation (PAR) in W/m², which represents the visible portion of the solar spectrum.
    # This is derived by scaling the total shortwave incoming radiation (SWin_Wm2) with the proportion of visible radiation (PAR_proportion). [previously: VIS, visible_Wm2]
    PAR_Wm2 = SWin_Wm2 * PAR_proportion

    # Calculate near-infrared radiation (NIR) in W/m², which represents the portion of the solar spectrum beyond visible light.
    # This is derived by scaling the total shortwave incoming radiation (SWin_Wm2) with the proportion of NIR radiation (NIR_proportion). [previously: NIR]
    NIR_Wm2 = SWin_Wm2 * NIR_proportion

    # Calculate diffuse visible radiation (PAR_diffuse_Wm2) in W/m² by scaling the total visible radiation (PAR_Wm2)
    # with the diffuse fraction of visible radiation (PAR_diffuse_fraction). The np.clip function ensures the value
    # remains within the range [0, PAR_Wm2]. Diffuse radiation is scattered sunlight that reaches the surface indirectly. [previously: VISdiff, visible_diffuse_Wm2]
    PAR_diffuse_Wm2 = np.clip(PAR_Wm2 * PAR_diffuse_fraction, 0, PAR_Wm2)

    # Calculate diffuse near-infrared radiation (NIR_diffuse_Wm2) in W/m² by scaling the total NIR radiation (NIR_Wm2)
    # with the diffuse fraction of NIR radiation (NIR_diffuse_fraction). The np.clip function ensures the value
    # remains within the range [0, NIR_Wm2]. [previously: NIRdiff]
    NIR_diffuse_Wm2 = np.clip(NIR_Wm2 * NIR_diffuse_fraction, 0, NIR_Wm2)

    # Calculate direct visible radiation (PAR_direct_Wm2) in W/m² by subtracting the diffuse visible radiation (PAR_diffuse_Wm2)
    # from the total visible radiation (PAR_Wm2). The np.clip function ensures the value remains within the range [0, PAR_Wm2].
    # Direct radiation is sunlight that reaches the surface without being scattered. [previously: VISdir, visible_direct_Wm2]
    PAR_direct_Wm2 = np.clip(PAR_Wm2 - PAR_diffuse_Wm2, 0, PAR_Wm2)

    # Calculate direct near-infrared radiation (NIR_direct_Wm2) in W/m² by subtracting the diffuse NIR radiation (NIR_diffuse_Wm2)
    # from the total NIR radiation (NIR_Wm2). The np.clip function ensures the value remains within the range [0, NIR_Wm2]. [previously: NIRdir, NIR_direct_Wm2]
    NIR_direct_Wm2 = np.clip(NIR_Wm2 - NIR_diffuse_Wm2, 0, NIR_Wm2)

    # Calculate upwelling (reflected) shortwave radiation in W/m² using broadband albedo
    # This represents the total solar radiation reflected back from the surface
    SWout_Wm2 = SWin_Wm2 * albedo

    # Partition spectral albedos using NDVI-based method (only if NDVI is provided)
    # Use NDVI-based spectral partitioning (Liang 2001, Schaaf et al. 2002)
    # This accounts for vegetation's distinct spectral signature:
    # - Low PAR reflectance due to chlorophyll absorption
    # - High NIR reflectance due to leaf cellular structure
    if NDVI is not None:
        # Determine the shape from albedo array for broadcasting NDVI if needed
        actual_shape = albedo.shape if hasattr(albedo, 'shape') else None
        NDVI_array = ensure_array(NDVI, actual_shape)
        
        PAR_albedo, NIR_albedo = partition_spectral_albedo_with_NDVI(
            broadband_albedo=albedo,
            NDVI=NDVI_array,
            PAR_proportion=PAR_proportion,
            NIR_proportion=NIR_proportion
        )
        
        # Calculate reflected radiation using spectral albedos
        PAR_reflected_Wm2 = PAR_Wm2 * PAR_albedo
        NIR_reflected_Wm2 = NIR_Wm2 * NIR_albedo
        
        # Store NDVI in results
        results["NDVI"] = NDVI

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
        SWout_Wm2 = rt.Raster(SWout_Wm2, geometry=geometry)
        
        if NDVI is not None:
            PAR_reflected_Wm2 = rt.Raster(PAR_reflected_Wm2, geometry=geometry)
            NIR_reflected_Wm2 = rt.Raster(NIR_reflected_Wm2, geometry=geometry)
            PAR_albedo = rt.Raster(PAR_albedo, geometry=geometry)
            NIR_albedo = rt.Raster(NIR_albedo, geometry=geometry)

    if isinstance(UV_Wm2, Raster):
        UV_Wm2.cmap = UV_CMAP

    # Update the results dictionary with new items instead of replacing it
    # Update the results dictionary with new items instead of replacing it
    results.update({
        "SWin_Wm2": SWin_Wm2,
        "SWin_TOA_Wm2": SWin_TOA_Wm2,
        "SWout_Wm2": SWout_Wm2,
        "UV_Wm2": UV_Wm2,
        "PAR_Wm2": PAR_Wm2,
        "NIR_Wm2": NIR_Wm2,
        "atmospheric_transmittance": atmospheric_transmittance,
        "UV_proportion": UV_proportion,
        "UV_diffuse_fraction": UV_diffuse_fraction,
        "PAR_proportion": PAR_proportion,
        "NIR_proportion": NIR_proportion,
        "PAR_diffuse_Wm2": PAR_diffuse_Wm2,
        "NIR_diffuse_Wm2": NIR_diffuse_Wm2,
        "PAR_direct_Wm2": PAR_direct_Wm2,
        "NIR_direct_Wm2": NIR_direct_Wm2,
        "PAR_diffuse_fraction": PAR_diffuse_fraction,
        "NIR_diffuse_fraction": NIR_diffuse_fraction
    })
    
    # Add NDVI-derived spectral albedo outputs only if NDVI was provided
    if NDVI is not None:
        results.update({
            "PAR_reflected_Wm2": PAR_reflected_Wm2,
            "NIR_reflected_Wm2": NIR_reflected_Wm2,
            "PAR_albedo": PAR_albedo,
            "NIR_albedo": NIR_albedo
        })
    # Convert results to Raster objects if raster geometry is given
    if isinstance(geometry, RasterGeometry):
        for key in results.keys():
            results[key] = rt.Raster(results[key], geometry=geometry)

    return results

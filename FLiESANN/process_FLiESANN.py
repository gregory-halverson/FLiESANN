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
from .retrieve_FLiESANN_GEOS5FP_inputs import retrieve_FLiESANN_GEOS5FP_inputs
from .retrieve_FLiESANN_static_inputs import retrieve_FLiESANN_static_inputs
from .ensure_array import ensure_array

def partition_spectral_albedo_with_NDVI(
        broadband_albedo: np.ndarray,
        NDVI: np.ndarray,
        PAR_proportion: np.ndarray,
        NIR_proportion: np.ndarray) -> tuple:
    """
    Partition broadband albedo into PAR and NIR spectral components using NDVI.
    
    This function implements an empirical relationship between NDVI and spectral reflectance
    properties based on peer-reviewed literature. Vegetation exhibits distinct spectral signatures
    with low reflectance in the visible (PAR) range due to chlorophyll absorption and high
    reflectance in the NIR range due to leaf cellular structure.
    
    References:
    -----------
    - Liang, S. (2001). "Narrowband to broadband conversions of land surface albedo I: Algorithms."
      Remote Sensing of Environment, 76(3), 213-238. DOI: 10.1016/S0034-4257(00)00205-4
      
    - Schaaf, C.B., et al. (2002). "First operational BRDF, albedo nadir reflectance products from MODIS."
      Remote Sensing of Environment, 83(1-2), 135-148. DOI: 10.1016/S0034-4257(02)00091-3
      
    - Wang, K., & Liang, S. (2009). "Estimation of daytime net radiation from shortwave radiation
      measurements and meteorological observations." Journal of Applied Meteorology and Climatology,
      48(3), 634-643. DOI: 10.1175/2008JAMC1959.1
      
    - Pinty, B., et al. (2006). "Simplifying the interaction of land surfaces with radiation for
      relating remote sensing products to climate models." Journal of Geophysical Research, 111, D02116.
      DOI: 10.1029/2005JD005952
    
    Empirical Relationships:
    -----------------------
    For dense vegetation (NDVI > 0.5):
        - PAR albedo: 0.03-0.10 (typically ~0.05-0.08)
        - NIR albedo: 0.30-0.50 (typically ~0.35-0.45)
        - NIR/PAR ratio: ~4-6
    
    For sparse vegetation (NDVI 0.2-0.5):
        - PAR albedo: 0.08-0.15
        - NIR albedo: 0.15-0.30
        - NIR/PAR ratio: ~1.5-3
    
    For bare soil/desert (NDVI < 0.2):
        - PAR albedo: 0.15-0.35
        - NIR albedo: 0.20-0.40
        - NIR/PAR ratio: ~1.0-1.5
    
    Args:
        broadband_albedo (np.ndarray): Broadband surface albedo (0.3-5.0 μm range)
        NDVI (np.ndarray): Normalized Difference Vegetation Index (-1 to 1)
        PAR_proportion (np.ndarray): Fraction of incoming solar radiation in PAR band (0.4-0.7 μm)
        NIR_proportion (np.ndarray): Fraction of incoming solar radiation in NIR band (0.7-3.0 μm)
    
    Returns:
        tuple: (PAR_albedo, NIR_albedo)
            - PAR_albedo (np.ndarray): Spectral albedo in photosynthetically active radiation band
            - NIR_albedo (np.ndarray): Spectral albedo in near-infrared band
    
    Notes:
        The implementation uses a continuous empirical function based on MODIS albedo products
        (Schaaf et al. 2002) and validated partitioning relationships (Liang 2001, Wang & Liang 2009).
        The spectral albedos are constrained to satisfy:
        
        broadband_albedo ≈ PAR_proportion × PAR_albedo + NIR_proportion × NIR_albedo
        
        The NIR/PAR albedo ratio increases with NDVI according to:
        ratio = 1.0 + 5.0 × NDVI^2  (for NDVI > 0)
        
        This quadratic relationship captures the nonlinear increase in NIR reflectance and decrease
        in PAR reflectance as vegetation density and health increase.
    """
    # Clip NDVI to valid range
    NDVI_clipped = np.clip(NDVI, -1, 1)
    
    # Calculate NIR/PAR albedo ratio from NDVI
    # Based on empirical relationships from Schaaf et al. (2002) and Pinty et al. (2006)
    # For vegetation, the ratio increases with NDVI as chlorophyll absorption increases in PAR
    # and cellular scattering increases in NIR
    
    # Quadratic relationship captures nonlinear vegetation spectral response
    # ratio ranges from ~1.0 (NDVI=0, bare soil) to ~6.0 (NDVI=1, dense vegetation)
    ratio = np.where(
        NDVI_clipped > 0,
        1.0 + 5.0 * NDVI_clipped**2,  # Quadratic: ratio from 1 at NDVI=0 to 6 at NDVI=1
        1.0  # For water, snow, or negative NDVI, assume similar PAR and NIR albedo
    )
    
    # Partition broadband albedo into spectral components
    # Constraint: broadband_albedo ≈ PAR_proportion × PAR_albedo + NIR_proportion × NIR_albedo
    # Substituting NIR_albedo = ratio × PAR_albedo:
    # broadband_albedo = PAR_proportion × PAR_albedo + NIR_proportion × ratio × PAR_albedo
    # broadband_albedo = PAR_albedo × (PAR_proportion + NIR_proportion × ratio)
    # Therefore: PAR_albedo = broadband_albedo / (PAR_proportion + NIR_proportion × ratio)
    
    denominator = PAR_proportion + NIR_proportion * ratio
    PAR_albedo = np.where(denominator > 0, broadband_albedo / denominator, broadband_albedo)
    NIR_albedo = PAR_albedo * ratio
    
    # Clip to physical range [0, 1]
    PAR_albedo = np.clip(PAR_albedo, 0, 1)
    NIR_albedo = np.clip(NIR_albedo, 0, 1)
    
    return PAR_albedo, NIR_albedo

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
        zero_COT_correction: bool = ZERO_COT_CORRECTION) -> dict:
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
    
    # Store in results
    results["elevation_m"] = elevation_m
    results["KG_climate"] = KG_climate

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
        zero_COT_correction=zero_COT_correction
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
    import pandas as pd
    
    # Helper function to filter DataFrame to match location-time pairs
    def filter_dataframe_to_location_time_pairs(df, geometry, time_UTC):
        """Filter DataFrame returned from GEOS5FP to match original location-time pairs"""
        if not isinstance(df, pd.DataFrame) or len(df) == len(geometry.geoms):
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
            if row_idx < len(df):
                selected_rows.append(row_idx)
        
        if len(selected_rows) == len(input_coords):
            return df.iloc[selected_rows, 0].values.astype(np.float32)
        else:
            # Fallback to first column if filtering fails
            return df.iloc[:, 0].values.astype(np.float32)
    
    if isinstance(COT, pd.DataFrame):
        COT = filter_dataframe_to_location_time_pairs(COT, geometry, time_UTC)
    if isinstance(AOT, pd.DataFrame):
        AOT = filter_dataframe_to_location_time_pairs(AOT, geometry, time_UTC)
    if isinstance(vapor_gccm, pd.DataFrame):
        vapor_gccm = filter_dataframe_to_location_time_pairs(vapor_gccm, geometry, time_UTC)
    if isinstance(ozone_cm, pd.DataFrame):
        ozone_cm = filter_dataframe_to_location_time_pairs(ozone_cm, geometry, time_UTC)
    
    # Store in results (after DataFrame conversion)
    results["COT"] = COT
    results["AOT"] = AOT
    results["vapor_gccm"] = vapor_gccm
    results["ozone_cm"] = ozone_cm
    
    # Update shape based on actual retrieved data arrays (after DataFrame conversion)
    # This handles cases where time-series data creates additional dimensions
    # Skip DataFrames when determining shape - they should have been converted above
    import pandas as pd
    if hasattr(COT, 'shape') and not isinstance(COT, pd.DataFrame):
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

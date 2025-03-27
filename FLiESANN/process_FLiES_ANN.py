from typing import Union
from time import process_time
from datetime import datetime
import numpy as np
import rasters as rt
from rasters import Raster, RasterGeometry
from geos5fp import GEOS5FP
from solar_apparent_time import solar_day_of_year_for_area, solar_hour_of_day_for_area
from sun_angles import calculate_SZA_from_DOY_and_hour
from koppengeiger import load_koppen_geiger
from NASADEM import NASADEM, NASADEMConnection

from .constants import *
from .determine_atype import determine_atype
from .determine_ctype import determine_ctype
from .run_FLiES_ANN_inference import run_FLiES_ANN_inference

def FLiESANN(
        albedo: Union[Raster, np.ndarray],
        COT: Union[Raster, np.ndarray] = None,
        AOT: Union[Raster, np.ndarray] = None,
        vapor_gccm: Union[Raster, np.ndarray] = None,
        ozone_cm: Union[Raster, np.ndarray] = None,
        elevation_km: Union[Raster, np.ndarray] = None,
        SZA: Union[Raster, np.ndarray] = None,
        KG_climate: Union[Raster, np.ndarray] = None,
        geometry: RasterGeometry = None,
        time_UTC: datetime = None,
        day_of_year: Union[Raster, np.ndarray] = None,
        hour_of_day: Union[Raster, np.ndarray] = None,
        GEOS5FP_connection: GEOS5FP = None,
        NASADEM_connection: NASADEMConnection = NASADEM,
        resampling: str = "cubic",
        ANN_model=None,
        model_filename=DEFAULT_MODEL_FILENAME,
        split_atypes_ctypes=SPLIT_ATYPES_CTYPES) -> dict:
    """
    Processes Forest Light Environmental Simulator (FLiES) calculations using an 
    artificial neural network (ANN) emulator.

    This function takes various atmospheric and environmental parameters as input,
    including day of year, albedo, cloud optical thickness (COT), aerosol optical 
    thickness (AOT), water vapor, ozone, elevation, solar zenith angle (SZA), and 
    Köppen-Geiger climate classification. It uses these inputs to estimate radiative 
    transfer components such as total transmittance, diffuse and direct radiation 
    in different spectral bands (UV, visible, near-infrared).

    Args:
        doy: Day of year (Raster or np.ndarray).
        albedo: Surface albedo (Raster or np.ndarray).
        COT: Cloud optical thickness (Raster or np.ndarray).
        AOT: Aerosol optical thickness (Raster or np.ndarray).
        vapor_gccm: Water vapor in grams per square centimeter (Raster or np.ndarray).
        ozone_cm: Ozone concentration in centimeters (Raster or np.ndarray).
        elevation_km: Elevation in kilometers (Raster or np.ndarray).
        SZA: Solar zenith angle (Raster or np.ndarray).
        KG_climate: Köppen-Geiger climate classification (Raster or np.ndarray).
        geometry: RasterGeometry object defining the spatial extent and resolution.
        GEOS5FP_connection: GEOS5FP object for accessing GEOS-5 FP data.
        GEOS5FP_directory: Directory containing GEOS-5 FP data files.
        ANN_model: Pre-loaded ANN model object. If None, it's loaded from the file.
        model_filename: Filename of the ANN model to load.
        split_atypes_ctypes: Boolean flag for handling aerosol and cloud types.

    Returns:
        dict: A dictionary containing the calculated radiative transfer components 
              as Raster objects or np.ndarrays, including:
              - Ra: Extraterrestrial solar radiation.
              - Rg: Global solar radiation.
              - UV: Ultraviolet radiation.
              - VIS: Visible radiation.
              - NIR: Near-infrared radiation.
              - VISdiff: Diffuse visible radiation.
              - NIRdiff: Diffuse near-infrared radiation.
              - VISdir: Direct visible radiation.
              - NIRdir: Direct near-infrared radiation.
              - tm: Total transmittance.
              - puv: Proportion of UV radiation.
              - pvis: Proportion of visible radiation.
              - pnir: Proportion of NIR radiation.
              - fduv: Diffuse fraction of UV radiation.
              - fdvis: Diffuse fraction of visible radiation.
              - fdnir: Diffuse fraction of NIR radiation.
    """

    if geometry is None and isinstance(albedo, Raster):
        geometry = albedo.geometry

    if (day_of_year is None or hour_of_day is None) and time_UTC is not None and geometry is not None:
        day_of_year = solar_day_of_year_for_area(time_UTC=time_UTC, geometry=geometry)
        hour_of_day = solar_hour_of_day_for_area(time_UTC=time_UTC, geometry=geometry)

    if time_UTC is None and day_of_year is None and hour_of_day is None:
        raise ValueError("no time given between time_UTC, day_of_year, and hour_of_day")

    if GEOS5FP_connection is None:
        GEOS5FP_connection = GEOS5FP(working_directory=DEFAULT_WORKING_DIRECTORY, download_directory=GEOS5FP_DIRECTORY)

    ## FIXME need to fetch default values for parameters: COT, AOT, vapor_gccm, ozone_cm, elevation_km, SZA, KG_climate 

    if SZA is None and geometry is not None:
        SZA = calculate_SZA_from_DOY_and_hour(
            lat=geometry.lat,
            lon=geometry.lon,
            DOY=day_of_year,
            hour=hour_of_day
        )

    if SZA is None:
        raise ValueError("solar zenith angle or geometry must be given")

    if KG_climate is None and geometry is not None:
        KG_climate = load_koppen_geiger(geometry=geometry)

    if KG_climate is None:
        raise ValueError("Koppen Geieger climate classification or geometry must be given")

    if COT is None and geometry is not None and time_UTC is not None:
        COT = GEOS5FP_connection.COT(
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling
        )
    
    if AOT is None and geometry is not None and time_UTC is not None:
        AOT = GEOS5FP_connection.AOT(
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling
        )

    if vapor_gccm is None and geometry is not None and time_UTC is not None:
        vapor_gccm = GEOS5FP_connection.vapor_gccm(
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling
        )

    if ozone_cm is None and geometry is not None and time_UTC is not None:
        ozone_cm = GEOS5FP_connection.ozone_cm(
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling
        )

    if elevation_km is None and geometry is not None:
        elevation_km = NASADEM.elevation_km(geometry=geometry)

    # Preprocess COT and determine aerosol/cloud types
    COT = np.clip(COT, 0, None)  # Ensure COT is non-negative
    COT = rt.where(COT < 0.001, 0, COT)  # Set very small COT values to 0
    atype = determine_atype(KG_climate, COT)  # Determine aerosol type
    ctype = determine_ctype(KG_climate, COT)  # Determine cloud type

    # Run ANN inference to get initial radiative transfer parameters
    prediction_start_time = process_time()
    results = run_FLiES_ANN_inference(
        atype=atype,
        ctype=ctype,
        COT=COT,
        AOT=AOT,
        vapor_gccm=vapor_gccm,
        ozone_cm=ozone_cm,
        albedo=albedo,
        elevation_km=elevation_km,
        SZA=SZA,
        ANN_model=ANN_model,
        model_filename=model_filename,
        split_atypes_ctypes=split_atypes_ctypes
    )
    prediction_end_time = process_time()
    prediction_duration = prediction_end_time - prediction_start_time

    # Extract individual components from the results dictionary
    tm = results['tm']
    puv = results['puv']
    pvis = results['pvis']
    pnir = results['pnir']
    fduv = results['fduv']
    fdvis = results['fdvis']
    fdnir = results['fdnir']

    ## Correction for diffuse PAR
    COT = rt.where(COT == 0.0, np.nan, COT)
    COT = rt.where(np.isfinite(COT), COT, np.nan)
    x = np.log(COT)
    p1 = 0.05088
    p2 = 0.04909
    p3 = 0.5017
    corr = np.array(p1 * x * x + p2 * x + p3)
    corr[np.logical_or(np.isnan(corr), corr > 1.0)] = 1.0
    fdvis = fdvis * corr * 0.915

    ## Radiation components
    dr = 1.0 + 0.033 * np.cos(2 * np.pi / 365.0 * day_of_year)  # Earth-sun distance correction factor
    Ra = 1333.6 * dr * np.cos(SZA * np.pi / 180.0)  # Extraterrestrial radiation
    Ra = rt.where(SZA > 90.0, 0, Ra)  # Set Ra to 0 when the sun is below the horizon
    Rg = Ra * tm  # Global radiation
    UV = Rg * puv  # Ultraviolet radiation
    VIS = Rg * pvis  # Visible radiation
    NIR = Rg * pnir  # Near-infrared radiation
    VISdiff = VIS * fdvis  # Diffuse visible radiation
    NIRdiff = NIR * fdnir  # Diffuse near-infrared radiation
    VISdir = VIS - VISdiff  # Direct visible radiation
    NIRdir = NIR - NIRdiff  # Direct near-infrared radiation

    # Store the results in a dictionary
    results = {
        "Ra": Ra,
        "Rg": Rg,
        "UV": UV,
        "VIS": VIS,
        "NIR": NIR,
        "VISdiff": VISdiff,
        "NIRdiff": NIRdiff,
        "VISdir": VISdir,
        "NIRdir": NIRdir,
        "tm": tm,
        "puv": puv,
        "pvis": pvis,
        "pnir": pnir,
        "fduv": fduv,
        "fdvis": fdvis,
        "fdnir": fdnir
    }

    # Convert results to Raster objects if raster geometry is given
    if geometry is not None:
        for key in results.keys():
            results[key] = rt.Raster(results[key], geometry=geometry)

    return results

FLiESANN = FLiESANN

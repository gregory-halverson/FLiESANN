from typing import Union
from datetime import datetime
import numpy as np
import rasters as rt
from rasters import Raster, RasterGeometry
from GEOS5FP import GEOS5FP
import shapely

from .constants import *

class MissingOfflineParameter(Exception):
    """Custom exception for missing parameters in offline mode."""
    pass

def retrieve_FLiESANN_GEOS5FP_inputs(
        COT: Union[Raster, np.ndarray, float] = None,
        AOT: Union[Raster, np.ndarray, float] = None,
        vapor_gccm: Union[Raster, np.ndarray, float] = None,
        ozone_cm: Union[Raster, np.ndarray, float] = None,
        geometry: Union[RasterGeometry, shapely.geometry.Point, rt.Point, shapely.geometry.MultiPoint, rt.MultiPoint] = None,
        time_UTC: datetime = None,
        GEOS5FP_connection: GEOS5FP = None,
        resampling: str = DEFAULT_RESAMPLING,
        zero_COT_correction: bool = ZERO_COT_CORRECTION,
        offline_mode: bool = False) -> dict:
    """
    Retrieve GEOS-5 FP atmospheric inputs for FLiESANN model.

    This function retrieves atmospheric parameters from GEOS-5 FP data if they are not
    already provided. Parameters that are given as input are passed through unchanged.

    Args:
        COT (Union[Raster, np.ndarray, float], optional): Cloud optical thickness. 
            If None and geometry/time_UTC are provided, will be retrieved from GEOS-5 FP.
        AOT (Union[Raster, np.ndarray, float], optional): Aerosol optical thickness.
            If None and geometry/time_UTC are provided, will be retrieved from GEOS-5 FP.
        vapor_gccm (Union[Raster, np.ndarray, float], optional): Water vapor in grams per square centimeter.
            If None and geometry/time_UTC are provided, will be retrieved from GEOS-5 FP.
        ozone_cm (Union[Raster, np.ndarray, float], optional): Ozone concentration in centimeters.
            If None and geometry/time_UTC are provided, will be retrieved from GEOS-5 FP.
        geometry (Union[RasterGeometry, Point, MultiPoint], optional): Spatial geometry for data retrieval.
        time_UTC (datetime, optional): UTC time for data retrieval.
        GEOS5FP_connection (GEOS5FP, optional): Connection to GEOS-5 FP data. If None, a new connection will be created.
        resampling (str, optional): Resampling method for raster data. Defaults to "cubic".
        zero_COT_correction (bool, optional): If True, sets COT to zero (clear sky conditions). Defaults to False.
        offline_mode (bool, optional): If True, raises MissingOfflineParameter for missing parameters instead of retrieving them. Defaults to False.

    Returns:
        dict: Dictionary containing the atmospheric inputs with keys:
            - COT: Cloud optical thickness
            - AOT: Aerosol optical thickness
            - vapor_gccm: Water vapor
            - ozone_cm: Ozone concentration

    Raises:
        ValueError: If a parameter cannot be retrieved and is required.
        MissingOfflineParameter: If offline_mode is True and a required parameter is missing.
    """
    results = {}

    if GEOS5FP_connection is None:
        GEOS5FP_connection = GEOS5FP()

    # Convert rasters.MultiPoint to shapely.geometry.MultiPoint if needed
    query_geometry = geometry
    if isinstance(geometry, rt.MultiPoint):
        # Extract coordinates from rasters.MultiPoint and create shapely.MultiPoint
        coords = [(point.x, point.y) for point in geometry.geoms]
        query_geometry = shapely.geometry.MultiPoint(coords)
    elif isinstance(geometry, rt.Point):
        # Convert rasters.Point to shapely.Point
        query_geometry = shapely.geometry.Point(geometry.x, geometry.y)

    # Retrieve or validate COT
    if zero_COT_correction:
        # Determine shape for zero array
        if isinstance(geometry, (Raster, np.ndarray)):
            shape = geometry.shape
        elif isinstance(geometry, (shapely.geometry.MultiPoint, rt.MultiPoint)):
            shape = (len(geometry.geoms),) if hasattr(geometry, 'geoms') else (len(geometry),)
        else:
            shape = (1,)
        COT = np.zeros(shape, dtype=np.float32)
    elif COT is None:
        if offline_mode:
            raise MissingOfflineParameter("COT is required in offline mode but not provided.")
        if geometry is not None and time_UTC is not None:
            COT = GEOS5FP_connection.COT(
                time_UTC=time_UTC,
                geometry=query_geometry,
                resampling=resampling
            )

    if COT is None:
        raise ValueError("cloud optical thickness or geometry and time must be given")

    # Constrain COT
    COT = rt.clip(COT, 0, None)  # Ensure COT is non-negative
    COT = rt.where(COT < 0.001, 0, COT)  # Set very small COT values to 0
    results["COT"] = COT

    # Retrieve or validate AOT
    if AOT is None:
        if offline_mode:
            raise MissingOfflineParameter("AOT is required in offline mode but not provided.")
        if geometry is not None and time_UTC is not None:
            AOT = GEOS5FP_connection.AOT(
                time_UTC=time_UTC,
                geometry=query_geometry,
                resampling=resampling
            )

    if AOT is None:
        raise ValueError("aerosol optical thickness or geometry and time must be given")

    results["AOT"] = AOT

    # Retrieve or validate water vapor
    if vapor_gccm is None:
        if offline_mode:
            raise MissingOfflineParameter("Water vapor is required in offline mode but not provided.")
        if geometry is not None and time_UTC is not None:
            vapor_gccm = GEOS5FP_connection.vapor_gccm(
                time_UTC=time_UTC,
                geometry=query_geometry,
                resampling=resampling
            )

    if vapor_gccm is None:
        raise ValueError("water vapor or geometry and time must be given")

    results["vapor_gccm"] = vapor_gccm

    # Retrieve or validate ozone
    if ozone_cm is None:
        if offline_mode:
            raise MissingOfflineParameter("Ozone concentration is required in offline mode but not provided.")
        if geometry is not None and time_UTC is not None:
            ozone_cm = GEOS5FP_connection.ozone_cm(
                time_UTC=time_UTC,
                geometry=query_geometry,
                resampling=resampling
            )

    if ozone_cm is None:
        raise ValueError("ozone concentration or geometry and time must be given")

    results["ozone_cm"] = ozone_cm

    return results

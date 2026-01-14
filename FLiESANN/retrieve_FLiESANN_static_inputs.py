from typing import Union
import numpy as np
import rasters as rt
from rasters import Raster, RasterGeometry
from koppengeiger import load_koppen_geiger
from NASADEM import NASADEM, NASADEMConnection
import shapely


def retrieve_FLiESANN_static_inputs(
        elevation_m: Union[Raster, np.ndarray, float] = None,
        KG_climate: Union[Raster, np.ndarray, int] = None,
        geometry: Union[RasterGeometry, shapely.geometry.Point, rt.Point, shapely.geometry.MultiPoint, rt.MultiPoint] = None,
        NASADEM_connection: NASADEMConnection = NASADEM,
        resampling: str = "cubic") -> dict:
    """
    Retrieve static inputs for FLiESANN model.
    
    This function retrieves static geographic parameters (elevation and climate classification)
    if they are not already provided. Parameters that are given as input are passed through unchanged.
    
    Args:
        elevation_m (Union[Raster, np.ndarray, float], optional): Elevation in meters.
            If None and geometry is provided, will be retrieved from NASADEM.
        KG_climate (Union[Raster, np.ndarray, int], optional): Köppen-Geiger climate classification.
            If None and geometry is provided, will be retrieved from Köppen-Geiger dataset.
        geometry (Union[RasterGeometry, Point, MultiPoint], optional): Spatial geometry for data retrieval.
        NASADEM_connection (NASADEMConnection, optional): Connection to NASADEM data. Defaults to NASADEM.
        resampling (str, optional): Resampling method for raster data. Defaults to "cubic".
    
    Returns:
        dict: Dictionary containing the static inputs with keys:
            - elevation_m: Elevation in meters
            - elevation_km: Elevation in kilometers
            - KG_climate: Köppen-Geiger climate classification
    
    Raises:
        ValueError: If a parameter cannot be retrieved and is required.
    """
    results = {}
    
    # Retrieve or validate elevation
    if elevation_m is None and geometry is not None:
        elevation_km = NASADEM_connection.elevation_km(geometry=geometry)
        elevation_m = elevation_km * 1000.0
    elif elevation_m is not None:
        elevation_km = elevation_m / 1000.0
    else:
        raise ValueError("elevation or geometry must be given")
    
    results["elevation_m"] = elevation_m
    results["elevation_km"] = elevation_km
    
    # Retrieve or validate Köppen-Geiger climate
    if KG_climate is None and geometry is not None:
        KG_climate = load_koppen_geiger(geometry=geometry)
    
    if KG_climate is None:
        raise ValueError("Köppen-Geiger climate classification or geometry must be given")
    
    results["KG_climate"] = KG_climate
    
    return results

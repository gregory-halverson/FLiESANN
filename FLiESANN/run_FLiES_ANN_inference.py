import numpy as np

from .constants import *
from .load_FLiES_model import load_FLiES_model
from .prepare_FLiES_ANN_inputs import prepare_FLiES_ANN_inputs

def run_FLiES_ANN_inference(
        atype: np.ndarray,
        ctype: np.ndarray,
        COT: np.ndarray,
        AOT: np.ndarray,
        vapor_gccm: np.ndarray,
        ozone_cm: np.ndarray,
        albedo: np.ndarray,
        elevation_km: np.ndarray,
        SZA: np.ndarray,
        ANN_model=None,
        model_filename=DEFAULT_MODEL_FILENAME,
        split_atypes_ctypes=SPLIT_ATYPES_CTYPES) -> dict:
    """
    Runs inference for an artificial neural network (ANN) emulator of the Forest Light
    Environmental Simulator (FLiES) radiative transfer model.

    This function takes atmospheric parameters as input, preprocesses them, and uses a 
    trained ANN model to predict radiative transfer outputs such as transmittance 
    and diffuse fraction.

    Args:
        atype: Aerosol type (np.ndarray).
        ctype: Cloud type (np.ndarray).
        COT: Cloud optical thickness (np.ndarray).
        AOT: Aerosol optical thickness (np.ndarray).
        vapor_gccm: Water vapor in grams per square centimeter (np.ndarray).
        ozone_cm: Ozone concentration in centimeters (np.ndarray).
        albedo: Surface albedo (reflectivity) (np.ndarray).
        elevation_km: Elevation in kilometers (np.ndarray).
        SZA: Solar zenith angle (np.ndarray).
        ANN_model: Optional pre-loaded ANN model object. If None, the model is loaded 
                   from the specified file.
        model_filename: Filename of the ANN model to load if ANN_model is not provided.
        split_atypes_ctypes: Boolean flag indicating how aerosol and cloud types are 
                             handled in input preparation.

    Returns:
        dict: A dictionary containing the predicted radiative transfer parameters:
              - 'tm': Total transmittance (np.ndarray).
              - 'puv': Proportion of radiation in the ultraviolet band (np.ndarray).
              - 'pvis': Proportion of radiation in the visible band (np.ndarray).
              - 'pnir': Proportion of radiation in the near-infrared band (np.ndarray).
              - 'fduv': Diffuse fraction of radiation in the ultraviolet band (np.ndarray).
              - 'fdvis': Diffuse fraction of radiation in the visible band (np.ndarray).
              - 'fdnir': Diffuse fraction of radiation in the near-infrared band (np.ndarray).
    """
    
    if ANN_model is None:
        # Load the ANN model if not provided
        ANN_model = load_FLiES_model(model_filename)

    # Prepare inputs for the ANN model
    inputs = prepare_FLiES_ANN_inputs(
        atype=atype,
        ctype=ctype,
        COT=COT,
        AOT=AOT,
        vapor_gccm=vapor_gccm,
        ozone_cm=ozone_cm,
        albedo=albedo,
        elevation_km=elevation_km,
        SZA=SZA,
        split_atypes_ctypes=split_atypes_ctypes
    )

    # Run inference using the ANN model
    outputs = ANN_model.predict(inputs)
    shape = COT.shape
    
    # Prepare the results dictionary
    results = {
        'tm': np.clip(outputs[:, 0].reshape(shape), 0, 1).astype(np.float32),  # Total transmittance
        'puv': np.clip(outputs[:, 1].reshape(shape), 0, 1).astype(np.float32), # Proportion of UV radiation
        'pvis': np.clip(outputs[:, 2].reshape(shape), 0, 1).astype(np.float32), # Proportion of visible radiation
        'pnir': np.clip(outputs[:, 3].reshape(shape), 0, 1).astype(np.float32), # Proportion of NIR radiation
        'fduv': np.clip(outputs[:, 4].reshape(shape), 0, 1).astype(np.float32), # Diffuse fraction of UV radiation
        'fdvis': np.clip(outputs[:, 5].reshape(shape), 0, 1).astype(np.float32), # Diffuse fraction of visible radiation
        'fdnir': np.clip(outputs[:, 6].reshape(shape), 0, 1).astype(np.float32)  # Diffuse fraction of NIR radiation
    }

    return results

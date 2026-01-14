import numpy as np
from tqdm.notebook import tqdm

from .constants import *
from .load_FLiESANN_model import load_FLiESANN_model
from .prepare_FLiESANN_inputs import prepare_FLiESANN_inputs

def run_FLiESANN_inference(
        atype: np.ndarray,
        ctype: np.ndarray,
        COT: np.ndarray,
        AOT: np.ndarray,
        vapor_gccm: np.ndarray,
        ozone_cm: np.ndarray,
        albedo: np.ndarray,
        elevation_m: np.ndarray,
        SZA: np.ndarray,
        ANN_model=None,
        model_filename=MODEL_FILENAME,
        split_atypes_ctypes=SPLIT_ATYPES_CTYPES,
        use_tqdm=False  # New parameter to toggle TQDM progress bar
) -> dict:
    """
    Runs inference for an artificial neural network (ANN) emulator of the Forest Light
    Environmental Simulator (FLiES) radiative transfer model.

    This function takes atmospheric and surface parameters as input, preprocesses them, and uses a 
    trained ANN model to predict radiative transfer outputs such as transmittance 
    and diffuse fraction.

    Args:
        atype (np.ndarray): Aerosol type.
        ctype (np.ndarray): Cloud type.
        COT (np.ndarray): Cloud optical thickness.
        AOT (np.ndarray): Aerosol optical thickness.
        vapor_gccm (np.ndarray): Water vapor in grams per square centimeter.
        ozone_cm (np.ndarray): Ozone concentration in centimeters.
        albedo (np.ndarray): Surface albedo (reflectivity).
        elevation_m (np.ndarray): Elevation in meters.
        SZA (np.ndarray): Solar zenith angle.
        ANN_model (optional): Pre-loaded ANN model object. If None, the model is loaded 
                              from the specified file.
        model_filename (str, optional): Filename of the ANN model to load if ANN_model is not provided.
        split_atypes_ctypes (bool, optional): Flag indicating how aerosol and cloud types are 
                                             handled in input preparation.
        use_tqdm (bool, optional): Flag to enable or disable the TQDM progress bar for predictions.

    Returns:
        dict: A dictionary containing the predicted radiative transfer parameters:
              - 'atmospheric_transmittance' (np.ndarray): Total transmittance.
              - 'UV_proportion' (np.ndarray): Proportion of radiation in the ultraviolet band.
              - 'PAR_proportion' (np.ndarray): Proportion of radiation in the visible band (Photosynthetically Active Radiation).
              - 'NIR_proportion' (np.ndarray): Proportion of radiation in the near-infrared band.
              - 'UV_diffuse_fraction' (np.ndarray): Diffuse fraction of radiation in the ultraviolet band.
              - 'PAR_diffuse_fraction' (np.ndarray): Diffuse fraction of radiation in the visible band.
              - 'NIR_diffuse_fraction' (np.ndarray): Diffuse fraction of radiation in the near-infrared band.

    Raises:
        ValueError: If the input data shapes are incompatible with the model.

    Notes:
        - The function automatically adjusts the input shape to match the model's expected input dimensions.
        - TensorFlow warnings and logs are suppressed during model loading and inference.
    """
    import os
    import warnings
    # Save current TF_CPP_MIN_LOG_LEVEL and TF logger level
    old_tf_log_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL', None)
    try:
        import tensorflow as tf
        old_logger_level = tf.get_logger().level
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.get_logger().setLevel('ERROR')
    except Exception:
        old_logger_level = None

    try:
        if ANN_model is None:
            # Load the ANN model if not provided
            ANN_model = load_FLiESANN_model(model_filename)

        # Ensure all inputs are of numerical type
        atype = np.asarray(atype, dtype=np.float32)
        ctype = np.asarray(ctype, dtype=np.float32)
        COT = np.asarray(COT, dtype=np.float32)
        AOT = np.asarray(AOT, dtype=np.float32)
        vapor_gccm = np.asarray(vapor_gccm, dtype=np.float32)
        ozone_cm = np.asarray(ozone_cm, dtype=np.float32)
        albedo = np.asarray(albedo, dtype=np.float32)
        elevation_m = np.asarray(elevation_m, dtype=np.float32)
        SZA = np.asarray(SZA, dtype=np.float32)

        # Check for NaN values and create a mask
        nan_mask = np.isnan(atype) | np.isnan(ctype) | np.isnan(COT) | \
                   np.isnan(AOT) | np.isnan(vapor_gccm) | np.isnan(ozone_cm) | \
                   np.isnan(albedo) | np.isnan(elevation_m) | np.isnan(SZA)

        # Replace NaNs with a placeholder value (e.g., 0) for processing
        atype = np.where(nan_mask, 0, atype)
        ctype = np.where(nan_mask, 0, ctype)
        COT = np.where(nan_mask, 0, COT)
        AOT = np.where(nan_mask, 0, AOT)
        vapor_gccm = np.where(nan_mask, 0, vapor_gccm)
        ozone_cm = np.where(nan_mask, 0, ozone_cm)
        albedo = np.where(nan_mask, 0, albedo)
        elevation_m = np.where(nan_mask, 0, elevation_m)
        SZA = np.where(nan_mask, 0, SZA)

        # Convert elevation from meters to kilometers after all array processing
        elevation_km = elevation_m / 1000.0

        # Prepare inputs for the ANN model
        inputs = prepare_FLiESANN_inputs(
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

        # Ensure all columns in the DataFrame are numerical
        inputs = inputs.astype(np.float32)

        # Convert DataFrame to numpy array and reshape for the model
        inputs_array = inputs.values

        # Check what input shape the model expects and adapt accordingly
        # Different TensorFlow/Keras versions may have different input requirements
        try:
            model_input_shape = ANN_model.input_shape
            if len(model_input_shape) == 3:
                # Model expects 3D input: (batch_size, sequence_length, features)
                # Reshape from (batch_size, features) to (batch_size, 1, features)
                inputs_array = inputs_array.reshape(inputs_array.shape[0], 1, inputs_array.shape[1])
                expects_3d = True
            elif len(model_input_shape) == 2:
                # Model expects 2D input: (batch_size, features)
                # Keep the original 2D shape
                expects_3d = False
            else:
                # Fallback: try 2D first
                expects_3d = False
        except (AttributeError, TypeError):
            # If input_shape is not available, try 2D first
            expects_3d = False

        # Run inference using the ANN model with warnings suppressed
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if use_tqdm:
                    # Use TQDM progress bar for predictions
                    outputs = []
                    for batch in tqdm(inputs_array, desc="Running Inference", unit="batch"):
                        batch_output = ANN_model.predict(batch[None, ...])  # Add batch dimension
                        outputs.append(batch_output)

                    outputs = np.vstack(outputs)  # Combine all batch outputs
                else:
                    # Run prediction without progress bar
                    outputs = ANN_model.predict(inputs_array)
        except ValueError as e:
            error_msg = str(e)
            if not expects_3d and ("expected shape" in error_msg or "incompatible" in error_msg):
                # Try reshaping to 3D if 2D failed
                inputs_array = inputs.values  # Reset to original 2D shape
                inputs_array = inputs_array.reshape(inputs_array.shape[0], 1, inputs_array.shape[1])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    if use_tqdm:
                        # Use TQDM progress bar for predictions
                        outputs = []
                        for batch in tqdm(inputs_array, desc="Running Inference", unit="batch"):
                            batch_output = ANN_model.predict(batch[None, ...])  # Add batch dimension
                            outputs.append(batch_output)

                        outputs = np.vstack(outputs)  # Combine all batch outputs
                    else:
                        # Run prediction without progress bar
                        outputs = ANN_model.predict(inputs_array)
                expects_3d = True
            else:
                raise e

        # Handle output dimensions based on input dimensions used
        if expects_3d and len(outputs.shape) == 3:
            outputs = outputs.squeeze(axis=1)

        shape = COT.shape

        # Prepare the results dictionary
        results = {
            'atmospheric_transmittance': np.where(nan_mask, np.nan, np.clip(outputs[:, 0].reshape(shape), 0, 1).astype(np.float32)),  # Total transmittance
            'UV_proportion': np.where(nan_mask, np.nan, np.clip(outputs[:, 1].reshape(shape), 0, 1).astype(np.float32)), # Proportion of UV radiation
            'PAR_proportion': np.where(nan_mask, np.nan, np.clip(outputs[:, 2].reshape(shape), 0, 1).astype(np.float32)), # Proportion of visible radiation
            'NIR_proportion': np.where(nan_mask, np.nan, np.clip(outputs[:, 3].reshape(shape), 0, 1).astype(np.float32)), # Proportion of NIR radiation
            'UV_diffuse_fraction': np.where(nan_mask, np.nan, np.clip(outputs[:, 4].reshape(shape), 0, 1).astype(np.float32)), # Diffuse fraction of UV radiation
            'PAR_diffuse_fraction': np.where(nan_mask, np.nan, np.clip(outputs[:, 5].reshape(shape), 0, 1).astype(np.float32)), # Diffuse fraction of visible radiation
            'NIR_diffuse_fraction': np.where(nan_mask, np.nan, np.clip(outputs[:, 6].reshape(shape), 0, 1).astype(np.float32))  # Diffuse fraction of NIR radiation
        }

        return results
    finally:
        # Restore previous TF_CPP_MIN_LOG_LEVEL and logger level
        if old_tf_log_level is not None:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = old_tf_log_level
        else:
            if 'TF_CPP_MIN_LOG_LEVEL' in os.environ:
                del os.environ['TF_CPP_MIN_LOG_LEVEL']
        try:
            import tensorflow as tf
            if old_logger_level is not None:
                tf.get_logger().setLevel(old_logger_level)
        except Exception:
            pass

import numpy as np

from .constants import *
from .load_FLiES_model import load_FLiES_model
from .prepare_FLiES_ANN_inputs import prepare_FLiES_ANN_inputs

def process_FLiES_ANN(
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
        split_atypes_ctypes=SPLIT_ATYPES_CTYPES) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    if ANN_model is None:
        # ANN_model = load_model(DEFAULT_MODEL_FILENAME)
        ANN_model = load_FLiES_model(model_filename)

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

    outputs = ANN_model.predict(inputs)
    shape = COT.shape
    tm = np.clip(outputs[:, 0].reshape(shape), 0, 1).astype(np.float32)
    puv = np.clip(outputs[:, 1].reshape(shape), 0, 1).astype(np.float32)
    pvis = np.clip(outputs[:, 2].reshape(shape), 0, 1).astype(np.float32)
    pnir = np.clip(outputs[:, 3].reshape(shape), 0, 1).astype(np.float32)
    fduv = np.clip(outputs[:, 4].reshape(shape), 0, 1).astype(np.float32)
    fdvis = np.clip(outputs[:, 5].reshape(shape), 0, 1).astype(np.float32)
    fdnir = np.clip(outputs[:, 6].reshape(shape), 0, 1).astype(np.float32)

    return tm, puv, pvis, pnir, fduv, fdvis, fdnir

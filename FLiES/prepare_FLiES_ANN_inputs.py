import numpy as np
import pandas as pd

from .constants import SPLIT_ATYPES_CTYPES

def prepare_FLiES_ANN_inputs(
        atype: np.ndarray,
        ctype: np.ndarray,
        COT: np.ndarray,
        AOT: np.ndarray,
        vapor_gccm: np.ndarray,
        ozone_cm: np.ndarray,
        albedo: np.ndarray,
        elevation_km: np.ndarray,
        SZA: np.ndarray,
        split_atypes_ctypes=SPLIT_ATYPES_CTYPES) -> pd.DataFrame:
    ctype_flat = np.array(ctype).flatten()
    atype_flat = np.array(atype).flatten()
    COT_flat = np.array(COT).flatten()
    AOT_flat = np.array(AOT).flatten()
    vapor_gccm_flat = np.array(vapor_gccm).flatten()
    ozone_cm_flat = np.array(ozone_cm).flatten()
    albedo_flat = np.array(albedo).flatten()
    elevation_km_flat = np.array(elevation_km).flatten()
    SZA_flat = np.array(SZA).flatten()

    inputs = pd.DataFrame({
        "ctype": ctype_flat,
        "atype": atype_flat,
        "COT": COT_flat,
        "AOT": AOT_flat,
        "vapor_gccm": vapor_gccm_flat,
        "ozone_cm": ozone_cm_flat,
        "albedo": albedo_flat,
        "elevation_km": elevation_km_flat,
        "SZA": SZA_flat
    })

    if split_atypes_ctypes:
        inputs["ctype0"] = np.float32(inputs.ctype == 0)
        inputs["ctype1"] = np.float32(inputs.ctype == 1)
        inputs["ctype3"] = np.float32(inputs.ctype == 3)
        inputs["atype1"] = np.float32(inputs.ctype == 1)
        inputs["atype2"] = np.float32(inputs.ctype == 2)
        inputs["atype4"] = np.float32(inputs.ctype == 4)
        inputs["atype5"] = np.float32(inputs.ctype == 5)

        inputs = inputs[
            ["ctype0", "ctype1", "ctype3", "atype1", "atype2", "atype4", "atype5", "COT", "AOT", "vapor_gccm",
            "ozone_cm", "albedo", "elevation_km", "SZA"]]
    
    return inputs

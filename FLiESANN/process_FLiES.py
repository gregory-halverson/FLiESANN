from time import process_time

import numpy as np
import rasters as rt

from .determine_atype import determine_atype
from .determine_ctype import determine_ctype
from .process_FLiES_ANN import process_FLiES_ANN

def process_FLiES(
        doy: np.ndarray,
        albedo: np.ndarray,
        COT: np.ndarray = None,
        AOT: np.ndarray = None,
        vapor_gccm: np.ndarray = None,
        ozone_cm: np.ndarray = None,
        elevation_km: np.ndarray = None,
        SZA: np.ndarray = None,
        KG_climate: np.ndarray = None):
    COT = np.clip(COT, 0, None)
    COT = rt.where(COT < 0.001, 0, COT)
    atype = determine_atype(KG_climate, COT)
    ctype = determine_ctype(KG_climate, COT)

    prediction_start_time = process_time()
    tm, puv, pvis, pnir, fduv, fdvis, fdnir = process_FLiES_ANN(
        atype=atype,
        ctype=ctype,
        COT=COT,
        AOT=AOT,
        vapor_gccm=vapor_gccm,
        ozone_cm=ozone_cm,
        albedo=albedo,
        elevation_km=elevation_km,
        SZA=SZA
    )

    prediction_end_time = process_time()
    prediction_duration = prediction_end_time - prediction_start_time

    ##  Correction for diffuse PAR
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
    dr = 1.0 + 0.033 * np.cos(2 * np.pi / 365.0 * doy)
    Ra = 1333.6 * dr * np.cos(SZA * np.pi / 180.0)
    Ra = rt.where(SZA > 90.0, 0, Ra)
    Rg = Ra * tm
    UV = Rg * puv
    VIS = Rg * pvis
    NIR = Rg * pnir
    # UVdiff = SSR.UV * fduv
    VISdiff = VIS * fdvis
    NIRdiff = NIR * fdnir
    # UVdir = SSR.UV - UVdiff
    VISdir = VIS - VISdiff
    NIRdir = NIR - NIRdiff

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

    if isinstance(albedo, rt.Raster):
        for key in results.keys():
            results[key] = rt.Raster(results[key], geometry=albedo.geometry)

    return results

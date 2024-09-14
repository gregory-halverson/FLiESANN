import numpy as np

def determine_atype(KG_climate: np.ndarray, COT: np.ndarray, dynamic: bool = True) -> np.ndarray:
    atype = np.full(KG_climate.shape, 1, dtype=np.uint16)

    if dynamic:
        atype = np.where((COT == 0) & ((KG_climate == 5) | (KG_climate == 6)), 1, atype)
        atype = np.where((COT == 0) & ((KG_climate == 3) | (KG_climate == 4)), 2, atype)
        atype = np.where((COT == 0) & (KG_climate == 1), 4, atype)
        atype = np.where((COT == 0) & (KG_climate == 2), 5, atype)
        atype = np.where((COT > 0) & ((KG_climate == 5) | (KG_climate == 6)), 1, atype)
        atype = np.where((COT > 0) & ((KG_climate == 3) | (KG_climate == 4)), 2, atype)
        atype = np.where((COT > 0) & (KG_climate == 2), 5, atype)
        atype = np.where((COT > 0) & (KG_climate == 1), 4, atype)

    return atype

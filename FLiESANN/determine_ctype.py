import numpy as np

def determine_ctype(KG_climate: np.ndarray, COT: np.ndarray, dynamic: bool = True) -> np.ndarray:
    ctype = np.full(KG_climate.shape, 0, dtype=np.uint16)

    if dynamic:
        ctype = np.where((COT == 0) & ((KG_climate == 5) | (KG_climate == 6)), 0, ctype)
        ctype = np.where((COT == 0) & ((KG_climate == 3) | (KG_climate == 4)), 0, ctype)
        ctype = np.where((COT == 0) & (KG_climate == 1), 0, ctype)
        ctype = np.where((COT == 0) & (KG_climate == 2), 0, ctype)
        ctype = np.where((COT > 0) & ((KG_climate == 5) | (KG_climate == 6)), 1, ctype)
        ctype = np.where((COT > 0) & ((KG_climate == 3) | (KG_climate == 4)), 1, ctype)
        ctype = np.where((COT > 0) & (KG_climate == 2), 1, ctype)
        ctype = np.where((COT > 0) & (KG_climate == 1), 3, ctype)

    return ctype

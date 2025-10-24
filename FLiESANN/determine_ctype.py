import numpy as np
from typing import Union

def determine_ctype(
    KG_climate: Union[int, np.ndarray], 
    COT: Union[float, np.ndarray], 
    dynamic: bool = True
) -> Union[int, np.ndarray]:
    # Convert inputs to numpy arrays
    KG_climate_arr = np.asarray(KG_climate, dtype=int)
    COT_arr = np.asarray(COT, dtype=float)
    
    # Determine if inputs were scalars
    KG_is_scalar = np.isscalar(KG_climate)
    COT_is_scalar = np.isscalar(COT)
    
    # Broadcast arrays to same shape if needed
    KG_climate_arr, COT_arr = np.broadcast_arrays(KG_climate_arr, COT_arr)
    
    # Initialize ctype array
    ctype = np.full(KG_climate_arr.shape, 0, dtype=np.uint16)

    if dynamic:
        ctype = np.where((COT_arr == 0) & ((KG_climate_arr == 5) | (KG_climate_arr == 6)), 0, ctype)
        ctype = np.where((COT_arr == 0) & ((KG_climate_arr == 3) | (KG_climate_arr == 4)), 0, ctype)
        ctype = np.where((COT_arr == 0) & (KG_climate_arr == 1), 0, ctype)
        ctype = np.where((COT_arr == 0) & (KG_climate_arr == 2), 0, ctype)
        ctype = np.where((COT_arr > 0) & ((KG_climate_arr == 5) | (KG_climate_arr == 6)), 1, ctype)
        ctype = np.where((COT_arr > 0) & ((KG_climate_arr == 3) | (KG_climate_arr == 4)), 1, ctype)
        ctype = np.where((COT_arr > 0) & (KG_climate_arr == 2), 1, ctype)
        ctype = np.where((COT_arr > 0) & (KG_climate_arr == 1), 3, ctype)

    # Return scalar if both inputs were scalars, otherwise return array
    if KG_is_scalar and COT_is_scalar:
        return int(ctype.item())
    else:
        return ctype

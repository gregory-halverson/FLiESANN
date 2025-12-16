import numpy as np


def ensure_array(value, shape=None):
    """
    Ensure the input is an array, converting scalar values if necessary.
    
    This function handles various input types and converts them to float32 numpy arrays.
    It properly handles None values by converting them to NaN, and can optionally
    broadcast scalar values to a specified shape.
    
    Args:
        value: Input value to convert to array. Can be:
            - int or float: Converted to array, optionally broadcast to shape
            - None: Returns None
            - np.ndarray: Converted to float32, with None values replaced by NaN
            - Other types (e.g., lists): Converted to array then to float32
        shape (tuple, optional): Shape to broadcast scalar values to. 
            If None and value is scalar, returns scalar array.
    
    Returns:
        np.ndarray or None: Float32 numpy array with None values replaced by NaN,
            or None if input is None.
    
    Examples:
        >>> ensure_array(5.0)
        array(5., dtype=float32)
        
        >>> ensure_array(10, shape=(2, 3))
        array([[10., 10., 10.],
               [10., 10., 10.]], dtype=float32)
        
        >>> ensure_array([1, 2, None, 4])
        array([ 1.,  2., nan,  4.], dtype=float32)
        
        >>> ensure_array(None)
        None
    """
    if isinstance(value, (int, float)):
        return np.full(shape, value, dtype=np.float32) if shape else np.array(value, dtype=np.float32)
    elif value is None:
        return None
    elif isinstance(value, np.ndarray):
        # Convert object arrays with None values to float arrays with NaN
        if value.dtype == object:
            # Replace None with NaN and convert to float32
            value_copy = value.copy()
            value_copy[value_copy == None] = np.nan
            return value_copy.astype(np.float32)
        else:
            return value.astype(np.float32)
    else:
        # For other types (like lists), convert to array and then ensure float32
        arr = np.array(value)
        if arr.dtype == object:
            arr[arr == None] = np.nan
            return arr.astype(np.float32)
        else:
            return arr.astype(np.float32)

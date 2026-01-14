import numpy as np
import pandas as pd


def ensure_array(value, shape=None):
    """
    Ensure the input is an array, converting scalar values if necessary.
    
    This function handles various input types and converts them to float32 numpy arrays.
    It properly handles None values by converting them to NaN, and can optionally
    broadcast scalar values or arrays to a specified shape.
    
    Args:
        value: Input value to convert to array. Can be:
            - int or float: Converted to array, optionally broadcast to shape
            - None: Returns None
            - pd.DataFrame: Converted to numpy array using .values
            - np.ndarray: Converted to float32, with None values replaced by NaN,
                         and optionally broadcast to match shape
            - Other types (e.g., lists): Converted to array then to float32
        shape (tuple, optional): Shape to broadcast values to. 
            If None and value is scalar, returns scalar array.
            If provided and value is array, broadcasts array to shape.
    
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
        
        >>> ensure_array(np.array([1, 2]), shape=(2, 3))
        array([[1., 1., 1.],
               [2., 2., 2.]], dtype=float32)
    """
    if isinstance(value, (int, float)):
        return np.full(shape, value, dtype=np.float32) if shape else np.array(value, dtype=np.float32)
    elif value is None:
        return None
    elif isinstance(value, pd.DataFrame):
        # Convert DataFrame to numpy array
        # If DataFrame has multiple columns, assume the first column contains the actual data values
        # (other columns might be metadata like lat, lon, lat_used, lon_used)
        if len(value.columns) > 1:
            # Extract just the first column (the data column)
            value_array = value.iloc[:, 0].values.astype(np.float32)
        else:
            value_array = value.values.astype(np.float32).squeeze()
        
        # If shape is provided and doesn't match, try to broadcast
        if shape is not None and value_array.shape != shape:
            try:
                value_array = np.broadcast_to(value_array, shape)
            except ValueError:
                # If direct broadcast fails, try expanding dimensions
                if len(value_array.shape) == 1 and len(shape) == 2:
                    if value_array.shape[0] == shape[0]:
                        value_array = np.broadcast_to(value_array[:, np.newaxis], shape)
                    elif value_array.shape[0] == shape[1]:
                        value_array = np.broadcast_to(value_array[np.newaxis, :], shape)
                    else:
                        raise ValueError(f"Cannot broadcast DataFrame array of shape {value_array.shape} to shape {shape}")
                else:
                    raise
        return value_array
    elif isinstance(value, np.ndarray):
        # Convert object arrays with None values to float arrays with NaN
        if value.dtype == object:
            # Replace None with NaN and convert to float32
            value_copy = value.copy()
            value_copy[value_copy == None] = np.nan
            value_array = value_copy.astype(np.float32)
        else:
            value_array = value.astype(np.float32)
        
        # Broadcast array to target shape if needed
        if shape is not None and value_array.shape != shape:
            # Try to broadcast the array to the target shape
            try:
                value_array = np.broadcast_to(value_array, shape)
            except ValueError:
                # If direct broadcast fails, try expanding dimensions
                # Handle 1D to 1D case: repeat elements to match target shape
                if len(value_array.shape) == 1 and len(shape) == 1:
                    # For 1D arrays, use np.repeat to match lengths
                    # Example: (2,) -> (4,) by repeating each element
                    if shape[0] % value_array.shape[0] == 0:
                        repeat_factor = shape[0] // value_array.shape[0]
                        value_array = np.repeat(value_array, repeat_factor)
                    else:
                        raise ValueError(f"Cannot broadcast array of shape {value_array.shape} to shape {shape}: shapes not compatible")
                # Handle 1D to 2D case
                elif len(value_array.shape) == 1 and len(shape) == 2:
                    # Try broadcasting along each dimension
                    if value_array.shape[0] == shape[0]:
                        # Expand to (n, 1) then broadcast to (n, m)
                        value_array = np.broadcast_to(value_array[:, np.newaxis], shape)
                    elif value_array.shape[0] == shape[1]:
                        # Expand to (1, m) then broadcast to (n, m)
                        value_array = np.broadcast_to(value_array[np.newaxis, :], shape)
                    else:
                        # Try repeating to match one dimension
                        if shape[0] % value_array.shape[0] == 0:
                            repeat_factor = shape[0] // value_array.shape[0]
                            repeated = np.repeat(value_array, repeat_factor)
                            value_array = np.broadcast_to(repeated[:, np.newaxis], shape)
                        elif shape[1] % value_array.shape[0] == 0:
                            repeat_factor = shape[1] // value_array.shape[0]
                            repeated = np.repeat(value_array, repeat_factor)
                            value_array = np.broadcast_to(repeated[np.newaxis, :], shape)
                        else:
                            raise ValueError(f"Cannot broadcast array of shape {value_array.shape} to shape {shape}")
                else:
                    raise ValueError(f"Cannot broadcast array of shape {value_array.shape} to shape {shape}")
        
        return value_array
    else:
        # For other types (like lists), convert to array and then ensure float32
        arr = np.array(value)
        if arr.dtype == object:
            arr[arr == None] = np.nan
            return arr.astype(np.float32)
        else:
            return arr.astype(np.float32)

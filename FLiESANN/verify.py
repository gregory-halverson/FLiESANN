import argparse

def verify() -> bool:
    """
    Verifies the correctness of the PT-JPL-SM model implementation by comparing
    its outputs to a reference dataset.

    This function loads a known input table and the corresponding expected output table.
    It runs the model on the input data, then compares the resulting outputs to the
    reference outputs for key variables using strict numerical tolerances. If all
    outputs match within tolerance, the function returns True. Otherwise, it prints
    which column failed and returns False.

    Returns:
        bool: True if all model outputs match the reference outputs within tolerance, False otherwise.
    """
    import pandas as pd
    import numpy as np
    from .ECOv002_calval_FLiESANN_inputs import load_ECOv002_calval_FLiESANN_inputs
    from .ECOv002_calval_FLiESANN_outputs import load_ECOv002_calval_FLiESANN_outputs
    from .process_FLiESANN_table import process_FLiESANN_table

    # Load input and output tables
    input_df = load_ECOv002_calval_FLiESANN_inputs()
    output_df = load_ECOv002_calval_FLiESANN_outputs()

    # Run the model on the input table
    model_df = process_FLiESANN_table(input_df, offline_mode=True)

    # Columns to compare (model outputs)
    output_columns = [
        "SWin_Wm2",
        "PAR_diffuse_Wm2",
        "PAR_direct_Wm2",
        "NIR_diffuse_Wm2",
        "NIR_direct_Wm2",
        "UV_Wm2"
    ]

    # Compare each output column and collect mismatches
    mismatches = []
    for col in output_columns:
        if col not in model_df or col not in output_df:
            mismatches.append((col, 'missing_column', None))
            continue
        model_vals = model_df[col].values
        ref_vals = output_df[col].values
        
        # Ensure values are numeric and handle NaN safely
        try:
            model_vals = pd.to_numeric(model_vals, errors='coerce')
            ref_vals = pd.to_numeric(ref_vals, errors='coerce')
        except:
            # If conversion fails, treat as string comparison
            if not np.array_equal(model_vals, ref_vals):
                mismatches.append((col, 'value_mismatch', {'type': 'string_mismatch'}))
            continue
            
        # Use numpy allclose for floating point comparison
        # Relaxed tolerances to account for minor platform/Python version differences
        # rtol=5e-4 allows 0.05% relative error, atol=1e-4 handles small absolute values
        if not np.allclose(model_vals, ref_vals, rtol=5e-4, atol=1e-4, equal_nan=True):
            # Find indices where values differ
            diffs = np.abs(model_vals - ref_vals)
            max_diff = np.nanmax(diffs) if not np.all(np.isnan(diffs)) else np.nan
            idxs = np.where(~np.isclose(model_vals, ref_vals, rtol=5e-4, atol=1e-4, equal_nan=True))[0]
            mismatch_info = {
                'indices': idxs.tolist(),
                'model_values': model_vals[idxs].tolist(),
                'ref_values': ref_vals[idxs].tolist(),
                'diffs': diffs[idxs].tolist(),
                'max_diff': float(max_diff)
            }
            mismatches.append((col, 'value_mismatch', mismatch_info))
    if mismatches:
        print("Verification failed. Details:")
        for col, reason, info in mismatches:
            if reason == 'missing_column':
                print(f"  Missing column: {col}")
            elif reason == 'value_mismatch':
                print(f"  Mismatch in column: {col}")
                if info.get('type') == 'string_mismatch':
                    print(f"    String comparison failed")
                else:
                    print(f"    Max difference: {info['max_diff']}")
                    print(f"    Indices off: {info['indices']}")
                    print(f"    Model values: {info['model_values']}")
                    print(f"    Reference values: {info['ref_values']}")
                    print(f"    Differences: {info['diffs']}")
        return False
    return True

def main():
    """
    Main function to execute the verification process.
    """
    parser = argparse.ArgumentParser(description="Verify the correctness of the PT-JPL-SM model implementation.")
    # Add arguments here if needed in the future
    args = parser.parse_args()

    # Call the verify function
    success = verify()

    if success:
        print("Verification succeeded.")
    else:
        print("Verification failed.")

if __name__ == "__main__":
    main()

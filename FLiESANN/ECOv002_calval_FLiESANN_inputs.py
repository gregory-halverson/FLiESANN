import os
import pandas as pd

def load_ECOv002_calval_FLiESANN_inputs() -> pd.DataFrame:
    """
    Load the input data for the FLiES-ANN model from the ECOSTRESS Collection 2 Cal-Val dataset.

    Returns:
        pd.DataFrame: A DataFrame containing the reference input data.
    """

    # Define the path to the output CSV file relative to this module's directory
    module_dir = os.path.dirname(os.path.abspath(__file__))
    load_file_path = os.path.join(module_dir, "ECOv002-cal-val-FLiESANN-inputs.csv")

    # Load the output data into a DataFrame
    inputs_df = pd.read_csv(load_file_path)

    return inputs_df
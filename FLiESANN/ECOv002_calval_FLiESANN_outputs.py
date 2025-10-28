import os
import pandas as pd

def load_ECOv002_calval_FLiESANN_outputs() -> pd.DataFrame:
    """
    Load the output data for the FLiESANN model from the ECOSTRESS Collection 2 Cal-Val dataset.

    Returns:
        pd.DataFrame: A DataFrame containing the reference output data.
    """

    # Define the path to the output CSV file relative to this module's directory
    module_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(module_dir, "ECOv002-cal-val-FLiESANN-outputs.csv")

    # Load the output data into a DataFrame
    outputs_df = pd.read_csv(output_file_path)

    return outputs_df
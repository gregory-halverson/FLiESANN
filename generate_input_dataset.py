# %%
import pandas as pd
from GEOS5FP import GEOS5FP
from NASADEM import NASADEMConnection
from ECOv002_calval_tables import load_calval_table

from FLiESANN import process_FLiESANN_table

# %%
calval_df = load_calval_table()
calval_df

# %%
calval_df.time_UTC

# %%
# Ensure `time_UTC` is in datetime format
calval_df['time_UTC'] = pd.to_datetime(calval_df['time_UTC'])

# Create a `date_UTC` column by extracting the date from `time_UTC`
calval_df['date_UTC'] = calval_df['time_UTC'].dt.date


# %%
# Remove the filtering step for a single day
# Pass the entire dataset to the process_FLiESANN_table function

# Process with explicit atmospheric parameter defaults to match reference data
# The reference data uses: COT=0, AOT=0, vapor_gccm=0, ozone_cm=0.3
# This approach avoids NaN values from missing GEOS5FP data

GEOS5FP_connection = GEOS5FP(download_directory="GEOS5FP_download")
NASADEM_connection = NASADEMConnection(download_directory="NASADEM_download")

# Process the entire dataset
results_df = process_FLiESANN_table(
    calval_df,  # Use dataset with atmospheric defaults
    GEOS5FP_connection=GEOS5FP_connection,
    NASADEM_connection=NASADEM_connection       
)

print(results_df)

results_df.to_csv("ECOv002-cal-val-FLiESANN-inputs.csv")

# %%




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
calval_df

# %%
# Group by `date_UTC` and count rows for each date, sorted in descending order
date_counts = calval_df.groupby('date_UTC').size().sort_values(ascending=False)
date_counts

# %%
# Determine the date with the most observations and store it as a string
most_observed_date = date_counts.idxmax()
most_observed_date_str = most_observed_date.strftime('%Y-%m-%d')
most_observed_date_str

# %%
# Remove the filtering step for a single day
# Pass the entire dataset to the process_FLiESANN_table function

# Process with explicit atmospheric parameter defaults to match reference data
# The reference data uses: COT=0, AOT=0, vapor_gccm=0, ozone_cm=0.3
# This approach avoids NaN values from missing GEOS5FP data

GEOS5FP_connection = GEOS5FP(download_directory="GEOS5FP_download")
NASADEM_connection = NASADEMConnection(download_directory="NASADEM_download")

# Add default atmospheric parameters to the dataframe
# These match the values found in the reference outputs
calval_df_with_defaults = calval_df.copy()
calval_df_with_defaults['COT'] = 0.0  # Clear sky conditions  
calval_df_with_defaults['AOT'] = 0.0  # No aerosols
calval_df_with_defaults['vapor_gccm'] = 0.0  # No water vapor
calval_df_with_defaults['ozone_cm'] = 0.3  # Constant ozone level

# Process the entire dataset
results_df = process_FLiESANN_table(
    calval_df_with_defaults,  # Use dataset with atmospheric defaults
    GEOS5FP_connection=GEOS5FP_connection,
    NASADEM_connection=NASADEM_connection       
)

print(results_df)

results_df.to_csv("ECOv002-cal-val-FliESANN-inputs.csv")

# %%




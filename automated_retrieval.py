# %%
import pandas as pd

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
single_day_df = calval_df[calval_df['date_UTC'] == pd.to_datetime(most_observed_date_str).date()]
single_day_df

# %%
results_df = process_FLiESANN_table(single_day_df)
print(results_df)

# %%




import pandas as pd

# File path to the input CSV file
input_file = "FLiESANN/ECOv002-cal-val-FLiESANN-inputs.csv"

def fix_sza_column(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Check if the 'SZA_deg' column exists
    if 'SZA_deg' in df.columns:
        # Remove square brackets and extract the value inside
        df['SZA_deg'] = df['SZA_deg'].str.extract(r'\[(.*?)\]')[0]

        # Write the updated DataFrame back to the file
        df.to_csv(file_path, index=False)
        print("Successfully updated the 'SZA_deg' column.")
    else:
        print("Column 'SZA_deg' not found in the file.")

if __name__ == "__main__":
    fix_sza_column(input_file)
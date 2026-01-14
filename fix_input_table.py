import pandas as pd

# File path to the input CSV file
input_file = "FLiESANN/ECOv002-cal-val-FLiESANN-inputs.csv"

def fix_table(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Remove square brackets from all cells in the DataFrame
    df = df.replace(to_replace=r'\[(.*?)\]', value=lambda m: m.group(1), regex=True)

    # Write the updated DataFrame back to the file
    df.to_csv(file_path, index=False)
    print("Successfully removed square brackets from all cells in the table.")

if __name__ == "__main__":
    fix_table(input_file)
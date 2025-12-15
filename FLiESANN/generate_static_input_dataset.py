import os
import pandas as pd
import geopandas as gpd
import rasters as rt
from ECOv002_calval_tables import load_combined_eco_flux_ec_filtered, load_metadata_ebc_filt
from koppengeiger import load_koppen_geiger


def generate_static_input_dataset():
    """
    Generate static input dataset for FLiESANN from tower locations.
    
    Creates a CSV file with static tower data including Koppen-Geiger climate
    classification and atmospheric parameters.
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the static tower data
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    generated_input_table_filename = os.path.join(script_dir, "ECOv002-static-tower-FLiESANN-inputs.csv")
    
    # Load tower location metadata
    tower_locations_df = load_metadata_ebc_filt()
    
    tower_IDs = list(tower_locations_df["Site ID"])
    tower_names = list(tower_locations_df.Name)
    
    # Load tower data (for reference, though not directly used in static dataset)
    tower_data_df = load_combined_eco_flux_ec_filtered()
    
    # Create MultiPoint geometry for tower locations
    tower_points = rt.MultiPoint(
        x=tower_locations_df['Long'].values,
        y=tower_locations_df['Lat'].values
    )
    
    # Load Koppen-Geiger climate classification for tower locations
    KG_climate = load_koppen_geiger(geometry=tower_points)
    
    # Create GeoDataFrame with static tower data
    tower_static_data_gdf = gpd.GeoDataFrame({
        "ID": tower_IDs,
        "name": tower_names,
        "KG_climate": KG_climate,
        "geometry": tower_points
    })
    
    # Save to CSV
    tower_static_data_gdf.to_csv(generated_input_table_filename, index=False)
    
    print(f"Static input dataset saved to: {generated_input_table_filename}")
    
    return tower_static_data_gdf


if __name__ == "__main__":
    generate_static_input_dataset()




#!/usr/bin/env python3
"""
Test script for retrieve_FLiESANN_static_inputs using calval table data.

This script tests the retrieve_FLiESANN_static_inputs function with data from
the ECOv002 calibration/validation table. It demonstrates automatic retrieval
of elevation and Köppen-Geiger climate classification from geometry points.
"""
import numpy as np
import pandas as pd
import rasters as rt
from ECOv002_calval_tables import load_calval_table
from FLiESANN.retrieve_FLiESANN_static_inputs import retrieve_FLiESANN_static_inputs

def main():
    # Load calibration/validation table
    print("Loading calibration/validation table...")
    calval_df = load_calval_table()
    print(f"Loaded {len(calval_df)} records from calval table")
    print(f"Columns: {list(calval_df.columns)}\n")
    
    # Take a small subset for testing (first 5 records)
    test_df = calval_df.head(5).copy()
    print(f"Testing with first {len(test_df)} records")
    print(f"Sites: {test_df['ID'].unique()}\n")
    
    # Extract geometries from the test dataframe
    if 'geometry' not in test_df.columns:
        print("ERROR: 'geometry' column not found in calval table")
        return
    
    geometries = test_df['geometry'].values
    x_coords = [g.x for g in geometries]
    y_coords = [g.y for g in geometries]
    geometry = rt.MultiPoint(x=x_coords, y=y_coords)
    
    # Test 1: Retrieve both elevation and Köppen-Geiger climate from geometry
    print("=" * 60)
    print("Test 1: Retrieve elevation and Köppen-Geiger from geometry")
    print("=" * 60)
    
    print(f"Using {len(geometries)} geometry points from table")
    print("Retrieving static inputs (both elevation and Köppen-Geiger climate)...")
    
    static_inputs = retrieve_FLiESANN_static_inputs(geometry=geometry)
    
    print("\nRetrieved static inputs:")
    print(f"  elevation_m shape: {static_inputs['elevation_m'].shape}")
    print(f"  elevation_km shape: {static_inputs['elevation_km'].shape}")
    print(f"  KG_climate shape: {static_inputs['KG_climate'].shape}")
    
    print(f"\nElevation (m): {static_inputs['elevation_m']}")
    print(f"Elevation (km): {static_inputs['elevation_km']}")
    print(f"Köppen-Geiger climate: {static_inputs['KG_climate']}")
    
    # Compare with table values if available
    if 'elevation_m' in test_df.columns:
        table_elevation = test_df['elevation_m'].values
        print(f"\nTable elevation (m): {table_elevation}")
        diff = static_inputs['elevation_m'] - table_elevation
        print(f"Difference from table: {diff}")
        print(f"Max absolute difference: {np.max(np.abs(diff)):.2f} m")
    
    # Test 2: Retrieve only Köppen-Geiger (provide elevation)
    print("\n" + "=" * 60)
    print("Test 2: Provide elevation, retrieve only Köppen-Geiger")
    print("=" * 60)
    
    # Use retrieved elevation from Test 1
    elevation_m_test2 = static_inputs['elevation_m']
    
    print(f"Providing retrieved elevation: {elevation_m_test2}")
    static_inputs_2 = retrieve_FLiESANN_static_inputs(
        elevation_m=elevation_m_test2,
        geometry=geometry
    )
    
    print("\nRetrieved static inputs:")
    print(f"  elevation_m: {static_inputs_2['elevation_m']}")
    print(f"  elevation_km: {static_inputs_2['elevation_km']}")
    print(f"  KG_climate: {static_inputs_2['KG_climate']}")
    
    # Verify elevation was passed through unchanged
    assert np.allclose(static_inputs_2['elevation_m'], elevation_m_test2), "Elevation should be passed through unchanged"
    print("\n✓ Elevation correctly passed through")
    
    # Verify Köppen-Geiger climate matches Test 1
    if isinstance(static_inputs_2['KG_climate'], np.ndarray):
        assert np.array_equal(static_inputs_2['KG_climate'], static_inputs['KG_climate']), "Köppen-Geiger should match Test 1"
    else:
        assert static_inputs_2['KG_climate'] == static_inputs['KG_climate'], "Köppen-Geiger should match Test 1"
    print("✓ Köppen-Geiger climate matches Test 1")
    
    # Test 3: Single point test - retrieve both elevation and Köppen-Geiger
    print("\n" + "=" * 60)
    print("Test 3: Single point - retrieve both elevation and Köppen-Geiger")
    print("=" * 60)
    
    first_record = test_df.iloc[0]
    single_point = first_record['geometry']
    print(f"Testing site: {first_record['ID']}")
    print(f"Location: ({single_point.y:.4f}, {single_point.x:.4f})")
    
    single_geometry = rt.Point(single_point.x, single_point.y)
    
    print("\nRetrieving static inputs for single point...")
    single_static = retrieve_FLiESANN_static_inputs(geometry=single_geometry)
    
    print("\nRetrieved static inputs:")
    print(f"  elevation_m: {single_static['elevation_m']}")
    print(f"  elevation_km: {single_static['elevation_km']}")
    print(f"  KG_climate: {single_static['KG_climate']}")
    
    # Compare with multipoint result for first point
    if isinstance(static_inputs['elevation_m'], np.ndarray):
        first_elevation_mp = static_inputs['elevation_m'][0]
        first_kg_mp = static_inputs['KG_climate'][0]
    else:
        first_elevation_mp = static_inputs['elevation_m']
        first_kg_mp = static_inputs['KG_climate']
    
    print(f"\nMultipoint result for first point:")
    print(f"  elevation_m: {first_elevation_mp}")
    print(f"  KG_climate: {first_kg_mp}")
    
    # Verify they match
    assert np.isclose(single_static['elevation_m'], first_elevation_mp), "Single point elevation should match multipoint"
    print("✓ Single point elevation matches multipoint result")
    
    assert single_static['KG_climate'] == first_kg_mp, "Single point Köppen-Geiger should match multipoint"
    print("✓ Single point Köppen-Geiger matches multipoint result")
            
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test FLiES processing with spectral albedo outputs on cal/val dataset
"""
import pandas as pd
import numpy as np
from FLiESANN.ECOv002_calval_FLiESANN_inputs import load_ECOv002_calval_FLiESANN_inputs
from FLiESANN import process_FLiESANN_table

print("Loading ECOv002 cal/val input dataset...")
inputs_df = load_ECOv002_calval_FLiESANN_inputs()
print(f"Loaded {len(inputs_df)} rows")

# Take a small sample for testing
test_df = inputs_df.head(10).copy()
print(f"\nTesting with {len(test_df)} rows")

# Ensure time_UTC is in datetime format
test_df['time_UTC'] = pd.to_datetime(test_df['time_UTC'])

# Check NDVI availability
print(f"\nNDVI range in test data: {test_df['NDVI'].min():.3f} to {test_df['NDVI'].max():.3f}")
print(f"Mean NDVI: {test_df['NDVI'].mean():.3f}")

print("\n" + "="*80)
print("TEST 1: Processing WITHOUT NDVI (uniform spectral albedo)")
print("="*80)

try:
    results_no_ndvi = process_FLiESANN_table(
        test_df.drop(columns=['NDVI']),  # Exclude NDVI
        GEOS5FP_connection=None,
        NASADEM_connection=None
    )
    
    print(f"Successfully processed! Results shape: {results_no_ndvi.shape}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TEST 2: Processing WITH NDVI (vegetation-based spectral partitioning)")
print("="*80)

print("\nProcessing with FLiESANN...")
try:
    results_df = process_FLiESANN_table(
        test_df,
        GEOS5FP_connection=None,  # Use provided inputs
        NASADEM_connection=None   # Use provided inputs
    )
    
    print(f"\nSuccessfully processed! Results shape: {results_df.shape}")
    
    # Check new spectral albedo outputs
    new_outputs = [
        'SWout_Wm2',
        'PAR_reflected_Wm2', 
        'NIR_reflected_Wm2',
        'PAR_albedo',
        'NIR_albedo'
    ]
    
    print("\n" + "="*80)
    print("SPECTRAL ALBEDO OUTPUTS (WITH NDVI-BASED PARTITIONING):")
    print("="*80)
    
    for col in new_outputs:
        if col in results_df.columns:
            values = results_df[col].values
            print(f"\n{col}:")
            print(f"  Min:    {np.nanmin(values):.6f}")
            print(f"  Max:    {np.nanmax(values):.6f}")
            print(f"  Mean:   {np.nanmean(values):.6f}")
            print(f"  Median: {np.nanmedian(values):.6f}")
            print(f"  NaN count: {np.sum(np.isnan(values))}")
        else:
            print(f"\n{col}: NOT FOUND IN OUTPUT")
    
    # Check existing outputs for comparison
    existing_outputs = [
        'SWin_Wm2',
        'PAR_Wm2',
        'NIR_Wm2',
        'albedo'
    ]
    
    print("\n" + "="*80)
    print("EXISTING OUTPUTS (for comparison):")
    print("="*80)
    
    for col in existing_outputs:
        if col in results_df.columns:
            values = results_df[col].values
            print(f"\n{col}:")
            print(f"  Min:    {np.nanmin(values):.6f}")
            print(f"  Max:    {np.nanmax(values):.6f}")
            print(f"  Mean:   {np.nanmean(values):.6f}")
    
    # Detailed albedo statistics
    print("\n" + "="*80)
    print("DETAILED ALBEDO DISTRIBUTION STATISTICS:")
    print("="*80)
    
    albedo_vars = ['albedo', 'PAR_albedo', 'NIR_albedo']
    
    for var in albedo_vars:
        if var in results_df.columns:
            values = results_df[var].values
            print(f"\n{var}:")
            print(f"  Count:       {len(values)}")
            print(f"  Min:         {np.nanmin(values):.6f}")
            print(f"  25th %ile:   {np.nanpercentile(values, 25):.6f}")
            print(f"  Median:      {np.nanmedian(values):.6f}")
            print(f"  Mean:        {np.nanmean(values):.6f}")
            print(f"  75th %ile:   {np.nanpercentile(values, 75):.6f}")
            print(f"  95th %ile:   {np.nanpercentile(values, 95):.6f}")
            print(f"  Max:         {np.nanmax(values):.6f}")
            print(f"  Std Dev:     {np.nanstd(values):.6f}")
            print(f"  Range:       {np.nanmax(values) - np.nanmin(values):.6f}")
            print(f"  NaN count:   {np.sum(np.isnan(values))}")
    
    # Comparison between spectral albedos
    if all(var in results_df.columns for var in ['albedo', 'PAR_albedo', 'NIR_albedo']):
        print("\n" + "="*80)
        print("ALBEDO COMPARISONS:")
        print("="*80)
        
        albedo = results_df['albedo'].values
        par_albedo = results_df['PAR_albedo'].values
        nir_albedo = results_df['NIR_albedo'].values
        
        # Difference from broadband
        par_diff = par_albedo - albedo
        nir_diff = nir_albedo - albedo
        
        print(f"\nPAR_albedo - albedo:")
        print(f"  Min diff:    {np.nanmin(par_diff):.6e}")
        print(f"  Max diff:    {np.nanmax(par_diff):.6e}")
        print(f"  Mean diff:   {np.nanmean(par_diff):.6e}")
        print(f"  Abs mean:    {np.nanmean(np.abs(par_diff)):.6e}")
        
        print(f"\nNIR_albedo - albedo:")
        print(f"  Min diff:    {np.nanmin(nir_diff):.6e}")
        print(f"  Max diff:    {np.nanmax(nir_diff):.6e}")
        print(f"  Mean diff:   {np.nanmean(nir_diff):.6e}")
        print(f"  Abs mean:    {np.nanmean(np.abs(nir_diff)):.6e}")
        
        # Ratio comparisons
        par_ratio = par_albedo / albedo
        nir_ratio = nir_albedo / albedo
        
        print(f"\nPAR_albedo / albedo ratio:")
        print(f"  Min:         {np.nanmin(par_ratio):.6f}")
        print(f"  Mean:        {np.nanmean(par_ratio):.6f}")
        print(f"  Max:         {np.nanmax(par_ratio):.6f}")
        
        print(f"\nNIR_albedo / albedo ratio:")
        print(f"  Min:         {np.nanmin(nir_ratio):.6f}")
        print(f"  Mean:        {np.nanmean(nir_ratio):.6f}")
        print(f"  Max:         {np.nanmax(nir_ratio):.6f}")
        
        # Compare with and without NDVI
        if 'results_no_ndvi' in locals():
            print("\n" + "="*80)
            print("COMPARISON: WITH vs WITHOUT NDVI")
            print("="*80)
            
            par_albedo_no_ndvi = results_no_ndvi['PAR_albedo'].values
            nir_albedo_no_ndvi = results_no_ndvi['NIR_albedo'].values
            
            print(f"\nPAR_albedo difference (with NDVI - without NDVI):")
            par_diff = par_albedo - par_albedo_no_ndvi
            print(f"  Min:         {np.nanmin(par_diff):.6f}")
            print(f"  Mean:        {np.nanmean(par_diff):.6f}")
            print(f"  Max:         {np.nanmax(par_diff):.6f}")
            
            print(f"\nNIR_albedo difference (with NDVI - without NDVI):")
            nir_diff = nir_albedo - nir_albedo_no_ndvi
            print(f"  Min:         {np.nanmin(nir_diff):.6f}")
            print(f"  Mean:        {np.nanmean(nir_diff):.6f}")
            print(f"  Max:         {np.nanmax(nir_diff):.6f}")
            
            print(f"\nNIR/PAR albedo ratio (with NDVI):")
            ratio_with = nir_albedo / np.where(par_albedo > 0, par_albedo, 1)
            print(f"  Min:         {np.nanmin(ratio_with):.6f}")
            print(f"  Mean:        {np.nanmean(ratio_with):.6f}")
            print(f"  Max:         {np.nanmax(ratio_with):.6f}")
            
            print(f"\nNIR/PAR albedo ratio (without NDVI):")
            ratio_without = nir_albedo_no_ndvi / np.where(par_albedo_no_ndvi > 0, par_albedo_no_ndvi, 1)
            print(f"  Min:         {np.nanmin(ratio_without):.6f}")
            print(f"  Mean:        {np.nanmean(ratio_without):.6f}")
            print(f"  Max:         {np.nanmax(ratio_without):.6f}")
    
    # Verify physical constraints
    print("\n" + "="*80)
    print("PHYSICAL CONSTRAINT CHECKS:")
    print("="*80)
    
    # Check that albedos are between 0 and 1
    if 'PAR_albedo' in results_df.columns:
        par_albedo = results_df['PAR_albedo'].values
        par_valid = np.logical_and(par_albedo >= 0, par_albedo <= 1)
        print(f"\nPAR_albedo in [0,1]: {np.sum(par_valid)}/{len(par_albedo)} ({100*np.mean(par_valid):.1f}%)")
        if not np.all(par_valid):
            print(f"  Invalid values: {par_albedo[~par_valid]}")
    
    if 'NIR_albedo' in results_df.columns:
        nir_albedo = results_df['NIR_albedo'].values
        nir_valid = np.logical_and(nir_albedo >= 0, nir_albedo <= 1)
        print(f"NIR_albedo in [0,1]: {np.sum(nir_valid)}/{len(nir_albedo)} ({100*np.mean(nir_valid):.1f}%)")
        if not np.all(nir_valid):
            print(f"  Invalid values: {nir_albedo[~nir_valid]}")
    
    # Check that reflected < incoming
    if all(col in results_df.columns for col in ['PAR_reflected_Wm2', 'PAR_Wm2']):
        par_valid = results_df['PAR_reflected_Wm2'] <= results_df['PAR_Wm2']
        print(f"\nPAR_reflected <= PAR_incoming: {np.sum(par_valid)}/{len(par_valid)} ({100*np.mean(par_valid):.1f}%)")
    
    if all(col in results_df.columns for col in ['NIR_reflected_Wm2', 'NIR_Wm2']):
        nir_valid = results_df['NIR_reflected_Wm2'] <= results_df['NIR_Wm2']
        print(f"NIR_reflected <= NIR_incoming: {np.sum(nir_valid)}/{len(nir_valid)} ({100*np.mean(nir_valid):.1f}%)")
    
    # Check that SWout = SWin * albedo
    if all(col in results_df.columns for col in ['SWout_Wm2', 'SWin_Wm2', 'albedo']):
        expected_swout = results_df['SWin_Wm2'] * results_df['albedo']
        actual_swout = results_df['SWout_Wm2']
        diff = np.abs(expected_swout - actual_swout)
        max_diff = np.nanmax(diff)
        print(f"\nSWout_Wm2 = SWin_Wm2 * albedo: max difference = {max_diff:.6e}")
    
    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)
    
except Exception as e:
    print(f"\nError during processing: {e}")
    import traceback
    traceback.print_exc()

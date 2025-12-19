"""
Test script to verify the broadcasting fix for KG_climate and COT arrays.
"""
import numpy as np
from FLiESANN.determine_atype import determine_atype

# Simulate the error condition: KG_climate with shape (2,) and COT with shape (4, 5)
print("Testing broadcasting fix...")

# Test 1: Scalar KG_climate with array COT
print("\nTest 1: Scalar KG_climate with array COT")
KG_climate = 5
COT = np.random.rand(4, 5)
try:
    result = determine_atype(KG_climate, COT)
    print(f"✓ Success! Result shape: {result.shape}")
except ValueError as e:
    print(f"✗ Failed: {e}")

# Test 2: 1D KG_climate with 2D COT (original error case)
print("\nTest 2: 1D KG_climate broadcast to 2D COT")
KG_climate = np.array([5, 6])
COT = np.zeros((2, 5))
try:
    result = determine_atype(KG_climate, COT)
    print(f"✓ Success! Result shape: {result.shape}")
except ValueError as e:
    print(f"✗ Failed: {e}")

# Test 3: Matching shapes
print("\nTest 3: Matching shapes")
KG_climate = np.array([[5, 5, 5], [6, 6, 6]])
COT = np.array([[0, 1, 2], [0, 1, 2]])
try:
    result = determine_atype(KG_climate, COT)
    print(f"✓ Success! Result shape: {result.shape}")
except ValueError as e:
    print(f"✗ Failed: {e}")

print("\nAll tests completed!")

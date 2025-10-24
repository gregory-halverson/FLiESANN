import pytest
from FLiESANN import verify

def test_verify():
    assert verify(), "Model verification failed: outputs do not match expected results."

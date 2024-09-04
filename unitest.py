import pytest
import pandas as pd


#@pytest.fixture

def test_load_dataframe(df):
    # Mock the pandas read_csv method to return sample_data
    assert df.shape == (150, 5) 
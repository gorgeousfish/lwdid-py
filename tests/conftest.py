"""
Pytest configuration file providing shared fixtures and helper functions.
"""
import os
import pandas as pd
import pytest


def find_smoking_data():
    """Helper function to locate the smoking.csv data file in common paths."""
    candidates = [
        os.path.join(os.path.dirname(__file__), 'data', 'smoking.csv'),
        'tests/data/smoking.csv',
        'data/smoking.csv',
    ]
    
    for path in candidates:
        if os.path.exists(path):
            return path
    
    # If none of the candidate paths exist, return None (tests will be skipped)
    return None


@pytest.fixture
def smoking_data_path():
    """Fixture that returns the path to the smoking.csv data file."""
    path = find_smoking_data()
    if path is None:
        pytest.skip("smoking.csv not found in expected locations")
    return path


@pytest.fixture
def smoking_data(smoking_data_path):
    """Fixture that loads the smoking.csv dataset as a pandas DataFrame."""
    return pd.read_csv(smoking_data_path)

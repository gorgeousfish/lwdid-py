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


def find_castle_data():
    """Helper function to locate the castle.csv data file."""
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', 'data', 'castle.csv'),
        os.path.join(os.path.dirname(__file__), 'data', 'castle.csv'),
        'data/castle.csv',
    ]
    
    for path in candidates:
        resolved = os.path.abspath(path)
        if os.path.exists(resolved):
            return resolved
    
    return None


@pytest.fixture
def castle_data():
    """
    Load and preprocess Castle Law data
    
    Data location: lwdid-py_v0.1.0/data/castle.csv
    
    Path calculation:
    - conftest.py is at lwdid-py_v0.1.0/tests/conftest.py
    - package_root = lwdid-py_v0.1.0/
    - data_path = lwdid-py_v0.1.0/data/castle.csv
    
    Returns a copy to avoid side effects between tests.
    """
    # tests/ -> lwdid-py_v0.1.0/ -> data/
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.dirname(tests_dir)
    data_path = os.path.join(package_root, 'data', 'castle.csv')
    
    if not os.path.exists(data_path):
        pytest.skip(f"Castle data file not found: {data_path}")
    
    data = pd.read_csv(data_path)
    
    # Preprocess gvar: NaN -> 0 (never treated), otherwise keep original value
    data['gvar'] = data['effyear'].fillna(0).astype(int)
    
    # Return copy to avoid side effects between tests
    return data.copy()

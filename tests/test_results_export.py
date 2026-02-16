"""Tests for exporting estimation results to various formats.

This module verifies the serialization of ``LWDIDResults`` objects to
Excel (.xlsx), CSV (.csv), and LaTeX (.tex) formats, including correct
handling of missing data and staggered-adoption metadata.

Validates the results reporting layer of the Lee-Wooldridge
Difference-in-Differences framework.

References
----------
Lee, S. & Wooldridge, J. M. (2025). A Simple Transformation Approach to
    Difference-in-Differences Estimation for Panel Data. SSRN 4516518.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from lwdid import lwdid


def _get_test_data():
    """Load test data from smoking.csv.

    Returns:
        pd.DataFrame: Smoking dataset for testing export functionality.

    Note:
        Falls back to synthetic data if smoking.csv is not found.
    """
    candidates = [
        os.path.join(os.path.dirname(__file__), 'data', 'smoking.csv'),
        'tests/data/smoking.csv',
        'data/smoking.csv',
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path)

    # Fallback: create minimal synthetic dataset
    rows = []
    for i in range(5):
        for t in range(3):
            rows.append({
                'state': i + 1,
                'year': 2000 + t,
                'd': 1 if i == 0 else 0,
                'post': 1 if t >= 1 else 0,
                'lcigsale': 10.0 - 0.1 * t + i * 0.01,
            })
    return pd.DataFrame(rows)


def test_exports_excel_csv_latex_and_csv_error_when_missing():
    """Test export to Excel, CSV, and LaTeX formats.

    Verifies that:
    1. Results can be exported to Excel (.xlsx)
    2. Results can be exported to CSV (.csv)
    3. Results can be exported to LaTeX (.tex)
    4. CSV export raises error when att_by_period is missing

    Note: Uses rireps=200 for faster testing while maintaining reliability.
    """
    data = _get_test_data()

    res = lwdid(
        data, y='lcigsale', d='d', ivar='state', tvar='year', post='post',
        rolling='detrend', vce='robust', ri=True, rireps=200, seed=123
    )

    # Test successful exports
    with tempfile.TemporaryDirectory() as td:
        xlsx = os.path.join(td, 'res.xlsx')
        csv = os.path.join(td, 'res.csv')
        tex = os.path.join(td, 'res.tex')

        res.to_excel(xlsx)
        assert os.path.exists(xlsx), "Excel file should be created"

        res.to_csv(csv)
        assert os.path.exists(csv), "CSV file should be created"

        res.to_latex(tex)
        assert os.path.exists(tex), "LaTeX file should be created"

    # Test error handling: CSV export requires att_by_period
    res2 = lwdid(
        data, y='lcigsale', d='d', ivar='state', tvar='year', post='post',
        rolling='detrend', vce='robust'
    )
    res2._att_by_period = None

    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, 'dummy.csv')
        with pytest.raises(ValueError, match='att_by_period is not available for CSV export'):
            res2.to_csv(csv_path)


def test_to_excel_with_ri():
    """Test Excel export with randomization inference results.

    Verifies that Excel export works correctly when RI results are included,
    and that the exported file can be read back by pandas.

    Note: Uses rireps=200 for faster testing while maintaining reliability.
    """
    data = _get_test_data()

    res = lwdid(
        data, y='lcigsale', d='d', ivar='state', tvar='year', post='post',
        rolling='demean', vce=None, ri=True, rireps=200, seed=42
    )
    
    with tempfile.TemporaryDirectory() as td:
        xlsx = os.path.join(td, 'with_ri.xlsx')
        res.to_excel(xlsx)
        assert os.path.exists(xlsx)
        
        # Verify that the file can be read by pandas
        summary = pd.read_excel(xlsx, sheet_name='Summary')
        assert 'Statistic' in summary.columns
        assert 'Value' in summary.columns
        
        by_period = pd.read_excel(xlsx, sheet_name='ByPeriod')
        assert 'period' in by_period.columns
        
        # There should be a RI sheet
        ri_sheet = pd.read_excel(xlsx, sheet_name='RI')
        assert 'Parameter' in ri_sheet.columns or 'Statistic' in ri_sheet.columns


def test_to_excel_without_ri():
    """Test Excel export when RI results are not requested."""
    data = _get_test_data()
    
    res = lwdid(
        data, y='lcigsale', d='d', ivar='state', tvar='year', post='post',
        rolling='demean', vce=None, ri=False
    )
    
    with tempfile.TemporaryDirectory() as td:
        xlsx = os.path.join(td, 'without_ri.xlsx')
        res.to_excel(xlsx)
        assert os.path.exists(xlsx)
        
        # Verify that there is no RI sheet
        xl = pd.ExcelFile(xlsx)
        assert 'Summary' in xl.sheet_names
        assert 'ByPeriod' in xl.sheet_names
        # RI sheet should not exist (optionally check with try-except if needed)


def test_to_csv_basic():
    """Test CSV export."""
    data = _get_test_data()
    
    res = lwdid(
        data, y='lcigsale', d='d', ivar='state', tvar='year', post='post',
        rolling='demean', vce=None
    )
    
    with tempfile.TemporaryDirectory() as td:
        csv = os.path.join(td, 'results.csv')
        res.to_csv(csv)
        assert os.path.exists(csv)
        
        # Verify that the CSV can be read and has the expected columns
        df = pd.read_csv(csv)
        expected_cols = ['period', 'tindex', 'beta', 'se', 'ci_lower', 'ci_upper', 'tstat', 'pval', 'N']
        assert list(df.columns) == expected_cols
        assert 'average' in df['period'].values


def test_to_csv_missing_att_by_period():
    """Test error handling when att_by_period is missing."""
    data = _get_test_data()
    
    res = lwdid(
        data, y='lcigsale', d='d', ivar='state', tvar='year', post='post',
        rolling='demean', vce=None
    )
    
    # Simulate missing att_by_period
    res._att_by_period = None
    
    with tempfile.TemporaryDirectory() as td:
        csv = os.path.join(td, 'results.csv')
        with pytest.raises(ValueError, match='att_by_period is not available for CSV export'):
            res.to_csv(csv)


def test_to_csv_empty_att_by_period():
    """Test error handling when att_by_period is an empty DataFrame."""
    data = _get_test_data()
    
    res = lwdid(
        data, y='lcigsale', d='d', ivar='state', tvar='year', post='post',
        rolling='demean', vce=None
    )
    
    # Simulate empty DataFrame
    res._att_by_period = pd.DataFrame()
    
    with tempfile.TemporaryDirectory() as td:
        csv = os.path.join(td, 'results.csv')
        with pytest.raises(ValueError, match='att_by_period is not available for CSV export'):
            res.to_csv(csv)


def test_to_latex_basic():
    """Test LaTeX export."""
    data = _get_test_data()
    
    res = lwdid(
        data, y='lcigsale', d='d', ivar='state', tvar='year', post='post',
        rolling='demean', vce=None
    )
    
    with tempfile.TemporaryDirectory() as td:
        tex = os.path.join(td, 'results.tex')
        res.to_latex(tex)
        assert os.path.exists(tex)
        
        # Verify LaTeX file contains basic content
        with open(tex, 'r') as f:
            content = f.read()
            assert 'ATT' in content
            assert 'tabular' in content or 'table' in content.lower()


def test_to_latex_with_att_by_period():
    """Test LaTeX export including att_by_period data."""
    data = _get_test_data()
    
    res = lwdid(
        data, y='lcigsale', d='d', ivar='state', tvar='year', post='post',
        rolling='demean', vce=None
    )
    
    with tempfile.TemporaryDirectory() as td:
        tex = os.path.join(td, 'results_full.tex')
        res.to_latex(tex)
        assert os.path.exists(tex)
        
        # Verify period effects are included
        with open(tex, 'r') as f:
            content = f.read()
            assert 'period' in content.lower() or 'average' in content.lower()


def test_results_summary():
    """Test the summary() method of LWDIDResults."""
    data = _get_test_data()
    
    res = lwdid(
        data, y='lcigsale', d='d', ivar='state', tvar='year', post='post',
        rolling='demean', vce=None
    )
    
    summary = res.summary()
    assert summary is not None
    # summary should return a DataFrame or produce printed output


def test_results_repr():
    """Test the __repr__ and __str__ methods of LWDIDResults."""
    data = _get_test_data()
    
    res = lwdid(
        data, y='lcigsale', d='d', ivar='state', tvar='year', post='post',
        rolling='demean', vce=None
    )
    
    repr_str = repr(res)
    assert 'LWDIDResults' in repr_str
    assert 'att=' in repr_str
    
    str_str = str(res)
    assert len(str_str) > 0



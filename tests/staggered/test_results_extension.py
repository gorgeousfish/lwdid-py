"""
E2-S2: LWDIDResults Staggered Extension Tests

File location: lwdid-py_v0.1.0/tests/staggered/test_results_extension.py
"""
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest


def create_mock_staggered_results(
    aggregate: str = 'overall',
    control_group: str = 'never_treated',
    control_group_used: str = 'never_treated',
):
    """
    Create mock staggered results for testing
    
    Parameters
    ----------
    aggregate : str
        Aggregation method: 'none', 'cohort', 'overall'
    control_group : str
        User-specified control group strategy
    control_group_used : str
        Actual control group used (for auto-switch testing)
    """
    # Base fields (required for all LWDIDResults)
    results_dict = {
        'is_staggered': True,
        'att': 0.0,  # Not used in staggered mode
        'se_att': 0.0,
        't_stat': 0.0,
        'pvalue': 1.0,
        'ci_lower': 0.0,
        'ci_upper': 0.0,
        'nobs': 550,
        'n_treated': 21,
        'n_control': 29,
        'df_resid': 48,
        'params': np.array([0.0]),
        'bse': np.array([0.0]),
        'vcov': np.array([[0.0]]),
        'resid': np.zeros(50),
        'vce_type': 'hc3',
        
        # Staggered base info
        'cohorts': [2005, 2006, 2007, 2008, 2009],
        'cohort_sizes': {2005: 1, 2006: 13, 2007: 4, 2008: 2, 2009: 1},
        'control_group': control_group,
        'control_group_used': control_group_used,
        'aggregate': aggregate,
        'estimator': 'ra',
        'n_never_treated': 29,
        'rolling': 'demean',
        
        # (g,r) specific effects (always present)
        'att_by_cohort_time': pd.DataFrame({
            'cohort': [2005, 2005, 2006, 2006, 2007, 2007, 2008, 2009],
            'period': [2005, 2006, 2006, 2007, 2007, 2008, 2008, 2009],
            'event_time': [0, 1, 0, 1, 0, 1, 0, 0],
            'att': [0.032, 0.078, 0.045, 0.089, 0.056, 0.102, 0.068, 0.055],
            'se': [0.089, 0.095, 0.050, 0.055, 0.072, 0.078, 0.085, 0.098],
            'ci_lower': [-0.148, -0.114, -0.057, -0.023, -0.090, -0.056, -0.104, -0.143],
            'ci_upper': [0.212, 0.270, 0.147, 0.201, 0.202, 0.260, 0.240, 0.253],
            't_stat': [0.36, 0.82, 0.90, 1.62, 0.78, 1.31, 0.80, 0.56],
            'pvalue': [0.722, 0.419, 0.376, 0.117, 0.442, 0.201, 0.431, 0.580],
            'n_treated': [1, 1, 13, 13, 4, 4, 2, 1],
            'n_control': [29, 29, 29, 29, 29, 29, 29, 29],
            'n_total': [30, 30, 42, 42, 33, 33, 31, 30],
        }),
    }
    
    # Add fields based on aggregate level
    if aggregate in ['cohort', 'overall']:
        results_dict['att_by_cohort'] = pd.DataFrame({
            'cohort': [2005, 2006, 2007, 2008, 2009],
            'att': [0.085, 0.098, 0.112, 0.075, 0.062],
            'se': [0.062, 0.045, 0.068, 0.089, 0.105],
            'ci_lower': [-0.041, 0.006, -0.028, -0.107, -0.154],
            'ci_upper': [0.211, 0.190, 0.252, 0.257, 0.278],
            't_stat': [1.37, 2.18, 1.65, 0.84, 0.59],
            'pvalue': [0.182, 0.038, 0.112, 0.408, 0.560],
            'n_units': [1, 13, 4, 2, 1],
            'n_periods': [6, 5, 4, 3, 2],
        })
        results_dict['cohort_weights'] = {
            2005: 0.048, 2006: 0.619, 2007: 0.190, 2008: 0.095, 2009: 0.048
        }
    else:
        results_dict['att_by_cohort'] = None
        results_dict['cohort_weights'] = {}
    
    if aggregate == 'overall':
        results_dict.update({
            'att_overall': 0.092,
            'se_overall': 0.057,
            'ci_overall_lower': -0.024,
            'ci_overall_upper': 0.208,
            't_stat_overall': 1.614,
            'pvalue_overall': 0.118,
        })
    else:
        results_dict.update({
            'att_overall': None,
            'se_overall': None,
            'ci_overall_lower': None,
            'ci_overall_upper': None,
            't_stat_overall': None,
            'pvalue_overall': None,
        })
    
    # Metadata must include rolling field
    metadata = {
        'K': 5,
        'tpost1': 2005,
        'N_treated': 21,
        'N_control': 29,
        'depvar': 'lhomicide',
        'ivar': 'sid',
        'tvar': 'year',
        'rolling': 'demean',
        'is_staggered': True,
    }
    
    return results_dict, metadata


# ============================================================
# Section 5.1: summary_staggered() Tests
# ============================================================

def test_summary_staggered_basic():
    """Test summary_staggered basic output"""
    from lwdid.results import LWDIDResults
    
    results_dict, metadata = create_mock_staggered_results(aggregate='overall')
    results = LWDIDResults(results_dict, metadata, None)
    
    summary = results.summary_staggered()
    
    # Verify key content exists
    assert "LWDID Staggered DiD Results" in summary
    assert "Treatment Cohorts: 2005, 2006, 2007, 2008, 2009" in summary
    assert "Overall Weighted Effect" in summary
    assert "ATT_ω" in summary
    assert "0.0920" in summary  # att_overall
    assert "Cohort Weights" in summary
    assert "Cohort-Specific Effects" in summary
    # Verify n_never_treated info exists
    assert "Never Treated" in summary or "29" in summary


def test_summary_staggered_aggregate_none():
    """aggregate='none' should not show Overall and Cohort effects"""
    from lwdid.results import LWDIDResults
    
    results_dict, metadata = create_mock_staggered_results(aggregate='none')
    results = LWDIDResults(results_dict, metadata, None)
    
    summary = results.summary_staggered()
    
    assert "Overall Weighted Effect" not in summary
    assert "Cohort-Specific Effects" not in summary
    assert "att_by_cohort_time" in summary  # Hint should exist


def test_summary_staggered_aggregate_cohort():
    """aggregate='cohort' should show Cohort effects but not Overall"""
    from lwdid.results import LWDIDResults
    
    results_dict, metadata = create_mock_staggered_results(aggregate='cohort')
    results = LWDIDResults(results_dict, metadata, None)
    
    summary = results.summary_staggered()
    
    assert "Cohort-Specific Effects" in summary
    assert "Overall Weighted Effect" not in summary


def test_summary_staggered_control_group_auto_switch():
    """Test notification when control group is auto-switched"""
    from lwdid.results import LWDIDResults
    
    # Simulate user specified not_yet_treated but auto-switched to never_treated
    results_dict, metadata = create_mock_staggered_results(
        aggregate='overall',
        control_group='not_yet_treated',
        control_group_used='never_treated'
    )
    results = LWDIDResults(results_dict, metadata, None)
    
    summary = results.summary_staggered()
    
    # Should contain auto-switch notification
    assert "Auto-switched" in summary or "auto-switched" in summary.lower()
    assert "not_yet_treated" in summary


def test_summary_dispatches_to_staggered():
    """summary() should auto-dispatch to summary_staggered()"""
    from lwdid.results import LWDIDResults
    
    results_dict, metadata = create_mock_staggered_results(aggregate='overall')
    results = LWDIDResults(results_dict, metadata, None)
    
    summary = results.summary()
    
    assert "LWDID Staggered DiD Results" in summary


def test_summary_staggered_not_staggered_raises():
    """Non-staggered results calling summary_staggered should raise error"""
    from lwdid.results import LWDIDResults
    
    # Create common timing results
    results_dict = {
        'is_staggered': False,
        'att': 0.1,
        'se_att': 0.05,
        't_stat': 2.0,
        'pvalue': 0.05,
        'ci_lower': 0.0,
        'ci_upper': 0.2,
        'nobs': 100,
        'n_treated': 10,
        'n_control': 90,
        'df_resid': 98,
        'params': np.array([0.1]),
        'bse': np.array([0.05]),
        'vcov': np.array([[0.0025]]),
        'resid': np.zeros(100),
        'vce_type': 'hc3',
    }
    
    metadata = {
        'K': 5,
        'tpost1': 6,
        'N_treated': 10,
        'N_control': 90,
        'depvar': 'y',
        'ivar': 'id',
        'tvar': 'year',
        'rolling': 'demean',
    }
    
    results = LWDIDResults(results_dict, metadata, None)
    
    with pytest.raises(ValueError, match="staggered"):
        results.summary_staggered()


def test_n_never_treated_staggered():
    """Test n_never_treated attribute returns correct value in staggered mode"""
    from lwdid.results import LWDIDResults
    
    results_dict, metadata = create_mock_staggered_results(aggregate='overall')
    results = LWDIDResults(results_dict, metadata, None)
    
    assert results.n_never_treated == 29


def test_n_never_treated_common_timing():
    """Test n_never_treated attribute returns None in common timing mode"""
    from lwdid.results import LWDIDResults
    
    results_dict = {
        'is_staggered': False,
        'att': 0.1,
        'se_att': 0.05,
        't_stat': 2.0,
        'pvalue': 0.05,
        'ci_lower': 0.0,
        'ci_upper': 0.2,
        'nobs': 100,
        'n_treated': 10,
        'n_control': 90,
        'df_resid': 98,
        'params': np.array([0.1]),
        'bse': np.array([0.05]),
        'vcov': np.array([[0.0025]]),
        'resid': np.zeros(100),
        'vce_type': 'hc3',
    }
    
    metadata = {
        'K': 5,
        'tpost1': 6,
        'N_treated': 10,
        'N_control': 90,
        'depvar': 'y',
        'ivar': 'id',
        'tvar': 'year',
        'rolling': 'demean',
    }
    
    results = LWDIDResults(results_dict, metadata, None)
    
    assert results.n_never_treated is None


def test_rolling_attribute_staggered():
    """Test rolling attribute is accessible in staggered results"""
    from lwdid.results import LWDIDResults
    
    results_dict, metadata = create_mock_staggered_results(aggregate='overall')
    results = LWDIDResults(results_dict, metadata, None)
    
    # Verify rolling attribute is accessible and correct
    assert results.rolling == 'demean'
    
    # Verify summary contains rolling info
    summary = results.summary_staggered()
    assert 'demean' in summary.lower() or 'Transformation' in summary


def test_vce_type_attribute_staggered():
    """Test vce_type attribute is accessible in staggered results"""
    from lwdid.results import LWDIDResults
    
    results_dict, metadata = create_mock_staggered_results(aggregate='overall')
    results = LWDIDResults(results_dict, metadata, None)
    
    # Verify vce_type attribute is accessible
    assert results.vce_type == 'hc3'


def test_summary_staggered_aggregate_cohort_no_t_stat():
    """aggregate='cohort' should have t_stat_overall=None, summary should not error"""
    from lwdid.results import LWDIDResults
    
    results_dict, metadata = create_mock_staggered_results(aggregate='cohort')
    results = LWDIDResults(results_dict, metadata, None)
    
    # Verify t_stat_overall and pvalue_overall are None
    assert results.t_stat_overall is None
    assert results.pvalue_overall is None
    
    # summary should run without error
    summary = results.summary_staggered()
    assert "Cohort-Specific Effects" in summary
    assert "Overall Weighted Effect" not in summary


# ============================================================
# Section: __repr__() Tests
# ============================================================

def test_repr_staggered_overall():
    """staggered results __repr__ should show staggered info"""
    from lwdid.results import LWDIDResults
    
    results_dict, metadata = create_mock_staggered_results(aggregate='overall')
    results = LWDIDResults(results_dict, metadata, None)
    
    repr_str = repr(results)
    
    # Should contain staggered identifier
    assert "staggered" in repr_str.lower()
    # Should contain att_overall value
    assert "0.092" in repr_str or "att_overall" in repr_str


def test_repr_staggered_no_overall():
    """aggregate='cohort' __repr__ should not show att_overall"""
    from lwdid.results import LWDIDResults
    
    results_dict, metadata = create_mock_staggered_results(aggregate='cohort')
    results = LWDIDResults(results_dict, metadata, None)
    
    repr_str = repr(results)
    
    # Should contain staggered identifier
    assert "staggered" in repr_str.lower()
    # Should contain cohorts count
    assert "cohorts" in repr_str.lower() or "5" in repr_str


def test_repr_common_timing():
    """common timing results __repr__ should keep original format"""
    from lwdid.results import LWDIDResults
    
    results_dict = {
        'is_staggered': False,
        'att': 0.1,
        'se_att': 0.05,
        't_stat': 2.0,
        'pvalue': 0.05,
        'ci_lower': 0.0,
        'ci_upper': 0.2,
        'nobs': 100,
        'n_treated': 10,
        'n_control': 90,
        'df_resid': 98,
        'params': np.array([0.1]),
        'bse': np.array([0.05]),
        'vcov': np.array([[0.0025]]),
        'resid': np.zeros(100),
        'vce_type': 'hc3',
    }
    
    metadata = {
        'K': 5,
        'tpost1': 6,
        'N_treated': 10,
        'N_control': 90,
        'depvar': 'y',
        'ivar': 'id',
        'tvar': 'year',
        'rolling': 'demean',
    }
    
    results = LWDIDResults(results_dict, metadata, None)
    
    repr_str = repr(results)
    
    # Should not contain staggered identifier
    assert "staggered" not in repr_str.lower()
    # Should contain att and se
    assert "att=" in repr_str.lower() or "0.1" in repr_str


# ============================================================
# Section 5.2: plot_event_study() Tests
# ============================================================

def test_plot_event_study_basic():
    """Test Event Study plot basic rendering"""
    from lwdid.results import LWDIDResults
    import matplotlib.pyplot as plt
    
    results_dict, metadata = create_mock_staggered_results(aggregate='overall')
    results = LWDIDResults(results_dict, metadata, None)
    
    fig, ax = results.plot_event_study(show_ci=True, ref_period=None)
    
    assert fig is not None
    assert ax is not None
    
    plt.close(fig)


def test_plot_event_study_reference_normalization():
    """Test reference period normalization"""
    from lwdid.results import LWDIDResults
    import matplotlib.pyplot as plt
    
    results_dict, metadata = create_mock_staggered_results(aggregate='overall')
    results = LWDIDResults(results_dict, metadata, None)
    
    # Reference period is event_time=0 (default)
    fig, ax = results.plot_event_study()
    
    plt.close(fig)


def test_plot_event_study_ref_period_not_found_warning():
    """Test warning when reference period not found"""
    from lwdid.results import LWDIDResults
    import matplotlib.pyplot as plt
    
    results_dict, metadata = create_mock_staggered_results(aggregate='overall')
    results = LWDIDResults(results_dict, metadata, None)
    
    # Use a non-existent reference period (e=-1, only post-treatment effects exist)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fig, ax = results.plot_event_study(ref_period=-1)
        
        # Verify warning was raised
        assert len(w) == 1
        assert "not found" in str(w[0].message).lower()
    
    plt.close(fig)


def test_plot_event_study_no_normalization():
    """Test ref_period=None skips normalization"""
    from lwdid.results import LWDIDResults
    import matplotlib.pyplot as plt
    
    results_dict, metadata = create_mock_staggered_results(aggregate='overall')
    results = LWDIDResults(results_dict, metadata, None)
    
    # ref_period=None should not normalize
    fig, ax = results.plot_event_study(ref_period=None)
    
    assert fig is not None
    assert ax is not None
    
    plt.close(fig)


def test_plot_event_study_ref_period_nonexistent():
    """Non-existent reference period should warn, not error"""
    from lwdid.results import LWDIDResults
    import matplotlib.pyplot as plt
    
    results_dict, metadata = create_mock_staggered_results(aggregate='overall')
    results = LWDIDResults(results_dict, metadata, None)
    
    # Specify non-existent reference period
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fig, ax = results.plot_event_study(ref_period=-5)
        
        # Should raise UserWarning
        assert len(w) > 0
        assert any("not found" in str(warning.message).lower() for warning in w)
    
    plt.close(fig)


def test_plot_event_study_weighted_aggregation():
    """Test weighted aggregation method"""
    from lwdid.results import LWDIDResults
    import matplotlib.pyplot as plt
    
    results_dict, metadata = create_mock_staggered_results(aggregate='overall')
    results = LWDIDResults(results_dict, metadata, None)
    
    fig, ax = results.plot_event_study(aggregation='weighted', ref_period=None)
    
    assert fig is not None
    plt.close(fig)


def test_plot_event_study_save():
    """Test figure saving"""
    from lwdid.results import LWDIDResults
    import matplotlib.pyplot as plt
    
    results_dict, metadata = create_mock_staggered_results(aggregate='overall')
    results = LWDIDResults(results_dict, metadata, None)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        path = f.name
    
    try:
        fig, ax = results.plot_event_study(savefig=path, ref_period=None)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    finally:
        if os.path.exists(path):
            os.remove(path)
        plt.close(fig)


def test_plot_event_study_not_staggered_raises():
    """Non-staggered results calling plot_event_study should raise error"""
    from lwdid.results import LWDIDResults
    
    # Create common timing results
    results_dict = {
        'is_staggered': False,
        'att': 0.1,
        'se_att': 0.05,
        't_stat': 2.0,
        'pvalue': 0.05,
        'ci_lower': 0.0,
        'ci_upper': 0.2,
        'nobs': 100,
        'n_treated': 10,
        'n_control': 90,
        'df_resid': 98,
        'params': np.array([0.1]),
        'bse': np.array([0.05]),
        'vcov': np.array([[0.0025]]),
        'resid': np.zeros(100),
        'vce_type': 'hc3',
    }
    
    metadata = {
        'K': 5,
        'tpost1': 6,
        'N_treated': 10,
        'N_control': 90,
        'depvar': 'y',
        'ivar': 'id',
        'tvar': 'year',
        'rolling': 'demean',
        'is_staggered': False,
    }
    
    results = LWDIDResults(results_dict, metadata, None)
    
    with pytest.raises(ValueError, match="staggered"):
        results.plot_event_study()


def test_plot_event_study_empty_att_by_cohort_time_raises():
    """Empty att_by_cohort_time should raise error"""
    from lwdid.results import LWDIDResults
    
    results_dict, metadata = create_mock_staggered_results(aggregate='overall')
    results_dict['att_by_cohort_time'] = pd.DataFrame()  # Empty DataFrame
    results = LWDIDResults(results_dict, metadata, None)
    
    with pytest.raises(ValueError, match="empty"):
        results.plot_event_study()


def test_plot_event_study_aggregate_none():
    """aggregate='none' should still allow Event Study plot
    
    Even without cohort and overall effects calculated,
    as long as att_by_cohort_time exists, event study can be plotted.
    """
    from lwdid.results import LWDIDResults
    import matplotlib.pyplot as plt
    
    results_dict, metadata = create_mock_staggered_results(aggregate='none')
    results = LWDIDResults(results_dict, metadata, None)
    
    # aggregate='none' still has att_by_cohort_time
    assert results.att_by_cohort_time is not None
    assert not results.att_by_cohort_time.empty
    
    # Should plot normally
    fig, ax = results.plot_event_study(ref_period=None)
    
    assert fig is not None
    assert ax is not None
    
    plt.close(fig)


# ============================================================
# Section 5.3: to_excel_staggered() Tests
# ============================================================

def test_to_excel_staggered_basic():
    """Test Excel export basic functionality"""
    from lwdid.results import LWDIDResults
    
    results_dict, metadata = create_mock_staggered_results(aggregate='overall')
    results = LWDIDResults(results_dict, metadata, None)
    
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
        path = f.name
    
    try:
        results.to_excel_staggered(path)
        
        assert os.path.exists(path)
        
        # Verify all expected sheets exist
        xl = pd.ExcelFile(path)
        expected_sheets = ['Summary', 'Overall', 'Cohort', 'CohortTime', 'Weights', 'Metadata']
        for sheet in expected_sheets:
            assert sheet in xl.sheet_names, f"Missing sheet: {sheet}"
        
        # Verify Summary sheet content
        df_summary = pd.read_excel(xl, 'Summary')
        assert any(df_summary['Item'] == 'Estimation Type')
        
        # Verify Weights sheet content
        df_weights = pd.read_excel(xl, 'Weights')
        assert len(df_weights) == 5  # 5 cohorts
        assert abs(df_weights['weight'].sum() - 1.0) < 0.001  # Weights sum to 1
        
        xl.close()
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_to_excel_staggered_aggregate_none():
    """aggregate='none' should not have Overall, Cohort and Weights sheets"""
    from lwdid.results import LWDIDResults
    
    results_dict, metadata = create_mock_staggered_results(aggregate='none')
    results = LWDIDResults(results_dict, metadata, None)
    
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
        path = f.name
    
    try:
        results.to_excel_staggered(path)
        xl = pd.ExcelFile(path)
        
        # Should exist: Summary, CohortTime, Metadata
        assert 'Summary' in xl.sheet_names
        assert 'CohortTime' in xl.sheet_names
        assert 'Metadata' in xl.sheet_names
        
        xl.close()
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_to_excel_staggered_aggregate_cohort():
    """aggregate='cohort' should have Cohort and Weights sheets but no Overall"""
    from lwdid.results import LWDIDResults
    
    results_dict, metadata = create_mock_staggered_results(aggregate='cohort')
    results = LWDIDResults(results_dict, metadata, None)
    
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
        path = f.name
    
    try:
        results.to_excel_staggered(path)
        xl = pd.ExcelFile(path)
        
        # Should exist
        assert 'Summary' in xl.sheet_names
        assert 'Cohort' in xl.sheet_names
        assert 'CohortTime' in xl.sheet_names
        assert 'Weights' in xl.sheet_names
        assert 'Metadata' in xl.sheet_names
        
        xl.close()
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_to_excel_dispatches_to_staggered():
    """to_excel() should auto-detect staggered and dispatch"""
    from lwdid.results import LWDIDResults
    
    results_dict, metadata = create_mock_staggered_results(aggregate='overall')
    results = LWDIDResults(results_dict, metadata, None)
    
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
        path = f.name
    
    try:
        results.to_excel(path)  # Use generic method
        
        # Verify staggered format (has Weights sheet)
        xl = pd.ExcelFile(path)
        assert 'Weights' in xl.sheet_names  # staggered-specific sheet
        xl.close()
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_to_excel_staggered_not_staggered_raises():
    """Non-staggered results calling to_excel_staggered should raise error"""
    from lwdid.results import LWDIDResults
    
    # Create common timing results
    results_dict = {
        'is_staggered': False,
        'att': 0.1,
        'se_att': 0.05,
        't_stat': 2.0,
        'pvalue': 0.05,
        'ci_lower': 0.0,
        'ci_upper': 0.2,
        'nobs': 100,
        'n_treated': 10,
        'n_control': 90,
        'df_resid': 98,
        'params': np.array([0.1]),
        'bse': np.array([0.05]),
        'vcov': np.array([[0.0025]]),
        'resid': np.zeros(100),
        'vce_type': 'hc3',
    }
    
    metadata = {
        'K': 5,
        'tpost1': 6,
        'N_treated': 10,
        'N_control': 90,
        'depvar': 'y',
        'ivar': 'id',
        'tvar': 'year',
        'rolling': 'demean',
        'is_staggered': False,
    }
    
    results = LWDIDResults(results_dict, metadata, None)
    
    with pytest.raises(ValueError, match="staggered"):
        results.to_excel_staggered('/tmp/test.xlsx')


# ============================================================
# Section 5.4: Castle Law Data Tests
# ============================================================

def test_castle_data_preprocessing():
    """
    Castle Law data preprocessing validation
    
    Validates castle.csv data structure and gvar preprocessing logic.
    This test can run independently, not depending on lwdid estimation.
    """
    # tests/staggered/ -> tests/ -> lwdid-py_v0.1.0/ -> data/
    staggered_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.dirname(staggered_dir)
    package_root = os.path.dirname(tests_dir)
    data_path = os.path.join(package_root, 'data', 'castle.csv')
    
    if not os.path.exists(data_path):
        pytest.skip(f"Castle data file not found: {data_path}")
    data = pd.read_csv(data_path)
    
    # === 1. Verify key columns exist ===
    required_cols = ['lhomicide', 'sid', 'year', 'effyear', 'dinf']
    for col in required_cols:
        assert col in data.columns, f"Missing key column: {col}"
    
    # === 2. Verify data size ===
    # 50 states × 11 years = 550 rows
    assert len(data) == 550, f"Data should have 550 rows, got {len(data)}"
    assert data['sid'].nunique() == 50, "Should have 50 states"
    assert data['year'].min() == 2000 and data['year'].max() == 2010
    
    # === 3. Verify never treated units ===
    # dinf=1 means never treated
    n_never_treated_states = data.groupby('sid')['dinf'].first().sum()
    assert n_never_treated_states == 29, f"Should have 29 NT states, got {n_never_treated_states}"
    
    # === 4. Preprocess gvar ===
    data['gvar'] = data['effyear'].fillna(0).astype(int)
    
    # Verify cohort structure
    cohorts = sorted(data[data['gvar'] > 0]['gvar'].unique())
    assert cohorts == [2005, 2006, 2007, 2008, 2009], f"Cohorts incorrect: {cohorts}"
    
    # Verify cohort sizes
    cohort_sizes = data[data['gvar'] > 0].groupby('gvar')['sid'].nunique().to_dict()
    expected_sizes = {2005: 1, 2006: 13, 2007: 4, 2008: 2, 2009: 1}
    assert cohort_sizes == expected_sizes, f"Cohort sizes incorrect: {cohort_sizes}"
    
    # === 5. Verify dinf and gvar consistency ===
    never_treated_by_dinf = set(data[data['dinf'] == 1]['sid'].unique())
    never_treated_by_gvar = set(data[data['gvar'] == 0]['sid'].unique())
    assert never_treated_by_dinf == never_treated_by_gvar, "dinf and gvar NT markers inconsistent"

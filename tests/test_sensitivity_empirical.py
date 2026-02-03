"""
Empirical data tests for sensitivity analysis module.
"""

import pytest
import numpy as np
import pandas as pd
import os

from lwdid.sensitivity import (
    robustness_pre_periods,
    sensitivity_no_anticipation,
    PrePeriodRobustnessResult,
    NoAnticipationSensitivityResult,
)


class TestMissingDataHandling:
    """Test handling of missing data."""
    
    @pytest.fixture
    def data_with_missing(self):
        """Create data with realistic missing patterns."""
        np.random.seed(42)
        n_units, n_periods = 100, 10
        treatment_period = 6
        data = []
        n_treated = n_units // 2
        for i in range(n_units):
            alpha_i = np.random.normal(0, 1)
            treated = i < n_treated
            for t in range(1, n_periods + 1):
                if np.random.random() < 0.02 * t:
                    continue
                d_it = 1 if treated and t >= treatment_period else 0
                tau = 2.0 if d_it else 0
                y = alpha_i + tau + np.random.normal(0, 0.5)
                data.append({
                    'unit': i,
                    'time': t,
                    'Y': y,
                    'first_treat': treatment_period if treated else 0,
                })
        return pd.DataFrame(data)
    
    def test_handles_attrition(self, data_with_missing):
        """Test that analysis handles attrition patterns."""
        result = robustness_pre_periods(
            data_with_missing,
            y='Y',
            ivar='unit',
            tvar='time',
            gvar='first_treat',
            rolling='demean',
            pre_period_range=(2, 4),
            verbose=False
        )
        assert result.n_specifications > 0
        assert not np.isnan(result.baseline_spec.att)


class TestRobustnessInterpretation:
    """Test interpretation of robustness results."""
    
    def test_robust_data_interpretation(self):
        """Test interpretation when data is truly robust."""
        np.random.seed(42)
        n_units, n_periods = 200, 12
        treatment_period = 7
        data = []
        n_treated = n_units // 2
        for i in range(n_units):
            alpha_i = np.random.normal(0, 1)
            treated = i < n_treated
            for t in range(1, n_periods + 1):
                gamma_t = 0.1 * t
                d_it = 1 if treated and t >= treatment_period else 0
                tau = 2.0 if d_it else 0
                y = alpha_i + gamma_t + tau + np.random.normal(0, 0.5)
                data.append({
                    'unit': i,
                    'time': t,
                    'Y': y,
                    'first_treat': treatment_period if treated else 0,
                })
        data = pd.DataFrame(data)
        result = robustness_pre_periods(
            data,
            y='Y',
            ivar='unit',
            tvar='time',
            gvar='first_treat',
            rolling='demean',
            pre_period_range=(2, 5),
            verbose=False
        )
        assert result.sensitivity_ratio < 0.50
        assert result.all_same_sign is True
    
    def test_sensitive_data_interpretation(self):
        """Test interpretation when data has heterogeneous trends."""
        np.random.seed(42)
        n_units, n_periods = 200, 12
        treatment_period = 7
        data = []
        n_treated = n_units // 2
        for i in range(n_units):
            alpha_i = np.random.normal(0, 1)
            treated = i < n_treated
            beta_i = 0.3 if treated else 0
            for t in range(1, n_periods + 1):
                gamma_t = 0.1 * t
                d_it = 1 if treated and t >= treatment_period else 0
                tau = 2.0 if d_it else 0
                y = alpha_i + gamma_t + beta_i * t + tau + np.random.normal(0, 0.5)
                data.append({
                    'unit': i,
                    'time': t,
                    'Y': y,
                    'first_treat': treatment_period if treated else 0,
                })
        data = pd.DataFrame(data)
        result = robustness_pre_periods(
            data,
            y='Y',
            ivar='unit',
            tvar='time',
            gvar='first_treat',
            rolling='demean',
            pre_period_range=(2, 5),
            verbose=False
        )
        assert result.sensitivity_ratio > 0.05


class TestVisualization:
    """Test visualization functionality."""
    
    @pytest.fixture
    def panel_data(self):
        """Generate panel data for visualization tests."""
        np.random.seed(42)
        n_units, n_periods = 100, 10
        treatment_period = 6
        data = []
        n_treated = n_units // 2
        for i in range(n_units):
            alpha_i = np.random.normal(0, 1)
            treated = i < n_treated
            for t in range(1, n_periods + 1):
                d_it = 1 if treated and t >= treatment_period else 0
                tau = 2.0 if d_it else 0
                y = alpha_i + tau + np.random.normal(0, 0.5)
                data.append({
                    'unit': i,
                    'time': t,
                    'Y': y,
                    'first_treat': treatment_period if treated else 0,
                })
        return pd.DataFrame(data)
    
    def test_plot_generation(self, panel_data):
        """Test that plot can be generated."""
        result = robustness_pre_periods(
            panel_data,
            y='Y',
            ivar='unit',
            tvar='time',
            gvar='first_treat',
            rolling='demean',
            pre_period_range=(2, 4),
            verbose=False
        )
        try:
            import matplotlib
            matplotlib.use('Agg')
            fig = result.plot()
            assert fig is not None
        except ImportError:
            pytest.skip("matplotlib not available")

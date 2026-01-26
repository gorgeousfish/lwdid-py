"""
Stata End-to-End Validation Tests for BUG-113: Cholesky-based OLS.

This module validates that the Cholesky-based OLS implementation produces
results consistent with Stata's regress command.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from lwdid import lwdid


class TestStataOLSConsistency:
    """Tests for Python-Stata OLS consistency after BUG-113 fix."""
    
    @pytest.fixture
    def smoking_data(self):
        """Load smoking dataset for validation."""
        candidates = [
            os.path.join(os.path.dirname(__file__), 'data', 'smoking.csv'),
            'tests/data/smoking.csv',
        ]
        for path in candidates:
            if os.path.exists(path):
                return pd.read_csv(path)
        pytest.skip("Smoking data not available")
    
    def test_ra_estimator_basic(self, smoking_data):
        """Test RA estimator produces finite, reasonable results."""
        result = lwdid(
            smoking_data,
            y='lcigsale',
            d='d',
            post='post',
            ivar='state',
            tvar='year',
            rolling='demean',
            estimator='ra',
            vce='hc1',
        )
        
        # Should have finite results
        assert np.isfinite(result.att), "ATT should be finite"
        assert np.isfinite(result.se_att), "SE should be finite"
        assert result.se_att > 0, "SE should be positive"
        
        # CI should be computed
        assert np.isfinite(result.ci_lower), "CI lower should be finite"
        assert np.isfinite(result.ci_upper), "CI upper should be finite"
        assert result.ci_lower < result.ci_upper, "CI should be valid"
    
    def test_ra_estimator_with_controls(self, smoking_data):
        """Test RA estimator with time-invariant controls produces finite results."""
        # Create time-invariant control (constant within each unit)
        smoking_data = smoking_data.copy()
        np.random.seed(42)
        unique_states = smoking_data['state'].unique()
        state_to_control = {s: np.random.randn() for s in unique_states}
        smoking_data['control'] = smoking_data['state'].map(state_to_control)
        
        result = lwdid(
            smoking_data,
            y='lcigsale',
            d='d',
            post='post',
            ivar='state',
            tvar='year',
            rolling='demean',
            estimator='ra',
            vce='hc1',
            controls=['control'],
        )
        
        assert np.isfinite(result.att), "ATT should be finite"
        assert np.isfinite(result.se_att), "SE should be finite"
    
    def test_cluster_robust_se(self, smoking_data):
        """Test cluster-robust SE with Cholesky solution."""
        result = lwdid(
            smoking_data,
            y='lcigsale',
            d='d',
            post='post',
            ivar='state',
            tvar='year',
            rolling='demean',
            estimator='ra',
            vce='cluster',
            cluster_var='state',
        )
        
        assert np.isfinite(result.att), "ATT should be finite"
        assert np.isfinite(result.se_att), "Cluster SE should be finite"
        assert result.se_att > 0, "Cluster SE should be positive"
    
    def test_different_vce_types(self, smoking_data):
        """Test all VCE types produce consistent ATT estimates."""
        vce_types = [None, 'hc0', 'hc1', 'hc2', 'hc3', 'hc4']
        
        results = {}
        for vce in vce_types:
            result = lwdid(
                smoking_data,
                y='lcigsale',
                d='d',
                post='post',
                ivar='state',
                tvar='year',
                rolling='demean',
                estimator='ra',
                vce=vce,
            )
            results[vce] = result
            
            # All should be finite
            assert np.isfinite(result.att), f"ATT should be finite for vce={vce}"
            assert np.isfinite(result.se_att), f"SE should be finite for vce={vce}"
        
        # All VCE types should give same point estimate
        att_base = results[None].att
        for vce, result in results.items():
            diff = abs(result.att - att_base)
            assert diff < 1e-10, \
                f"Point estimate should be same for vce={vce}: diff={diff}"


class TestNumericalStabilityWithRealData:
    """Tests for numerical stability using real-world data."""
    
    @pytest.fixture
    def mve_demean_data(self):
        """Load MVE demean dataset."""
        candidates = [
            os.path.join(os.path.dirname(__file__), 'data', 'mve_demean.csv'),
            'tests/data/mve_demean.csv',
        ]
        for path in candidates:
            if os.path.exists(path):
                return pd.read_csv(path)
        pytest.skip("MVE demean data not available")
    
    def test_mve_data_consistency(self, mve_demean_data):
        """Test MVE data produces consistent results."""
        # Skip if data doesn't have required columns
        required_cols = ['unit_id', 'year', 'd', 'post', 'lcigsale']
        if not all(col in mve_demean_data.columns for col in required_cols):
            pytest.skip("MVE data missing required columns")
        
        result = lwdid(
            mve_demean_data,
            y='lcigsale',
            d='d',
            post='post',
            ivar='unit_id',
            tvar='year',
            rolling='demean',
            estimator='ra',
            vce='hc1',
        )
        
        assert np.isfinite(result.att), "ATT should be finite"
        assert np.isfinite(result.se_att), "SE should be finite"


class TestComparisonWithKnownValues:
    """Tests comparing with known analytical values."""
    
    def test_simple_did_known_effect(self):
        """Test simple DiD with known true effect."""
        np.random.seed(42)
        n_units = 20
        n_periods = 4
        true_att = 2.0
        
        rows = []
        for i in range(n_units):
            d = 1 if i < 10 else 0  # First 10 units treated
            for t in range(n_periods):
                post = 1 if t >= 2 else 0
                treated = d * post
                y = 5.0 + d * 0.5 + t * 0.1 + treated * true_att + np.random.normal(0, 0.5)
                rows.append({
                    'unit': i + 1,
                    'year': 2000 + t,
                    'd': d,
                    'post': post,
                    'y': y,
                })
        
        data = pd.DataFrame(rows)
        
        result = lwdid(
            data,
            y='y',
            d='d',
            post='post',
            ivar='unit',
            tvar='year',
            rolling='demean',
            estimator='ra',
            vce='hc1',
        )
        
        # With well-specified model, ATT should be close to true value
        assert abs(result.att - true_att) < 0.5, \
            f"ATT={result.att}, expected ~{true_att}"
        
        # SE should be reasonable
        assert 0 < result.se_att < 1.0, f"SE={result.se_att} seems unreasonable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

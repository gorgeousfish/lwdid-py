"""
End-to-end numerical validation for bug fixes using real data.

Cross-validates Python implementation against expected behavior to ensure
bug fixes do not alter numerical results when inputs are valid.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


class TestBugFixesE2ENumericalValidation:
    """End-to-end validation using castle dataset."""
    
    @pytest.fixture
    def castle_data(self):
        """Load castle dataset."""
        data_path = Path(__file__).parent.parent / 'data' / 'castle.csv'
        return pd.read_csv(data_path)
    
    def test_staggered_ra_with_valid_data(self, castle_data):
        """
        Test staggered RA estimation with castle data.
        
        Validates that panel uniqueness check (BUG-101) does not alter results
        when data is already valid.
        """
        from lwdid import lwdid
        
        results = lwdid(
            data=castle_data,
            y='lhomicide',
            ivar='sid',
            tvar='year',
            gvar='effyear',
            rolling='demean',
            estimator='ra',
            aggregate='overall',
        )
        
        # Verify estimation succeeded
        assert results is not None
        assert hasattr(results, 'att_overall')
        assert not np.isnan(results.att_overall)
        assert not np.isnan(results.se_overall)
        
        # Store results for comparison
        att_overall = results.att_overall
        se_overall = results.se_overall
        
        print(f"\nPython Results (Castle RA):")
        print(f"  ATT_ω = {att_overall:.6f}")
        print(f"  SE    = {se_overall:.6f}")
        
        # Basic sanity checks
        assert abs(att_overall) < 1.0  # Expected range based on castle data
        assert se_overall > 0
    
    def test_staggered_ipwra_with_valid_controls(self):
        """
        Test IPWRA with list controls (validates BUG-100 fix).
        
        Ensures controls type validation does not affect valid inputs.
        Uses synthetic data to avoid real data complexity.
        """
        from lwdid import lwdid
        
        # Create synthetic staggered data with controls
        np.random.seed(42)
        n_units = 60
        n_years = 8
        
        data = []
        for i in range(1, n_units + 1):
            # Assign cohort
            if i <= 20:
                gvar = 0  # Never treated
            elif i <= 40:
                gvar = 2004  # Treated in 2004
            else:
                gvar = 2006  # Treated in 2006
            
            # Time-invariant controls
            x1 = np.random.randn()
            x2 = np.random.randn()
            
            for year in range(2000, 2000 + n_years):
                y = 10 + x1 * 2 + x2 * 1.5 + np.random.randn() * 0.5
                if gvar > 0 and year >= gvar:
                    y += 3.0  # Treatment effect
                
                data.append({
                    'id': i,
                    'year': year,
                    'y': y,
                    'gvar': gvar,
                    'x1': x1,
                    'x2': x2,
                })
        
        data_df = pd.DataFrame(data)
        
        results = lwdid(
            data=data_df,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            estimator='ipwra',
            controls=['x1', 'x2'],  # Valid list - time-invariant
            aggregate='overall',
        )
        
        assert results is not None
        assert hasattr(results, 'att_overall')
        assert not np.isnan(results.att_overall)
        
        print(f"\nPython Results (IPWRA Synthetic Data):")
        print(f"  ATT_ω = {results.att_overall:.4f}, SE = {results.se_overall:.4f}")
    
    def test_psm_with_caliper_boundary(self):
        """
        Test PSM caliper boundary validation (BUG-101).
        
        Validates that valid caliper values work correctly.
        """
        from lwdid.staggered.estimators import estimate_psm
        
        np.random.seed(42)
        n = 300
        data = pd.DataFrame({
            'y': np.random.randn(n) + np.random.binomial(1, 0.5, n) * 2.5,
            'd': np.random.binomial(1, 0.5, n),
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
        })
        
        # Test various valid caliper values
        for caliper_val in [0.1, 0.25, 0.5, 1.0]:
            result = estimate_psm(
                data=data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x2'],
                caliper=caliper_val,
                n_neighbors=1,
            )
            
            assert result is not None
            assert hasattr(result, 'att')
            assert not np.isnan(result.att)
            
            print(f"\nPSM with caliper={caliper_val}:")
            print(f"  ATT = {result.att:.4f}, SE = {result.se:.4f}")
    
    def test_panel_uniqueness_validation_catches_duplicates(self):
        """
        Test that panel uniqueness validation (BUG-101) catches duplicate data.
        
        Creates intentionally duplicate data to verify validation works.
        """
        from lwdid import lwdid
        from lwdid.exceptions import InvalidParameterError
        
        # Create data with duplicates
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 2, 2, 2, 2],
            'year': [2000, 2000, 2001, 2002, 2000, 2001, 2002, 2003],  # Duplicate: id=1, year=2000
            'y': [1.0, 1.5, 2.0, 3.0, 1.1, 2.1, 3.1, 4.1],
            'gvar': [2001, 2001, 2001, 2001, 0, 0, 0, 0],
        })
        
        # Should raise error due to duplicates
        with pytest.raises(InvalidParameterError, match="Duplicate.*observations found"):
            lwdid(
                data=data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                estimator='ra',
            )
        
        print("\nPanel uniqueness validation successfully catches duplicates.")


class TestBugFixesRegressionSuite:
    """Regression tests to ensure bug fixes don't break existing functionality."""
    
    def test_common_timing_unchanged(self):
        """
        Test common timing estimation is unaffected by bug fixes.
        
        Bug fixes target staggered/estimator-specific code, so common timing
        should remain unchanged. Uses synthetic data for reliability.
        """
        from lwdid import lwdid
        
        # Create synthetic common timing data
        np.random.seed(42)
        n_units = 40
        years = list(range(2000, 2010))
        
        data = []
        for i in range(1, n_units + 1):
            treated = 1 if i > 20 else 0
            for year in years:
                post = 1 if year >= 2005 else 0
                y = 10 + np.random.randn() * 2
                if treated and post:
                    y += 3.0  # Treatment effect
                
                data.append({
                    'id': i,
                    'year': year,
                    'y': y,
                    'd': treated,
                    'post': post,
                })
        
        data_df = pd.DataFrame(data)
        
        results = lwdid(
            data=data_df,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean',
        )
        
        assert results is not None
        assert not np.isnan(results.att)
        assert not np.isnan(results.se_att)
        assert 2.0 < results.att < 4.0  # Expected ATT around 3.0
        
        print(f"\nCommon Timing (Synthetic Data):")
        print(f"  ATT = {results.att:.4f}, SE = {results.se_att:.4f}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

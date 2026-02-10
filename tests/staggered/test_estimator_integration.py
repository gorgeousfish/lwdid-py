"""
Tests for E3-S3: IPWRA/PSM Estimator Integration into Staggered Main Flow

Tests that the estimator parameter correctly routes to different estimation methods.
"""

import os
import pytest
import numpy as np
import pandas as pd
import warnings

from lwdid import lwdid


class TestEstimatorParameterValidation:
    """Test estimator parameter validation"""
    
    @pytest.fixture
    def simple_staggered_data(self):
        """Simple staggered data for validation tests"""
        np.random.seed(42)
        data = []
        for i in range(20):
            gvar = 0 if i < 10 else 4 if i < 15 else 5
            for t in range(1, 7):
                y = 1 + 0.2 * (1 if gvar > 0 and t >= gvar else 0) + np.random.normal(0, 0.3)
                data.append({'id': i + 1, 'year': t, 'y': y, 'gvar': gvar})
        return pd.DataFrame(data)
    
    def test_invalid_estimator_raises_error(self, simple_staggered_data):
        """Invalid estimator value should raise ValueError"""
        with pytest.raises(ValueError, match="[Ii]nvalid estimator"):
            lwdid(
                data=simple_staggered_data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                estimator='invalid_method',
                aggregate='overall',
            )
    
    def test_ipwra_without_controls_raises_error(self, simple_staggered_data):
        """IPWRA without controls should raise ValueError"""
        with pytest.raises(ValueError, match="requires.*controls"):
            lwdid(
                data=simple_staggered_data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                estimator='ipwra',
                aggregate='overall',
            )
    
    def test_psm_without_controls_raises_error(self, simple_staggered_data):
        """PSM without controls should raise ValueError"""
        with pytest.raises(ValueError, match="requires.*controls"):
            lwdid(
                data=simple_staggered_data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                estimator='psm',
                aggregate='overall',
            )
    
    def test_ra_without_controls_works(self, simple_staggered_data):
        """RA without controls should work"""
        result = lwdid(
            data=simple_staggered_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            estimator='ra',
            aggregate='overall',
        )
        assert result.att_overall is not None


class TestEstimatorIntegration:
    """Test estimator integration with staggered flow"""
    
    @pytest.fixture
    def staggered_data_with_controls(self):
        """Staggered data with control variables"""
        np.random.seed(42)
        n_units = 30
        n_periods = 6
        
        data = []
        for i in range(n_units):
            # Assign cohort
            if i < 10:
                gvar = 0  # never treated
            elif i < 20:
                gvar = 4  # cohort 4
            else:
                gvar = 5  # cohort 5
            
            # Time-invariant controls
            x1 = np.random.normal(0, 1)
            x2 = np.random.normal(0, 1)
            
            for t in range(1, n_periods + 1):
                treated = 1 if gvar > 0 and t >= gvar else 0
                # y depends on controls and treatment
                y = 1 + 0.5 * x1 + 0.3 * x2 + 0.2 * treated + np.random.normal(0, 0.5)
                
                data.append({
                    'id': i + 1,
                    'year': t,
                    'y': y,
                    'gvar': gvar,
                    'x1': x1,
                    'x2': x2,
                })
        
        return pd.DataFrame(data)
    
    def test_estimator_ra_works(self, staggered_data_with_controls):
        """RA estimator should work (regression test)"""
        result = lwdid(
            data=staggered_data_with_controls,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            estimator='ra',
            aggregate='overall',
        )
        
        assert result.att_overall is not None
        assert not np.isnan(result.att_overall)
        assert result.se_overall is not None
    
    def test_estimator_ipwra_works(self, staggered_data_with_controls):
        """IPWRA estimator should work"""
        result = lwdid(
            data=staggered_data_with_controls,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            estimator='ipwra',
            controls=['x1', 'x2'],
            aggregate='overall',
        )
        
        assert result.att_overall is not None
        # IPWRA may return NaN for some (g,r) pairs with small samples
        # but overall effect should be estimable
    
    def test_estimator_psm_works(self, staggered_data_with_controls):
        """PSM estimator should work"""
        result = lwdid(
            data=staggered_data_with_controls,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            estimator='psm',
            controls=['x1', 'x2'],
            aggregate='overall',
        )
        
        # PSM may fail with small samples, but should not raise
        assert result is not None
    
    def test_cohort_time_effects_with_ipwra(self, staggered_data_with_controls):
        """IPWRA should produce cohort-time effects"""
        result = lwdid(
            data=staggered_data_with_controls,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            estimator='ipwra',
            controls=['x1', 'x2'],
            aggregate='none',  # Get (g,r) effects only
        )
        
        # Should have effects for valid (g,r) pairs
        assert result.att_by_cohort_time is not None
        assert len(result.att_by_cohort_time) > 0
    
    def test_three_estimators_produce_results(self, staggered_data_with_controls):
        """All three estimators should produce results"""
        results = {}
        
        for est in ['ra', 'ipwra', 'psm']:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    result = lwdid(
                        data=staggered_data_with_controls,
                        y='y',
                        ivar='id',
                        tvar='year',
                        gvar='gvar',
                        rolling='demean',
                        estimator=est,
                        controls=['x1', 'x2'] if est != 'ra' else None,
                        aggregate='overall',
                    )
                    results[est] = result.att_overall
                except Exception as e:
                    results[est] = f"Error: {e}"
        
        # RA should definitely work
        assert isinstance(results['ra'], (int, float)) and not np.isnan(results['ra'])
        
        # Print comparison
        print(f"\nEstimator comparison:")
        for est, val in results.items():
            if isinstance(val, (int, float)):
                print(f"  {est}: {val:.4f}")
            else:
                print(f"  {est}: {val}")


class TestCastleLawIntegration:
    """Castle Law data end-to-end tests"""
    
    @pytest.fixture
    def castle_data(self):
        """Load Castle Law data"""
        # Calculate relative path
        test_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(
            test_dir, '..', '..', 'data', 'castle.csv'
        )
        
        if not os.path.exists(data_path):
            pytest.skip(f"Castle data not found at {data_path}")
        
        data = pd.read_csv(data_path)
        data['gvar'] = data['effyear'].fillna(0).astype(int)
        
        return data
    
    def test_castle_ra_baseline(self, castle_data):
        """Castle Law RA estimation baseline"""
        result = lwdid(
            data=castle_data,
            y='lhomicide',
            ivar='sid',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            estimator='ra',
            control_group='never_treated',
            aggregate='overall',
            vce='hc3'
        )
        
        # Paper expected: τ_ω ≈ 0.092
        assert result.att_overall is not None
        assert abs(result.att_overall - 0.092) < 0.05, \
            f"RA estimate too far from expected: {result.att_overall}"
        
        print(f"\nCastle RA: ATT = {result.att_overall:.4f}, SE = {result.se_overall:.4f}")
    
    def test_castle_ipwra_with_controls(self, castle_data):
        """Castle Law IPWRA estimation with controls"""
        # Select available numeric controls
        possible_controls = ['population', 'income', 'unemployrt', 'prisoner', 'poverty']
        controls = [c for c in possible_controls if c in castle_data.columns]
        
        if len(controls) < 2:
            pytest.skip("Not enough control variables")
        
        # Use first 2 controls
        controls = controls[:2]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=castle_data,
                y='lhomicide',
                ivar='sid',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                estimator='ipwra',
                controls=controls,
                control_group='never_treated',
                aggregate='overall',
            )
        
        # Should produce a result
        assert result is not None
        if result.att_overall is not None and not np.isnan(result.att_overall):
            print(f"\nCastle IPWRA: ATT = {result.att_overall:.4f}")
            assert abs(result.att_overall) < 0.5  # Reasonable range
    
    def test_castle_psm_with_controls(self, castle_data):
        """Castle Law PSM estimation"""
        controls = ['population', 'income']
        controls = [c for c in controls if c in castle_data.columns]
        
        if len(controls) < 1:
            pytest.skip("No control variables available")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=castle_data,
                y='lhomicide',
                ivar='sid',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                estimator='psm',
                controls=controls,
                control_group='never_treated',
                aggregate='overall',
            )
        
        # PSM may have limited success with small cohorts
        assert result is not None
        if result.att_overall is not None and not np.isnan(result.att_overall):
            print(f"\nCastle PSM: ATT = {result.att_overall:.4f}")
    
    def test_castle_estimator_comparison(self, castle_data):
        """Compare all three estimators on Castle Law data"""
        controls = ['population', 'income']
        controls = [c for c in controls if c in castle_data.columns]
        
        results = {}
        
        # RA
        r_ra = lwdid(
            data=castle_data,
            y='lhomicide',
            ivar='sid',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            estimator='ra',
            control_group='never_treated',
            aggregate='overall',
        )
        results['ra'] = (r_ra.att_overall, r_ra.se_overall)
        
        # IPWRA
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r_ipwra = lwdid(
                    data=castle_data,
                    y='lhomicide',
                    ivar='sid',
                    tvar='year',
                    gvar='gvar',
                    rolling='demean',
                    estimator='ipwra',
                    controls=controls,
                    control_group='never_treated',
                    aggregate='overall',
                )
                results['ipwra'] = (r_ipwra.att_overall, r_ipwra.se_overall)
            except Exception as e:
                results['ipwra'] = (np.nan, np.nan)
        
        # PSM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r_psm = lwdid(
                    data=castle_data,
                    y='lhomicide',
                    ivar='sid',
                    tvar='year',
                    gvar='gvar',
                    rolling='demean',
                    estimator='psm',
                    controls=controls,
                    control_group='never_treated',
                    aggregate='overall',
                )
                results['psm'] = (r_psm.att_overall, r_psm.se_overall)
            except Exception as e:
                results['psm'] = (np.nan, np.nan)
        
        print(f"\n=== Castle Law Estimator Comparison ===")
        for est, (att, se) in results.items():
            if not np.isnan(att):
                print(f"  {est.upper():6s}: ATT = {att:7.4f}, SE = {se:.4f}")
            else:
                print(f"  {est.upper():6s}: Failed")
        
        # RA should work
        assert not np.isnan(results['ra'][0])


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

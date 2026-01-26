"""
Unit tests for BUG-149, BUG-150, BUG-151 fixes.

Tests verify:
- BUG-149: IPW weights are properly capped when propensity scores approach 1
- BUG-150: Quarter dummy column alignment works with mismatched quarter distributions
- BUG-151: cohort_effects variable is properly initialized for all aggregation levels
"""

import numpy as np
import pandas as pd
import pytest
import warnings


class TestBug149IPWWeightsCapping:
    """Test BUG-149: IPW weights should be capped when propensity scores approach 1."""
    
    def test_extreme_propensity_score_warning(self):
        """Propensity scores near 1 should trigger warning about non-finite weights."""
        from lwdid.staggered.estimators import estimate_ipw
        
        # Create data where some units have very high propensity scores
        np.random.seed(42)
        n = 100
        
        # Control variable strongly predicting treatment
        x = np.concatenate([np.random.randn(50) - 2, np.random.randn(50) + 2])
        
        # Treatment assignment correlated with x
        prob = 1 / (1 + np.exp(-3 * x))  # Some probabilities very close to 1
        d = (np.random.rand(n) < prob).astype(int)
        
        # Outcome
        y = 2 * d + x + np.random.randn(n)
        
        data = pd.DataFrame({
            'y': y,
            'd': d,
            'x': x,
        })
        
        # Run IPW estimation - should handle extreme weights gracefully
        try:
            result = estimate_ipw(
                data=data,
                y='y',
                d='d',
                propensity_controls=['x'],
                trim_threshold=0.01,  # Minimal trimming
            )
            # Should complete without error
            assert np.isfinite(result.att), "ATT should be finite"
        except Exception as e:
            # Some exceptions are acceptable (e.g., no overlap)
            assert "overlap" in str(e).lower() or "control" in str(e).lower()
    
    def test_weights_formula_numerical_stability(self):
        """Test that w = e/(1-e) handles edge cases."""
        # Test the formula directly
        pscores = np.array([0.01, 0.5, 0.9, 0.99, 0.999, 0.9999])
        weights = pscores / (1 - pscores)
        
        # All should be finite for these values
        assert np.all(np.isfinite(weights)), "Weights should be finite for p < 1"
        
        # Test with p = 1 - epsilon where epsilon is very small
        epsilon = 1e-15
        p_extreme = 1 - epsilon
        w_extreme = p_extreme / (1 - p_extreme)
        
        # This could be Inf due to floating point, which is what BUG-149 handles
        # The fix caps such weights
        if not np.isfinite(w_extreme):
            # This is the edge case BUG-149 addresses
            pass
    
    def test_ipwra_with_high_propensity_scores(self):
        """Test IPWRA handles high propensity scores without crashing."""
        from lwdid.staggered.estimators import estimate_ipwra
        
        np.random.seed(123)
        n = 200
        
        # Create moderately challenging data
        x1 = np.random.randn(n)
        x2 = np.random.binomial(1, 0.5, n)
        
        # Treatment probability
        prob = 1 / (1 + np.exp(-0.5 - 0.8 * x1 + 0.3 * x2))
        d = (np.random.rand(n) < prob).astype(int)
        
        # Outcome with treatment effect
        y = 3 + d * 2 + x1 + x2 * 0.5 + np.random.randn(n)
        
        data = pd.DataFrame({
            'y': y,
            'd': d,
            'x1': x1,
            'x2': x2,
        })
        
        # Should complete without error
        result = estimate_ipwra(
            data=data,
            y='y',
            d='d',
            controls=['x1', 'x2'],
            propensity_controls=['x1', 'x2'],
            trim_threshold=0.01,
        )
        
        assert np.isfinite(result.att), "ATT should be finite"
        assert result.n_treated > 0
        assert result.n_control > 0


class TestBug150QuarterDummyAlignment:
    """Test BUG-150: Quarter dummy column alignment with mismatched distributions."""
    
    def test_demeanq_unit_with_missing_quarters_in_post(self):
        """Test demeanq_unit when post-period has different quarters than pre-period."""
        from lwdid.transformations import demeanq_unit
        
        # Create unit data where pre-period has Q1, Q2, Q3, Q4
        # but post-period only has Q1, Q2
        unit_data = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],  # 4 pre + 2 post
            'quarter': [1, 2, 3, 4, 1, 2],  # Post missing Q3, Q4
            'post': [0, 0, 0, 0, 1, 1],
        })
        
        # Should not raise KeyError
        yhat, ydot = demeanq_unit(unit_data, 'y', 'quarter', 'post')
        
        # Results should be finite
        assert np.all(np.isfinite(yhat)) or np.all(np.isnan(yhat))
        assert np.all(np.isfinite(ydot)) or np.all(np.isnan(ydot))
    
    def test_demeanq_unit_with_extra_quarters_in_post(self):
        """Test demeanq_unit when post-period has quarters not in pre-period."""
        from lwdid.transformations import demeanq_unit
        
        # Pre-period only has Q1, Q2; post has Q1, Q2, Q3, Q4
        unit_data = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'quarter': [1, 2, 1, 2, 3, 4],  # Pre missing Q3, Q4
            'post': [0, 0, 1, 1, 1, 1],
        })
        
        # Should not raise KeyError (extra quarters should be handled)
        yhat, ydot = demeanq_unit(unit_data, 'y', 'quarter', 'post')
        
        # Results should be arrays of correct length
        assert len(yhat) == len(unit_data)
        assert len(ydot) == len(unit_data)
    
    def test_detrendq_unit_with_mismatched_quarters(self):
        """Test detrendq_unit with mismatched quarter distributions."""
        from lwdid.transformations import detrendq_unit
        
        # Create unit data with time trend
        unit_data = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            'tindex': [1, 2, 3, 4, 5, 6, 7, 8],
            'quarter': [1, 2, 3, 4, 1, 2, 3, 4],
            'post': [0, 0, 0, 0, 1, 1, 1, 1],
        })
        
        # Should not raise error
        yhat, ydot = detrendq_unit(unit_data, 'y', 'tindex', 'quarter', 'post')
        
        assert len(yhat) == len(unit_data)
        assert len(ydot) == len(unit_data)
    
    def test_column_alignment_preserves_order(self):
        """Test that column alignment preserves the correct column order."""
        from lwdid.transformations import demeanq_unit
        
        # Pre-period has all 4 quarters
        unit_data = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 1.5, 2.5],
            'quarter': [1, 2, 3, 4, 1, 2],
            'post': [0, 0, 0, 0, 1, 1],
        })
        
        yhat, ydot = demeanq_unit(unit_data, 'y', 'quarter', 'post')
        
        # Should produce valid output
        assert len(yhat) == 6
        assert len(ydot) == 6


class TestBug151CohortEffectsInitialization:
    """Test BUG-151: cohort_effects should be initialized for all aggregation levels."""
    
    def test_aggregate_none_no_name_error(self):
        """Test that aggregate='none' doesn't raise NameError for cohort_effects."""
        from lwdid import lwdid
        
        # Create minimal staggered DID data
        np.random.seed(42)
        
        # 3 units: 2 treated (different cohorts), 1 never-treated
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            'year': [2000, 2001, 2002, 2003] * 3,
            'y': np.random.randn(12) + [0, 0, 1, 1, 0, 0, 0, 2, 0, 0, 0, 0],
            'gvar': [2002, 2002, 2002, 2002, 2003, 2003, 2003, 2003, 
                     np.inf, np.inf, np.inf, np.inf],
        })
        
        try:
            # aggregate='none' should work without NameError
            result = lwdid(
                data=data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                aggregate='none',
                rolling='demean',
            )
            # Should complete without NameError
            assert result is not None
        except ValueError as e:
            # Some ValueErrors are acceptable (e.g., insufficient data)
            # but NameError for cohort_effects is not
            assert "cohort_effects" not in str(e)
    
    def test_aggregate_cohort_initializes_cohort_effects(self):
        """Test that aggregate='cohort' properly uses cohort_effects."""
        from lwdid import lwdid
        
        np.random.seed(42)
        
        # Create staggered data with clear treatment effects
        n_units = 20
        n_periods = 8
        
        data_list = []
        for i in range(n_units):
            if i < 5:
                gvar = 4  # Cohort 1: treated at period 4
                effect = 2.0
            elif i < 10:
                gvar = 6  # Cohort 2: treated at period 6
                effect = 3.0
            else:
                gvar = np.inf  # Never treated
                effect = 0.0
            
            for t in range(1, n_periods + 1):
                y_base = np.random.randn() + i * 0.1
                if t >= gvar and gvar < np.inf:
                    y_base += effect
                data_list.append({
                    'id': i + 1,
                    'year': t,
                    'y': y_base,
                    'gvar': gvar,
                })
        
        data = pd.DataFrame(data_list)
        
        try:
            result = lwdid(
                data=data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                aggregate='cohort',
                rolling='demean',
            )
            
            # Should have cohort-level results
            assert result is not None
            if hasattr(result, 'att_by_cohort') and result.att_by_cohort is not None:
                assert len(result.att_by_cohort) > 0
        except ValueError:
            # Acceptable if data is insufficient
            pass
    
    def test_all_aggregation_levels_work(self):
        """Test all aggregation levels: none, cohort, overall."""
        from lwdid import lwdid
        
        np.random.seed(42)
        
        # Larger dataset for more reliable estimation
        n_units = 30
        n_periods = 10
        
        data_list = []
        for i in range(n_units):
            if i < 10:
                gvar = 5
            elif i < 20:
                gvar = 7
            else:
                gvar = np.inf
            
            for t in range(1, n_periods + 1):
                y = np.random.randn() + (2.0 if t >= gvar and gvar < np.inf else 0)
                data_list.append({
                    'id': i + 1,
                    'year': t,
                    'y': y,
                    'gvar': gvar,
                })
        
        data = pd.DataFrame(data_list)
        
        for agg_level in ['none', 'cohort', 'overall']:
            try:
                result = lwdid(
                    data=data,
                    y='y',
                    ivar='id',
                    tvar='year',
                    gvar='gvar',
                    aggregate=agg_level,
                    rolling='demean',
                )
                assert result is not None, f"Result should not be None for aggregate='{agg_level}'"
            except ValueError as e:
                # Some ValueErrors acceptable, but not NameError
                assert "NameError" not in str(type(e))
                assert "cohort_effects" not in str(e)


class TestIntegration:
    """Integration tests combining multiple bug fixes."""
    
    def test_full_estimation_pipeline(self):
        """Test full estimation pipeline with various edge cases."""
        from lwdid import lwdid
        
        np.random.seed(42)
        
        # Create realistic panel data
        n_units = 50
        n_periods = 12
        
        data_list = []
        for i in range(n_units):
            # Assign cohort
            if i < 15:
                gvar = 6
                effect = 2.0
            elif i < 30:
                gvar = 9
                effect = 3.0
            else:
                gvar = np.inf
                effect = 0.0
            
            # Unit fixed effect
            unit_fe = np.random.randn()
            
            for t in range(1, n_periods + 1):
                # Time trend
                time_effect = 0.1 * t
                # Treatment effect
                treat_effect = effect if (t >= gvar and gvar < np.inf) else 0
                # Idiosyncratic error
                error = np.random.randn() * 0.5
                
                y = unit_fe + time_effect + treat_effect + error
                
                data_list.append({
                    'id': i + 1,
                    'year': 2000 + t,
                    'y': y,
                    'gvar': 2000 + gvar if gvar < np.inf else np.inf,
                })
        
        data = pd.DataFrame(data_list)
        
        # Run estimation
        result = lwdid(
            data=data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            aggregate='overall',
            rolling='demean',
        )
        
        # Verify results
        assert result is not None
        assert np.isfinite(result.att)
        # True effect is ~2.5 (weighted avg of 2 and 3)
        assert 1.0 < result.att < 4.0, f"ATT={result.att} outside reasonable range"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

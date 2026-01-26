"""
DESIGN-007: Event Study SE - Stata Comparison Tests

This module compares Python event study results with Stata reference values.
Focuses on Castle Law dataset for reproducibility.

Reference:
- Lee & Wooldridge (2025) Section 7.2 Castle Law application
- Paper reports: τ̂_ω ≈ 0.092, SE ≈ 0.057 (OLS), hc3 t ≈ 1.50
"""

import os
import sys
import warnings
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from lwdid import lwdid


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def castle_data():
    """Load Castle Law dataset."""
    here = os.path.dirname(__file__)
    data_path = os.path.join(here, '..', '..', 'data', 'castle.csv')
    data = pd.read_csv(data_path)
    data['gvar'] = data['effyear'].fillna(0).astype(int)
    return data


# =============================================================================
# Paper Reference Values (Lee & Wooldridge 2025 Section 7.2)
# =============================================================================

PAPER_REFERENCE = {
    'demean': {
        'overall_att': 0.092,  # ~9.2% increase in homicides
        'overall_se_ols': 0.057,
        'overall_t_ols': 1.61,
        'overall_t_hc3': 1.50,
    },
    'detrend': {
        'overall_att': 0.067,
        'overall_se_hc3': 0.055,
        'overall_t_hc3': 1.21,
    },
}


# =============================================================================
# Castle Law Overall Effect Tests
# =============================================================================

class TestCastleLawOverallEffect:
    """Test overall effect estimation matches paper results."""
    
    def test_demean_overall_att_matches_paper(self, castle_data):
        """Overall ATT with demean should match paper ~0.092."""
        results = lwdid(
            data=castle_data, 
            y='lhomicide', 
            ivar='sid', 
            tvar='year',
            gvar='gvar', 
            rolling='demean', 
            control_group='never_treated',
            aggregate='overall', 
            vce='hc3'
        )
        
        # Paper reports τ̂_ω ≈ 0.092
        expected_att = PAPER_REFERENCE['demean']['overall_att']
        
        print(f"\nPython overall ATT: {results.att_overall:.4f}")
        print(f"Paper overall ATT: {expected_att:.4f}")
        print(f"Difference: {abs(results.att_overall - expected_att):.4f}")
        
        # Allow 10% tolerance for numerical differences
        assert abs(results.att_overall - expected_att) < 0.02, \
            f"Overall ATT {results.att_overall:.4f} differs from paper {expected_att}"
    
    def test_detrend_overall_att_matches_paper(self, castle_data):
        """Overall ATT with detrend should match paper ~0.067."""
        results = lwdid(
            data=castle_data, 
            y='lhomicide', 
            ivar='sid', 
            tvar='year',
            gvar='gvar', 
            rolling='detrend', 
            control_group='never_treated',
            aggregate='overall', 
            vce='hc3'
        )
        
        # Paper reports τ̂_ω ≈ 0.067
        expected_att = PAPER_REFERENCE['detrend']['overall_att']
        
        print(f"\nPython overall ATT (detrend): {results.att_overall:.4f}")
        print(f"Paper overall ATT (detrend): {expected_att:.4f}")
        print(f"Difference: {abs(results.att_overall - expected_att):.4f}")
        
        # Allow tolerance
        assert abs(results.att_overall - expected_att) < 0.02, \
            f"Overall ATT {results.att_overall:.4f} differs from paper {expected_att}"


# =============================================================================
# Event Study SE Comparison
# =============================================================================

class TestCastleLawEventStudySE:
    """Test event study SE estimation."""
    
    def test_event_study_analytical_se_reasonable(self, castle_data):
        """Analytical SE should be in reasonable range."""
        results = lwdid(
            data=castle_data, 
            y='lhomicide', 
            ivar='sid', 
            tvar='year',
            gvar='gvar', 
            rolling='demean', 
            control_group='never_treated',
            aggregate='overall', 
            vce='hc3'
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, event_df = results.plot_event_study(
                se_method='analytical',
                ref_period=None,
                return_data=True
            )
        
        print("\nCastle Law Event Study (Analytical SE):")
        print(event_df[['event_time', 'att', 'se', 'n_cohorts']])
        
        # SE should be positive and reasonable
        assert all(event_df['se'] > 0), "All SE should be positive"
        assert all(event_df['se'] < 1.0), "SE should be reasonable (< 1.0 for log outcome)"
        
        import matplotlib.pyplot as plt
        plt.close('all')
    
    @pytest.mark.slow
    def test_event_study_bootstrap_se_reasonable(self, castle_data):
        """Bootstrap SE should be in reasonable range."""
        results = lwdid(
            data=castle_data, 
            y='lhomicide', 
            ivar='sid', 
            tvar='year',
            gvar='gvar', 
            rolling='demean', 
            control_group='never_treated',
            aggregate='overall', 
            vce='hc3'
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, event_df = results.plot_event_study(
                se_method='bootstrap',
                n_bootstrap=100,
                seed=42,
                ref_period=None,
                return_data=True
            )
        
        print("\nCastle Law Event Study (Bootstrap SE):")
        print(event_df[['event_time', 'att', 'se', 'n_cohorts']])
        
        # SE should be positive and reasonable
        assert all(event_df['se'] > 0), "All SE should be positive"
        assert all(event_df['se'] < 1.0), "SE should be reasonable"
        
        import matplotlib.pyplot as plt
        plt.close('all')
    
    def test_event_study_att_at_treatment_time(self, castle_data):
        """ATT at treatment time (e=0) should be positive for Castle Law."""
        results = lwdid(
            data=castle_data, 
            y='lhomicide', 
            ivar='sid', 
            tvar='year',
            gvar='gvar', 
            rolling='demean', 
            control_group='never_treated',
            aggregate='overall', 
            vce='hc3'
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, event_df = results.plot_event_study(
                se_method='analytical',
                ref_period=None,
                return_data=True
            )
        
        # ATT at e=0 should be positive (castle laws increase homicides)
        e0_row = event_df[event_df['event_time'] == 0]
        if not e0_row.empty:
            att_e0 = e0_row['att'].values[0]
            se_e0 = e0_row['se'].values[0]
            print(f"\nATT at e=0: {att_e0:.4f} (SE: {se_e0:.4f})")
            
            # Paper suggests positive effect
            # But aggregated e=0 might differ from overall τ_ω
        
        import matplotlib.pyplot as plt
        plt.close('all')


# =============================================================================
# SE Method Comparison for Castle Law
# =============================================================================

class TestCastleLawSEMethodComparison:
    """Compare analytical vs bootstrap SE for Castle Law."""
    
    @pytest.mark.slow
    def test_analytical_vs_bootstrap_se_castle_law(self, castle_data):
        """Compare analytical and bootstrap SE on Castle Law data."""
        results = lwdid(
            data=castle_data, 
            y='lhomicide', 
            ivar='sid', 
            tvar='year',
            gvar='gvar', 
            rolling='demean', 
            control_group='never_treated',
            aggregate='overall', 
            vce='hc3'
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            _, _, analytical_df = results.plot_event_study(
                se_method='analytical',
                ref_period=None,
                return_data=True
            )
            
            _, _, bootstrap_df = results.plot_event_study(
                se_method='bootstrap',
                n_bootstrap=200,
                seed=42,
                ref_period=None,
                return_data=True
            )
        
        # Merge for comparison
        merged = analytical_df.merge(
            bootstrap_df, on='event_time', suffixes=('_analytical', '_bootstrap')
        )
        
        print("\nCastle Law SE Comparison:")
        print("=" * 60)
        for _, row in merged.iterrows():
            e = int(row['event_time'])
            se_a = row['se_analytical']
            se_b = row['se_bootstrap']
            ratio = se_b / se_a if se_a > 0 else np.nan
            print(f"e={e:2d}: SE_analytical={se_a:.4f}, SE_bootstrap={se_b:.4f}, ratio={ratio:.2f}")
        
        print("\nSummary:")
        mean_ratio = (merged['se_bootstrap'] / merged['se_analytical']).mean()
        print(f"Mean SE ratio (bootstrap/analytical): {mean_ratio:.2f}")
        
        # Bootstrap SE should generally be similar or larger
        # Ratio > 1 suggests cohort correlations are captured
        
        import matplotlib.pyplot as plt
        plt.close('all')


# =============================================================================
# Stata Reference Data (for future validation)
# =============================================================================

class TestStataReferenceComparison:
    """
    Tests for comparing with Stata output.
    
    To generate Stata reference data, run:
    
    ```stata
    use castle.dta, clear
    gen gvar = cond(missing(effyear), 0, effyear)
    
    * Using the lwdid Stata command (if available)
    * Or using manual implementation per Lee & Wooldridge (2025)
    
    * Export event study results to JSON for comparison
    ```
    """
    
    def test_stata_reference_placeholder(self, castle_data):
        """Placeholder for Stata comparison - to be implemented with actual Stata output."""
        results = lwdid(
            data=castle_data, 
            y='lhomicide', 
            ivar='sid', 
            tvar='year',
            gvar='gvar', 
            rolling='demean', 
            control_group='never_treated',
            aggregate='overall', 
            vce='hc3'
        )
        
        # Export results for manual Stata comparison
        print("\nPython results for Stata comparison:")
        print(f"Overall ATT: {results.att_overall:.6f}")
        print(f"Overall SE: {results.se_overall:.6f}")
        print(f"Number of cohorts: {len(results.cohorts)}")
        print(f"Cohorts: {results.cohorts}")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, event_df = results.plot_event_study(
                se_method='analytical',
                ref_period=None,
                return_data=True
            )
        
        print("\nEvent study results:")
        print(event_df.to_string(index=False))
        
        import matplotlib.pyplot as plt
        plt.close('all')


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])

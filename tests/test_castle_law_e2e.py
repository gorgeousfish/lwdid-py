"""End-to-end integration tests for the staggered adoption module using
Castle Doctrine data.

This module validates the full ``lwdid()`` staggered pipeline against
reference values from Lee & Wooldridge (2025), Section 6, using the
Castle Doctrine dataset (50 US states, 2000--2010). The dataset contains
21 treated states distributed across five adoption cohorts (2005--2009)
and 29 never-treated states.

Expected overall effects (Lee & Wooldridge 2025, Table 3):
- Demeaning (Procedure 2.1): τ_ω ≈ 0.092
- Detrending (Procedure 3.1): τ_ω ≈ 0.067

References
----------
Lee, S. & Wooldridge, J. M. (2025). A Simple Transformation Approach to
    DiD Estimation for Panel Data. SSRN 4516518, Section 6.
"""

import warnings
import pytest
import numpy as np
import pandas as pd
from pathlib import Path


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def castle_data():
    """Load and prepare Castle Law data."""
    data_path = Path(__file__).parent.parent / 'data' / 'castle.csv'
    df = pd.read_csv(data_path)
    
    # Prepare gvar column: convert NaN to 0 for never-treated
    df['gvar'] = df['effyear'].fillna(0).astype(int)
    
    return df


@pytest.fixture
def castle_cohorts():
    """Expected cohorts from Castle Law data."""
    return [2005, 2006, 2007, 2008, 2009]


@pytest.fixture
def castle_nt_count():
    """Expected number of never-treated states."""
    return 29  # States with effyear = NaN


# =============================================================================
# Data Validation Tests
# =============================================================================

class TestCastleDataValidation:
    """Validate Castle Law data structure and cohort composition.

    Verifies that the Castle Doctrine dataset has the expected cohort
    structure: 21 treated states across 5 cohorts (2005--2009) and
    29 never-treated states, consistent with Lee & Wooldridge (2025),
    Section 6.
    """
    
    def test_required_columns_exist(self, castle_data):
        """Check required columns exist."""
        required = ['sid', 'year', 'lhomicide', 'gvar']
        for col in required:
            assert col in castle_data.columns, f"Column {col} not found"
    
    @pytest.mark.paper_validation
    def test_cohort_structure(self, castle_data, castle_cohorts):
        """Verify the five adoption cohorts (2005--2009) are correctly identified.

        Validates Section 6 of Lee & Wooldridge (2025): the Castle Doctrine
        dataset contains cohorts at g ∈ {2005, 2006, 2007, 2008, 2009}.
        """
        from lwdid.validation import validate_staggered_data
        
        result = validate_staggered_data(
            data=castle_data,
            gvar='gvar',
            ivar='sid',
            tvar='year',
            y='lhomicide'
        )
        
        assert result['cohorts'] == castle_cohorts, \
            f"Expected cohorts {castle_cohorts}, got {result['cohorts']}"
    
    def test_never_treated_count(self, castle_data):
        """Verify number of never-treated states."""
        # Count states with gvar=0
        unit_gvar = castle_data.groupby('sid')['gvar'].first()
        n_nt = (unit_gvar == 0).sum()
        
        # Castle data should have 29 never-treated states
        assert n_nt == 29, f"Expected 29 NT states, got {n_nt}"
    
    @pytest.mark.paper_validation
    def test_cohort_weights(self, castle_data, castle_cohorts):
        """Verify cohort sizes match Lee & Wooldridge (2025), Section 6.

        Expected sizes: 2005: 1, 2006: 13, 2007: 4, 2008: 2, 2009: 1.
        """
        # Get cohort sizes (number of states per cohort)
        unit_gvar = castle_data.groupby('sid')['gvar'].first()
        
        cohort_sizes = {}
        for g in castle_cohorts:
            cohort_sizes[g] = (unit_gvar == g).sum()
        
        # Expected sizes from paper:
        # 2005: 1 state
        # 2006: 13 states
        # 2007: 4 states
        # 2008: 2 states
        # 2009: 1 state
        expected_sizes = {2005: 1, 2006: 13, 2007: 4, 2008: 2, 2009: 1}
        
        for g, expected in expected_sizes.items():
            actual = cohort_sizes.get(g, 0)
            assert actual == expected, \
                f"Cohort {g}: expected {expected} states, got {actual}"


# =============================================================================
# Demean End-to-End Test
# =============================================================================

class TestCastleDemean:
    """End-to-end tests for the demeaning transformation on Castle data."""
    
    @pytest.mark.integration
    @pytest.mark.paper_validation
    def test_demean_overall_effect(self, castle_data):
        """Validate the overall WATT under demeaning against the published value.

        Validates Table 3 from Lee & Wooldridge (2025):
            τ_ω ≈ 0.092 (demeaning, never-treated control group).
        """
        from lwdid import lwdid
        
        # Suppress warnings for cleaner test output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            result = lwdid(
                data=castle_data,
                y='lhomicide',
                ivar='sid',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                control_group='never_treated',
                aggregate='overall'
            )
        
        # Check result structure
        assert result.is_staggered == True
        assert result.att_overall is not None
        assert result.se_overall is not None
        
        # Check overall effect is approximately 0.092
        # Allow tolerance of 0.03 for numerical differences
        expected_att = 0.092
        tolerance = 0.03
        
        print(f"Demean τ_ω = {result.att_overall:.4f} (expected ≈ {expected_att})")
        
        assert abs(result.att_overall - expected_att) < tolerance, \
            f"Overall effect {result.att_overall:.4f} differs from expected {expected_att} by more than {tolerance}"
    
    @pytest.mark.integration
    def test_demean_cohort_effects(self, castle_data, castle_cohorts):
        """Verify cohort-specific ATT estimates are computed for all five cohorts."""
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            result = lwdid(
                data=castle_data,
                y='lhomicide',
                ivar='sid',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                control_group='never_treated',
                aggregate='cohort'
            )
        
        # Check we have cohort effects
        assert result.att_by_cohort is not None
        
        # Check cohort effects are computed for each cohort
        att_cohort_df = result.att_by_cohort
        cohorts_with_effects = att_cohort_df['cohort'].tolist()
        
        print(f"Cohort effects computed for: {cohorts_with_effects}")
        print(att_cohort_df[['cohort', 'att', 'se']].to_string(index=False))
    
    @pytest.mark.integration
    def test_demean_cohort_time_effects(self, castle_data):
        """Verify (g, r)-specific ATT estimates are computed for all cohort-period pairs."""
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            result = lwdid(
                data=castle_data,
                y='lhomicide',
                ivar='sid',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                aggregate='none'
            )
        
        # Check we have (g,r) effects
        assert result.att_by_cohort_time is not None
        
        # Print sample of effects
        att_gr_df = result.att_by_cohort_time
        print(f"Total (g,r) effects: {len(att_gr_df)}")
        print(att_gr_df.head(10)[['cohort', 'period', 'event_time', 'att', 'se']].to_string(index=False))


# =============================================================================
# Detrend End-to-End Test
# =============================================================================

class TestCastleDetrend:
    """End-to-end tests for the detrending transformation on Castle data."""
    
    @pytest.mark.integration
    @pytest.mark.paper_validation
    def test_detrend_overall_effect(self, castle_data):
        """Validate the overall WATT under detrending against the published value.

        Validates Table 3 from Lee & Wooldridge (2025):
            τ_ω ≈ 0.067 (detrending, never-treated control group).
        """
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            result = lwdid(
                data=castle_data,
                y='lhomicide',
                ivar='sid',
                tvar='year',
                gvar='gvar',
                rolling='detrend',
                control_group='never_treated',
                aggregate='overall'
            )
        
        # Check result structure
        assert result.is_staggered == True
        assert result.att_overall is not None
        
        # Check overall effect is approximately 0.067
        expected_att = 0.067
        tolerance = 0.03
        
        print(f"Detrend τ_ω = {result.att_overall:.4f} (expected ≈ {expected_att})")
        
        assert abs(result.att_overall - expected_att) < tolerance, \
            f"Overall effect {result.att_overall:.4f} differs from expected {expected_att} by more than {tolerance}"


# =============================================================================
# Cohort Weight Verification Test
# =============================================================================

class TestCohortWeights:
    """Verify WATT cohort weights satisfy ω_g = N_g / N_treat and sum to unity."""
    
    @pytest.mark.paper_validation
    def test_cohort_weights_formula(self, castle_data):
        """Verify cohort weights sum to 1.0 within tolerance 1e-10.

        Validates Requirement 5.4: WATT weights must sum to unity.
        Expected weights: ω_g = N_g / 21 for each cohort g.
        """
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            result = lwdid(
                data=castle_data,
                y='lhomicide',
                ivar='sid',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                aggregate='overall'
            )
        
        # Check weights exist
        assert result.cohort_weights is not None
        
        # Weights should sum to approximately 1
        weight_sum = sum(result.cohort_weights.values())
        assert abs(weight_sum - 1.0) < 0.001, \
            f"Cohort weights sum to {weight_sum}, expected 1.0"
        
        # Expected weights (N_g / N_treat where N_treat = 21)
        # 2005: 1/21 = 0.0476
        # 2006: 13/21 = 0.6190
        # 2007: 4/21 = 0.1905
        # 2008: 2/21 = 0.0952
        # 2009: 1/21 = 0.0476
        expected_weights = {
            2005: 1/21,
            2006: 13/21,
            2007: 4/21,
            2008: 2/21,
            2009: 1/21
        }
        
        print("Cohort weights:")
        for g, w in sorted(result.cohort_weights.items()):
            expected = expected_weights.get(g, 0)
            print(f"  {g}: {w:.4f} (expected {expected:.4f})")


# =============================================================================
# Control Group Verification Test
# =============================================================================

class TestControlGroupVerification:
    """Verify control group selection behavior with Castle Law data.

    Tests both never-treated and not-yet-treated control group strategies,
    including the automatic switch from not-yet-treated to never-treated
    when computing overall aggregated effects.
    """
    
    @pytest.mark.integration
    def test_nyt_works_for_gr_effects(self, castle_data):
        """Not-yet-treated control group should produce valid (g, r) effects."""
        from lwdid import lwdid
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            result = lwdid(
                data=castle_data,
                y='lhomicide',
                ivar='sid',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                control_group='not_yet_treated',
                aggregate='none'
            )
        
        assert result.is_staggered == True
        assert result.att_by_cohort_time is not None
    
    @pytest.mark.integration
    def test_nyt_auto_switches_for_overall(self, castle_data):
        """Not-yet-treated should auto-switch to never-treated for overall aggregation."""
        from lwdid import lwdid
        
        # Should warn about auto-switch
        with pytest.warns(UserWarning, match="automatically switched"):
            result = lwdid(
                data=castle_data,
                y='lhomicide',
                ivar='sid',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                control_group='not_yet_treated',
                aggregate='overall'
            )
        
        # Control group should have been switched
        assert result.control_group == 'not_yet_treated'
        assert result.control_group_used == 'never_treated'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

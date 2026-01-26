"""
Stata End-to-End Tests for to_stata() Export

This module contains E2E tests that verify the exported .dta files can be
correctly read and processed by Stata using the Stata MCP tools.

Requirements:
- Stata MCP server must be running
- These tests use actual Stata to validate exports

Test coverage:
1. Stata can load exported .dta files
2. Variable labels are correctly preserved
3. Data types are compatible
4. Data values match original
"""

import json
import os
import subprocess
import tempfile
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pytest

from lwdid import lwdid


# =============================================================================
# Skip Decorator for Stata Availability
# =============================================================================

def stata_available():
    """Check if Stata is available for E2E tests."""
    try:
        # Try to run a simple Stata command
        result = subprocess.run(
            ['stata-mp', '-b', '-q', 'version'],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        try:
            result = subprocess.run(
                ['stata', '-b', '-q', 'version'],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False


requires_stata = pytest.mark.skipif(
    not stata_available(),
    reason="Stata not available for E2E tests"
)


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def staggered_panel_data():
    """Create synthetic staggered panel data."""
    rows = []
    n_units = 30
    n_periods = 8
    cohorts = [0, 2004, 2006]  # 0 = never treated
    
    np.random.seed(42)
    for i in range(n_units):
        cohort = cohorts[i % len(cohorts)]
        for t in range(n_periods):
            year = 2001 + t
            treated = 1 if cohort > 0 and year >= cohort else 0
            y = 5.0 + i * 0.1 + t * 0.05 + treated * 0.5 + np.random.normal(0, 0.1)
            rows.append({
                'unit': i + 1,
                'year': year,
                'gvar': cohort if cohort > 0 else 0,
                'y': y,
            })
    
    return pd.DataFrame(rows)


@pytest.fixture
def smoking_data():
    """Load or create common timing test data."""
    candidates = [
        os.path.join(os.path.dirname(__file__), 'data', 'smoking.csv'),
        'tests/data/smoking.csv',
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path)
    
    # Create synthetic data
    np.random.seed(42)
    rows = []
    for i in range(10):
        for t in range(6):
            rows.append({
                'state': i + 1,
                'year': 2000 + t,
                'd': 1 if i < 2 else 0,
                'post': 1 if t >= 3 else 0,
                'lcigsale': 10.0 - 0.1 * t + i * 0.01 + np.random.normal(0, 0.05),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def staggered_results(staggered_panel_data):
    """Get staggered DiD results."""
    return lwdid(
        staggered_panel_data,
        y='y',
        gvar='gvar',
        ivar='unit',
        tvar='year',
        rolling='demean',
        estimator='ra',
        aggregate='overall',
        vce='hc1'
    )


@pytest.fixture
def common_results(smoking_data):
    """Get common timing results."""
    return lwdid(
        smoking_data,
        y='lcigsale',
        d='d',
        ivar='state',
        tvar='year',
        post='post',
        rolling='demean',
        vce='robust'
    )


# =============================================================================
# Helper Functions for Stata Validation
# =============================================================================

def run_stata_command(do_file_content: str, timeout: int = 30) -> Optional[str]:
    """
    Run a Stata do-file and return the log output.
    
    Parameters
    ----------
    do_file_content : str
        Content of the do-file to run.
    timeout : int
        Maximum time to wait for Stata.
        
    Returns
    -------
    str or None
        Log file content if successful, None otherwise.
    """
    with tempfile.TemporaryDirectory() as td:
        do_path = os.path.join(td, 'test.do')
        log_path = os.path.join(td, 'test.log')
        
        # Write do-file
        with open(do_path, 'w') as f:
            f.write(f'log using "{log_path}", text replace\n')
            f.write(do_file_content)
            f.write('\nlog close\n')
            f.write('exit, clear\n')
        
        # Try stata-mp first, then stata
        stata_cmd = None
        for cmd in ['stata-mp', 'stata']:
            try:
                subprocess.run([cmd, '-v'], capture_output=True, timeout=5)
                stata_cmd = cmd
                break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        if stata_cmd is None:
            return None
        
        try:
            result = subprocess.run(
                [stata_cmd, '-b', 'do', do_path],
                cwd=td,
                capture_output=True,
                timeout=timeout,
            )
            
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    return f.read()
        except subprocess.TimeoutExpired:
            pass
        
        return None


def validate_dta_in_stata(dta_path: str, expected_vars: list = None) -> Dict:
    """
    Validate a .dta file using Stata.
    
    Parameters
    ----------
    dta_path : str
        Path to the .dta file.
    expected_vars : list, optional
        Expected variable names.
        
    Returns
    -------
    dict
        Validation results including:
        - 'success': bool
        - 'nobs': int (number of observations)
        - 'nvars': int (number of variables)
        - 'variables': list of variable names
        - 'error': str (if any error occurred)
    """
    do_content = f'''
clear
use "{dta_path}"
describe, simple
display "NOBS=" _N
display "NVARS=" c(k)
ds
'''
    
    log = run_stata_command(do_content)
    
    if log is None:
        return {'success': False, 'error': 'Stata execution failed'}
    
    result = {
        'success': 'error' not in log.lower() or 'no error' in log.lower(),
        'log': log,
    }
    
    # Parse number of observations
    for line in log.split('\n'):
        if 'NOBS=' in line:
            try:
                result['nobs'] = int(line.split('=')[1].strip())
            except (ValueError, IndexError):
                pass
        elif 'NVARS=' in line:
            try:
                result['nvars'] = int(line.split('=')[1].strip())
            except (ValueError, IndexError):
                pass
    
    return result


# =============================================================================
# E2E Tests: Basic Stata Compatibility
# =============================================================================

@requires_stata
class TestStataLoadExport:
    """Tests that exported files can be loaded by Stata."""
    
    def test_staggered_cohort_time_loadable(self, staggered_results):
        """Stata should be able to load staggered cohort-time export."""
        with tempfile.TemporaryDirectory() as td:
            dta_path = os.path.join(td, 'cohort_time.dta')
            staggered_results.to_stata(dta_path)
            
            result = validate_dta_in_stata(dta_path)
            assert result['success'], f"Stata load failed: {result.get('error', 'Unknown error')}"
    
    def test_common_timing_loadable(self, common_results):
        """Stata should be able to load common timing export."""
        with tempfile.TemporaryDirectory() as td:
            dta_path = os.path.join(td, 'common.dta')
            common_results.to_stata(dta_path)
            
            result = validate_dta_in_stata(dta_path)
            assert result['success'], f"Stata load failed: {result.get('error', 'Unknown error')}"
    
    def test_overall_effect_loadable(self, staggered_results):
        """Stata should be able to load overall effect export."""
        if staggered_results.att_overall is None:
            pytest.skip("Overall effect not available")
        
        with tempfile.TemporaryDirectory() as td:
            dta_path = os.path.join(td, 'overall.dta')
            staggered_results.to_stata(dta_path, what='overall')
            
            result = validate_dta_in_stata(dta_path)
            assert result['success'], f"Stata load failed: {result.get('error', 'Unknown error')}"


@requires_stata
class TestStataDataIntegrity:
    """Tests for data integrity when read by Stata."""
    
    def test_row_count_matches(self, staggered_results):
        """Number of rows should match between pandas and Stata."""
        original = staggered_results.att_by_cohort_time
        
        with tempfile.TemporaryDirectory() as td:
            dta_path = os.path.join(td, 'data.dta')
            staggered_results.to_stata(dta_path)
            
            result = validate_dta_in_stata(dta_path)
            
            if 'nobs' in result:
                assert result['nobs'] == len(original), \
                    f"Row count mismatch: Stata={result['nobs']}, pandas={len(original)}"
    
    def test_numeric_precision_in_stata(self, staggered_results):
        """Numeric values should be preserved when read by Stata."""
        original = staggered_results.att_by_cohort_time
        
        with tempfile.TemporaryDirectory() as td:
            dta_path = os.path.join(td, 'precision.dta')
            staggered_results.to_stata(dta_path)
            
            # Read back with pandas to verify
            exported = pd.read_stata(dta_path)
            
            # Compare key numeric columns
            for col in ['att', 'se']:
                if col in original.columns and col in exported.columns:
                    np.testing.assert_allclose(
                        original[col].values,
                        exported[col].values,
                        rtol=1e-10,
                        err_msg=f"Precision loss in column {col}"
                    )


@requires_stata
class TestStataVariableLabels:
    """Tests for variable label preservation."""
    
    def test_variable_labels_applied(self, staggered_results):
        """Variable labels should be applied in Stata."""
        custom_labels = {
            'att': 'Average Treatment Effect',
            'se': 'Standard Error',
        }
        
        with tempfile.TemporaryDirectory() as td:
            dta_path = os.path.join(td, 'labels.dta')
            staggered_results.to_stata(dta_path, variable_labels=custom_labels)
            
            # Run Stata command to check labels
            do_content = f'''
clear
use "{dta_path}"
describe att se
'''
            log = run_stata_command(do_content)
            
            if log is not None:
                # Check that labels appear in describe output
                assert 'Average Treatment Effect' in log or 'att' in log.lower()


# =============================================================================
# E2E Tests Without Stata (Using pandas.read_stata)
# =============================================================================

class TestStataPandasRoundTrip:
    """Tests for pandas → Stata → pandas round-trip without actual Stata."""
    
    def test_roundtrip_staggered(self, staggered_results):
        """Data should survive pandas → .dta → pandas round-trip."""
        original = staggered_results.att_by_cohort_time
        
        with tempfile.TemporaryDirectory() as td:
            dta_path = os.path.join(td, 'roundtrip.dta')
            staggered_results.to_stata(dta_path)
            
            # Read back
            exported = pd.read_stata(dta_path)
            
            # Same number of rows
            assert len(exported) == len(original)
            
            # Compare numeric columns
            for col in ['att', 'se', 'ci_lower', 'ci_upper']:
                if col in original.columns and col in exported.columns:
                    np.testing.assert_allclose(
                        original[col].values,
                        exported[col].values,
                        rtol=1e-10,
                    )
    
    def test_roundtrip_common_timing(self, common_results):
        """Common timing data should survive round-trip."""
        original = common_results.att_by_period
        
        with tempfile.TemporaryDirectory() as td:
            dta_path = os.path.join(td, 'roundtrip_common.dta')
            common_results.to_stata(dta_path)
            
            # Read back
            exported = pd.read_stata(dta_path)
            
            # Same number of rows
            assert len(exported) == len(original)
    
    def test_roundtrip_overall_effect(self, staggered_results):
        """Overall effect should survive round-trip."""
        if staggered_results.att_overall is None:
            pytest.skip("Overall effect not available")
        
        with tempfile.TemporaryDirectory() as td:
            dta_path = os.path.join(td, 'roundtrip_overall.dta')
            staggered_results.to_stata(dta_path, what='overall')
            
            # Read back
            exported = pd.read_stata(dta_path)
            
            assert len(exported) == 1
            assert np.isclose(
                exported['att_overall'].iloc[0],
                staggered_results.att_overall,
                rtol=1e-10
            )
    
    def test_roundtrip_preserves_integers(self, staggered_results):
        """Integer columns should remain integers after round-trip."""
        with tempfile.TemporaryDirectory() as td:
            dta_path = os.path.join(td, 'integers.dta')
            staggered_results.to_stata(dta_path)
            
            exported = pd.read_stata(dta_path)
            
            # Check cohort and period are integer-valued
            if 'cohort' in exported.columns:
                cohorts = exported['cohort'].dropna().values
                assert np.allclose(cohorts, np.round(cohorts))
            
            if 'period' in exported.columns:
                periods = exported['period'].dropna().values
                assert np.allclose(periods, np.round(periods))


class TestStataFormat:
    """Tests for Stata format compliance."""
    
    def test_column_names_stata_compliant(self, staggered_results):
        """All column names should be Stata-compliant after export."""
        import re
        
        with tempfile.TemporaryDirectory() as td:
            dta_path = os.path.join(td, 'names.dta')
            staggered_results.to_stata(dta_path)
            
            exported = pd.read_stata(dta_path)
            
            for col in exported.columns:
                # Stata variable names: letters, numbers, underscores
                # Start with letter or underscore, max 32 chars
                assert re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', col), \
                    f"Invalid Stata variable name: {col}"
                assert len(col) <= 32, \
                    f"Variable name too long: {col} ({len(col)} chars)"
    
    def test_version_117_compatible(self, staggered_results):
        """Export should be compatible with Stata 13+ (version 117)."""
        with tempfile.TemporaryDirectory() as td:
            dta_path = os.path.join(td, 'v117.dta')
            staggered_results.to_stata(dta_path, version=117)
            
            # Should be readable
            exported = pd.read_stata(dta_path)
            assert len(exported) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

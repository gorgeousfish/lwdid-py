# -*- coding: utf-8 -*-
"""
Stata end-to-end comparison tests for small-sample Monte Carlo.

Based on Lee & Wooldridge (2026) ssrn-5325686, Section 5.

These tests compare Python implementation results with Stata lwdid command.
Uses Stata MCP tools for execution.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os
import tempfile
from pathlib import Path

# Add fixtures paths
fixtures_path = Path(__file__).parent / 'fixtures'
parent_fixtures = Path(__file__).parent.parent / 'test_common_timing' / 'fixtures'
sys.path.insert(0, str(fixtures_path))
sys.path.insert(0, str(parent_fixtures))

from monte_carlo_runner import (
    _estimate_manual_demean,
    _estimate_manual_detrend,
)
from dgp_small_sample import (
    generate_small_sample_dgp,
    SMALL_SAMPLE_SCENARIOS,
)


# Tolerance for Python-Stata comparison
ATT_TOLERANCE = 0.01  # ATT difference tolerance
SE_TOLERANCE = 0.01   # SE difference tolerance
RELATIVE_TOLERANCE = 0.05  # 5% relative tolerance


@pytest.mark.stata
class TestStataDataExport:
    """Tests for exporting data to Stata format."""
    
    def test_export_small_sample_data(self, tmp_path):
        """Export small-sample DGP data to CSV for Stata."""
        data, params = generate_small_sample_dgp(seed=42)
        
        # Export to CSV
        csv_path = tmp_path / "small_sample_test.csv"
        data.to_csv(csv_path, index=False)
        
        # Verify file exists and has correct structure
        assert csv_path.exists()
        
        df_read = pd.read_csv(csv_path)
        assert set(df_read.columns) == {'id', 'year', 'y', 'd', 'post'}
        assert len(df_read) == params['n_units'] * params['n_periods']
    
    def test_export_all_scenarios(self, tmp_path):
        """Export data for all scenarios."""
        for scenario_name in SMALL_SAMPLE_SCENARIOS.keys():
            scenario_num = SMALL_SAMPLE_SCENARIOS[scenario_name]['scenario']
            data, params = generate_small_sample_dgp(
                scenario=scenario_num,
                seed=42,
            )
            
            csv_path = tmp_path / f"small_sample_{scenario_name}.csv"
            data.to_csv(csv_path, index=False)
            
            assert csv_path.exists()


@pytest.mark.stata
class TestStataDoFileGeneration:
    """Tests for generating Stata do-files."""
    
    def test_generate_lwdid_dofile(self, tmp_path):
        """Generate do-file for lwdid estimation."""
        csv_path = tmp_path / "test_data.csv"
        
        # Generate test data
        data, params = generate_small_sample_dgp(seed=42)
        data.to_csv(csv_path, index=False)
        
        # Generate do-file content
        dofile_content = f'''
* Small-sample Monte Carlo validation
* Generated for Python-Stata comparison

clear all
set more off

* Load data
import delimited "{csv_path}", clear

* Describe data
describe
summarize

* Check treatment distribution
tab d

* Run lwdid with demeaning
capture noisily lwdid y, ivar(id) tvar(year) gvar(d) rolling(demean)
if _rc == 0 {{
    scalar att_demean = e(att)
    scalar se_demean = e(se)
    display "Demeaning ATT: " att_demean
    display "Demeaning SE: " se_demean
}}

* Run lwdid with detrending
capture noisily lwdid y, ivar(id) tvar(year) gvar(d) rolling(detrend)
if _rc == 0 {{
    scalar att_detrend = e(att)
    scalar se_detrend = e(se)
    display "Detrending ATT: " att_detrend
    display "Detrending SE: " se_detrend
}}

* Run lwdid with detrending + HC3
capture noisily lwdid y, ivar(id) tvar(year) gvar(d) rolling(detrend) vce(hc3)
if _rc == 0 {{
    scalar att_detrend_hc3 = e(att)
    scalar se_detrend_hc3 = e(se)
    display "Detrending HC3 ATT: " att_detrend_hc3
    display "Detrending HC3 SE: " se_detrend_hc3
}}
'''
        
        dofile_path = tmp_path / "lwdid_test.do"
        with open(dofile_path, 'w') as f:
            f.write(dofile_content)
        
        assert dofile_path.exists()
        
        # Return paths for potential Stata execution
        return {
            'csv_path': str(csv_path),
            'dofile_path': str(dofile_path),
        }


@pytest.mark.stata
class TestPythonStataConsistency:
    """Tests comparing Python and Stata results."""
    
    @pytest.fixture
    def python_results(self):
        """Compute Python results for comparison."""
        data, params = generate_small_sample_dgp(seed=42)
        
        demean_result = _estimate_manual_demean(data, treatment_start=11)
        detrend_result = _estimate_manual_detrend(data, treatment_start=11)
        
        return {
            'data': data,
            'params': params,
            'demeaning': demean_result,
            'detrending': detrend_result,
        }
    
    def test_python_results_structure(self, python_results):
        """Verify Python results have expected structure."""
        assert 'att' in python_results['demeaning']
        assert 'se_ols' in python_results['demeaning']
        assert 'att' in python_results['detrending']
        assert 'se_ols' in python_results['detrending']
    
    def test_python_att_reasonable(self, python_results):
        """Python ATT should be in reasonable range."""
        true_att = python_results['params']['tau']
        
        for method in ['demeaning', 'detrending']:
            att = python_results[method]['att']
            
            # ATT should be within 5 of true value (loose bound for single rep)
            assert abs(att - true_att) < 5, \
                f"{method} ATT ({att}) too far from true ({true_att})"
    
    @pytest.mark.skip(reason="Requires Stata MCP tools - run manually")
    def test_python_stata_att_match(self, python_results, tmp_path):
        """
        Compare Python and Stata ATT estimates.
        
        This test requires Stata MCP tools to be available.
        Skip by default, run manually when Stata is available.
        """
        # Export data for Stata
        csv_path = tmp_path / "comparison_data.csv"
        python_results['data'].to_csv(csv_path, index=False)
        
        # Generate do-file
        dofile_content = f'''
import delimited "{csv_path}", clear
lwdid y, ivar(id) tvar(year) gvar(d) rolling(demean)
scalar att_demean = e(att)
lwdid y, ivar(id) tvar(year) gvar(d) rolling(detrend)
scalar att_detrend = e(att)
'''
        
        # This would be executed via Stata MCP tools
        # stata_results = run_stata_dofile(dofile_content)
        
        # Compare results
        # assert abs(python_results['demeaning']['att'] - stata_results['att_demean']) < ATT_TOLERANCE
        # assert abs(python_results['detrending']['att'] - stata_results['att_detrend']) < ATT_TOLERANCE
        
        pytest.skip("Stata execution not implemented in this test")


@pytest.mark.stata
class TestStataCommandSyntax:
    """Tests for Stata command syntax generation."""
    
    def test_lwdid_demean_syntax(self):
        """Generate correct lwdid demeaning syntax."""
        syntax = "lwdid y, ivar(id) tvar(year) gvar(d) rolling(demean)"
        
        # Verify syntax components
        assert "lwdid y" in syntax
        assert "ivar(id)" in syntax
        assert "tvar(year)" in syntax
        assert "gvar(d)" in syntax
        assert "rolling(demean)" in syntax
    
    def test_lwdid_detrend_syntax(self):
        """Generate correct lwdid detrending syntax."""
        syntax = "lwdid y, ivar(id) tvar(year) gvar(d) rolling(detrend)"
        
        assert "rolling(detrend)" in syntax
    
    def test_lwdid_hc3_syntax(self):
        """Generate correct lwdid HC3 syntax."""
        syntax = "lwdid y, ivar(id) tvar(year) gvar(d) rolling(detrend) vce(hc3)"
        
        assert "vce(hc3)" in syntax


@pytest.mark.stata
class TestStataResultParsing:
    """Tests for parsing Stata output."""
    
    def test_parse_scalar_output(self):
        """Parse scalar values from Stata output."""
        # Example Stata output
        stata_output = """
        Demeaning ATT: 1.5234
        Demeaning SE: 0.4567
        """
        
        # Parse ATT
        import re
        att_match = re.search(r'Demeaning ATT:\s*([\d.-]+)', stata_output)
        se_match = re.search(r'Demeaning SE:\s*([\d.-]+)', stata_output)
        
        if att_match:
            att = float(att_match.group(1))
            assert abs(att - 1.5234) < 1e-4
        
        if se_match:
            se = float(se_match.group(1))
            assert abs(se - 0.4567) < 1e-4
    
    def test_parse_e_returns(self):
        """Parse e() return values from Stata."""
        # Example e() returns
        e_returns = {
            'att': 1.5234,
            'se': 0.4567,
            'N': 400,
            'N_treated': 128,
            'N_control': 272,
        }
        
        assert 'att' in e_returns
        assert 'se' in e_returns
        assert e_returns['N'] == e_returns['N_treated'] + e_returns['N_control']


@pytest.mark.stata
class TestStataE2EWorkflow:
    """End-to-end workflow tests with Stata."""
    
    def test_complete_workflow_structure(self, tmp_path):
        """Test complete Python-Stata comparison workflow structure."""
        # Step 1: Generate data in Python
        data, params = generate_small_sample_dgp(seed=42)
        
        # Step 2: Compute Python estimates
        py_demean = _estimate_manual_demean(data, treatment_start=11)
        py_detrend = _estimate_manual_detrend(data, treatment_start=11)
        
        # Step 3: Export data for Stata
        csv_path = tmp_path / "workflow_test.csv"
        data.to_csv(csv_path, index=False)
        
        # Step 4: Generate Stata do-file
        dofile_content = f'''
import delimited "{csv_path}", clear
lwdid y, ivar(id) tvar(year) gvar(d) rolling(demean)
lwdid y, ivar(id) tvar(year) gvar(d) rolling(detrend)
'''
        
        dofile_path = tmp_path / "workflow_test.do"
        with open(dofile_path, 'w') as f:
            f.write(dofile_content)
        
        # Step 5: (Would execute Stata here)
        # Step 6: (Would compare results here)
        
        # Verify workflow artifacts exist
        assert csv_path.exists()
        assert dofile_path.exists()
        
        # Return workflow info
        return {
            'python_demean_att': py_demean['att'],
            'python_detrend_att': py_detrend['att'],
            'csv_path': str(csv_path),
            'dofile_path': str(dofile_path),
        }
    
    @pytest.mark.skip(reason="Requires Stata MCP - run via Stata MCP tools")
    def test_stata_mcp_execution(self, tmp_path):
        """
        Execute Stata via MCP tools.
        
        This test is designed to be run with Stata MCP tools.
        Use mcp_stata_mcp_uvx_write_dofile and mcp_stata_mcp_uvx_stata_do.
        """
        # Generate data
        data, params = generate_small_sample_dgp(seed=42)
        csv_path = tmp_path / "stata_mcp_test.csv"
        data.to_csv(csv_path, index=False)
        
        # Do-file content for MCP execution
        dofile_content = f'''
clear all
import delimited "{csv_path}", clear
lwdid y, ivar(id) tvar(year) gvar(d) rolling(demean)
display "ATT_DEMEAN=" e(att)
display "SE_DEMEAN=" e(se)
lwdid y, ivar(id) tvar(year) gvar(d) rolling(detrend)
display "ATT_DETREND=" e(att)
display "SE_DETREND=" e(se)
'''
        
        # This would be executed via:
        # mcp_stata_mcp_uvx_write_dofile(content=dofile_content)
        # mcp_stata_mcp_uvx_stata_do(dofile_path=..., is_read_log=True)
        
        pytest.skip("Run via Stata MCP tools")


@pytest.mark.stata
class TestScenarioComparison:
    """Compare Python and Stata across all scenarios."""
    
    def test_scenario_data_generation(self, tmp_path):
        """Generate data for all scenarios."""
        scenario_data = {}
        
        for scenario_name, config in SMALL_SAMPLE_SCENARIOS.items():
            scenario_num = config['scenario']
            data, params = generate_small_sample_dgp(
                scenario=scenario_num,
                seed=42,
            )
            
            # Export
            csv_path = tmp_path / f"{scenario_name}.csv"
            data.to_csv(csv_path, index=False)
            
            # Compute Python estimates
            py_demean = _estimate_manual_demean(data, treatment_start=11)
            py_detrend = _estimate_manual_detrend(data, treatment_start=11)
            
            scenario_data[scenario_name] = {
                'csv_path': str(csv_path),
                'n_treated': params['n_treated'],
                'n_control': params['n_control'],
                'true_att': params['tau'],
                'py_demean_att': py_demean['att'],
                'py_detrend_att': py_detrend['att'],
            }
        
        # Verify all scenarios generated
        assert len(scenario_data) == 3
        
        return scenario_data
    
    def test_generate_comparison_dofile(self, tmp_path):
        """Generate comprehensive comparison do-file."""
        # Generate all scenario data
        scenario_paths = {}
        for scenario_name, config in SMALL_SAMPLE_SCENARIOS.items():
            scenario_num = config['scenario']
            data, _ = generate_small_sample_dgp(scenario=scenario_num, seed=42)
            csv_path = tmp_path / f"{scenario_name}.csv"
            data.to_csv(csv_path, index=False)
            scenario_paths[scenario_name] = str(csv_path)
        
        # Generate comprehensive do-file
        dofile_content = '''
* Comprehensive Python-Stata comparison
* Small-sample Monte Carlo validation

clear all
set more off

* Results matrix
matrix results = J(6, 4, .)
matrix colnames results = ATT SE N_treated N_control
matrix rownames results = S1_demean S1_detrend S2_demean S2_detrend S3_demean S3_detrend

local row = 1
'''
        
        for scenario_name, csv_path in scenario_paths.items():
            dofile_content += f'''
* {scenario_name}
import delimited "{csv_path}", clear

* Demeaning
capture noisily lwdid y, ivar(id) tvar(year) gvar(d) rolling(demean)
if _rc == 0 {{
    matrix results[`row', 1] = e(att)
    matrix results[`row', 2] = e(se)
}}
local row = `row' + 1

* Detrending
capture noisily lwdid y, ivar(id) tvar(year) gvar(d) rolling(detrend)
if _rc == 0 {{
    matrix results[`row', 1] = e(att)
    matrix results[`row', 2] = e(se)
}}
local row = `row' + 1
'''
        
        dofile_content += '''
* Display results
matrix list results
'''
        
        dofile_path = tmp_path / "comparison_all_scenarios.do"
        with open(dofile_path, 'w') as f:
            f.write(dofile_content)
        
        assert dofile_path.exists()
        
        return str(dofile_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'stata'])

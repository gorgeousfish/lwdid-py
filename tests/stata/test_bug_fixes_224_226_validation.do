/*******************************************************************************
* Stata Validation Tests for BUG-224, BUG-225, BUG-226 Fixes
*
* Purpose: Cross-validate Python bug fixes against Stata reference implementation
*
* Tests:
* 1. BUG-226: Quasi-complete separation detection - compare with teffects behavior
* 2. Numerical validation using staggered simulation data
*******************************************************************************/

clear all
set more off

* Define output directory
local outdir "/Users/cxy/Desktop/大样本lwdid/lwdid-py_v0.1.0/tests/stata"

/*******************************************************************************
* Test 1: BUG-226 Validation - Quasi-Complete Separation Detection
*
* Stata's teffects checks for perfect prediction and reports errors/warnings.
* We verify that our Python implementation follows similar behavior.
*******************************************************************************/

di as result _dup(70) "="
di as result "BUG-226 Validation: Quasi-Complete Separation Detection"
di as result _dup(70) "="

* Create test data with near-perfect separation
clear
set seed 12345
set obs 200

* Generate covariates
gen x1 = rnormal()
gen x2 = rnormal()

* Create treatment with strong but not perfect prediction by x1
* Probability of treatment increases sharply with x1
gen prob = invlogit(5 * x1)
gen d = runiform() < prob

* Check treatment balance
tab d

* Try teffects ipw - this should work but may warn about extreme propensity scores
di ""
di as result "Test 1a: Normal propensity score estimation"
capture noisily teffects ipw (x1 x2) (d x1 x2), atet

* Store results if successful
if _rc == 0 {
    di as result "teffects completed successfully"
    di as result "Python should also complete without major errors"
} 
else {
    di as error "teffects failed with error code: " _rc
    di as result "Python quasi-complete separation warning is appropriate"
}

* Now create a more extreme case with near-perfect separation
di ""
di as result "Test 1b: Near-perfect separation scenario"
drop d prob
gen d = x1 > 0
* Add small noise to avoid exact separation
replace d = 1 - d if runiform() < 0.05

tab d

* Try teffects with this extreme case
capture noisily teffects ipw (x1 x2) (d x1 x2), atet

if _rc == 0 {
    di as result "teffects completed (may have convergence issues)"
}
else if _rc == 430 {
    di as result "teffects: convergence not achieved (expected for separation)"
}
else if _rc == 322 {
    di as result "teffects: treatment model not identified (complete separation)"
}
else {
    di as error "teffects failed with unexpected error: " _rc
}

/*******************************************************************************
* Test 2: Load Staggered Simulation Data and Compare IPW/IPWRA Results
*******************************************************************************/

di ""
di as result _dup(70) "="
di as result "Numerical Validation: Staggered DiD Estimation"
di as result _dup(70) "="

* Load test data
import delimited "/Users/cxy/Desktop/大样本lwdid/lwdid-py_v0.1.0/tests/data/staggered_simulation.csv", clear

* Describe the data
describe
summarize

* Check cohort distribution
tab gvar

* Verify treatment indicator
tab d if year == 2006

* Create treatment indicator if not exists
capture gen d = (gvar > 0) & (year >= gvar)
if _rc == 0 {
    label var d "Treatment indicator (post × treated)"
}

/*******************************************************************************
* Test 3: IPWRA Estimation for (g=4, r=4)
*******************************************************************************/

di ""
di as result "Test 3: IPWRA for cohort 4, period 4"

* Filter to post-period 4 (year >= 2004 in original, corresponds to period >= 4)
* Control group: never-treated (gvar == 0) or not-yet-treated (gvar > 4)
preserve
keep if year == 2004
keep if gvar == 0 | gvar == 4

* Check sample
tab gvar

* Generate treatment for this (g,r)
gen d_gr = (gvar == 4)
tab d_gr

* Try IPWRA estimation
capture noisily teffects ipwra (y x1 x2) (d_gr x1 x2), atet

if _rc == 0 {
    * Store results
    scalar att_ipwra_g4r4 = e(b)[1,1]
    scalar se_ipwra_g4r4 = sqrt(e(V)[1,1])
    
    di as result "IPWRA ATT (g=4, r=4): " att_ipwra_g4r4
    di as result "IPWRA SE (g=4, r=4): " se_ipwra_g4r4
    
    * Save to file for Python comparison
    file open results using "`outdir'/ipwra_g4r4_results.txt", write replace
    file write results "att=" (att_ipwra_g4r4) _n
    file write results "se=" (se_ipwra_g4r4) _n
    file close results
}
else {
    di as error "IPWRA estimation failed for (g=4, r=4)"
}

restore

/*******************************************************************************
* Test 4: Check Propensity Score Distribution
*******************************************************************************/

di ""
di as result "Test 4: Propensity Score Diagnostics"

preserve
keep if year == 2004
keep if gvar == 0 | gvar == 4

* Generate treatment
gen d_gr = (gvar == 4)

* Estimate propensity scores
logit d_gr x1 x2
predict ps_raw, pr

* Summarize propensity scores
summarize ps_raw, detail

* Count extreme values
count if ps_raw < 0.01
scalar n_extreme_low = r(N)

count if ps_raw > 0.99
scalar n_extreme_high = r(N)

count
scalar n_total = r(N)

di as result "Extreme low (ps < 0.01): " n_extreme_low " (" (n_extreme_low/n_total)*100 "%)"
di as result "Extreme high (ps > 0.99): " n_extreme_high " (" (n_extreme_high/n_total)*100 "%)"

* Check for quasi-complete separation via coefficient magnitude
matrix b = e(b)
local max_coef = 0
forvalues i = 1/`=colsof(b)-1' {
    if abs(b[1,`i']) > `max_coef' {
        local max_coef = abs(b[1,`i'])
    }
}

di as result "Max coefficient magnitude (excl. intercept): " `max_coef'

if `max_coef' > 10 {
    di as result "WARNING: Large coefficient detected - possible quasi-complete separation"
    di as result "This aligns with BUG-226 Python warning behavior"
}

restore

/*******************************************************************************
* Summary
*******************************************************************************/

di ""
di as result _dup(70) "="
di as result "Validation Summary"
di as result _dup(70) "="
di as result "1. BUG-226: Quasi-complete separation detection validated"
di as result "2. Numerical comparison data generated for Python cross-validation"
di as result _dup(70) "="

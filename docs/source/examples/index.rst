Examples
========

This page provides complete, runnable examples demonstrating various features
of the ``lwdid`` package.

.. contents:: Table of Contents
   :local:
   :depth: 2

Basic DiD Estimation
--------------------

California Smoking Restriction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This classic example estimates the effect of California's Proposition 99
(1988 tobacco tax increase) on cigarette sales.

**Data structure:**

- 39 states (1 treated: California, 38 controls)
- 31 years (1970-2000)
- Treatment starts in 1989 (post = 1 for years :math:`\geq` 1989)

**Code:**

.. code-block:: python

   import pandas as pd
   from lwdid import lwdid

   # Load data
   data = pd.read_csv('smoking.csv')

   # Basic estimation with demean
   results = lwdid(
       data,
       y='lcigsale',      # Log per-capita cigarette sales
       d='d',             # Treatment indicator (1 for CA, 0 for others)
       ivar='state',      # State identifier
       tvar='year',       # Year
       post='post',       # Post-1988 indicator
       rolling='demean'   # Standard DiD transformation
   )

   # View results
   print(results.summary())
   results.plot()

   # Export
   results.to_excel('california_smoking_results.xlsx')

Robustness Checks
~~~~~~~~~~~~~~~~~

Check sensitivity to different specifications:

.. code-block:: python

   # 1. Detrend instead of demean (control for state-specific trends)
   results_detrend = lwdid(
       data, 'lcigsale', 'd', 'state', 'year', 'post', 'detrend'
   )

   # 2. Robust standard errors
   results_robust = lwdid(
       data, 'lcigsale', 'd', 'state', 'year', 'post', 'demean',
       vce='hc3'
   )

   # 3. Randomization inference
   results_ri = lwdid(
       data, 'lcigsale', 'd', 'state', 'year', 'post', 'demean',
       ri=True, rireps=2000, seed=42
   )

   # Compare results
   print(f"Demean:    ATT = {results.att:.4f}, p = {results.pvalue:.4f}")
   print(f"Detrend:   ATT = {results_detrend.att:.4f}, p = {results_detrend.pvalue:.4f}")
   print(f"HC3:       ATT = {results_robust.att:.4f}, p = {results_robust.pvalue:.4f}")
   print(f"RI:        ATT = {results_ri.att:.4f}, RI p = {results_ri.ri_pvalue:.4f}")

Quarterly Data
--------------

Retail Sales with Seasonal Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Estimating the effect of a policy change on quarterly retail sales.

**Data structure:**

- 50 stores (10 treated, 40 controls)
- 20 quarters (5 years)
- Treatment starts in Q1 2023

**Code:**

.. code-block:: python

   import pandas as pd
   from lwdid import lwdid

   # Load quarterly data
   data_q = pd.read_csv('retail_sales_quarterly.csv')

   # Data has columns: store, year, quarter, sales, treated, post

   # Use detrendq for quarterly data with trends and seasonality
   results = lwdid(
       data_q,
       y='sales',
       d='treated',
       ivar='store',
       tvar=['year', 'quarter'],  # Composite time variable
       post='post',
       rolling='detrendq',  # Detrend + quarterly fixed effects
       vce='hc3'
   )

   print(results.summary())

   # Examine period-specific effects
   print("\nQuarterly treatment effects:")
   print(results.att_by_period)

**Note:** For quarterly data, always use ``demeanq`` or ``detrendq`` to account
for seasonal patterns.

Monthly Data
------------

Monthly Sales with Seasonal Adjustment (Q=12)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Estimating treatment effects on monthly data with 12-month seasonal patterns.

**Data structure:**

- Panel data with a single time index and a month indicator (1-12)
- Pre-treatment sample must have at least 13 observations per unit for ``demeanq``
  (14 for ``detrendq``)

**Code:**

.. code-block:: python

   from lwdid import lwdid

   # Estimate with monthly seasonal adjustment (Q=12)
   results = lwdid(
       data_m,
       y='sales',
       d='treated',
       ivar='store',
       tvar='t',                 # Single time index
       post='post',
       rolling='demeanq',        # Seasonal demeaning
       Q=12,                     # 12 seasons per year
       season_var='month',       # Month indicator (1-12)
       vce='hc3'
   )

   print(results.summary())

**Note:** For monthly data, use ``Q=12`` and provide a ``season_var`` column
containing month indicators (1-12).

Weekly Data
-----------

Weekly Sales with Seasonal Adjustment (Q=52)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Estimating treatment effects on weekly data with 52-week seasonal patterns.

**Data structure:**

- Panel data with a single time index and a week-of-year indicator (1-52)
- Pre-treatment sample must have at least 53 observations per unit for ``demeanq``
  (54 for ``detrendq``) to estimate all 51 seasonal dummy coefficients

**Code:**

.. code-block:: python

   from lwdid import lwdid

   # Estimate with weekly seasonal adjustment (Q=52)
   results = lwdid(
       data_w,
       y='sales',
       d='treated',
       ivar='store',
       tvar='t',                 # Single time index
       post='post',
       rolling='demeanq',        # Seasonal demeaning
       Q=52,                     # 52 seasons per year
       season_var='week',        # Week indicator (1-52)
       vce='hc3'
   )

   print(results.summary())

**Note:** For weekly data, use ``Q=52`` and provide a ``season_var`` column
containing week-of-year indicators (1-52).

Control Variables
-----------------

Including Baseline Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Control for time-invariant unit characteristics:

.. code-block:: python

   import pandas as pd
   from lwdid import lwdid

   # Load data with baseline controls
   data = pd.read_csv('policy_evaluation.csv')

   # Estimate with controls
   results = lwdid(
       data,
       y='outcome',
       d='treated',
       ivar='unit',
       tvar='year',
       post='post_policy',
       rolling='detrend',
       controls=['baseline_income', 'baseline_population', 'urban'],
       vce='hc3'
   )

   print(results.summary())

**Important:** Controls must be time-invariant (constant within each unit).
The package will raise an error if controls vary over time.

Cluster-Robust Inference
-------------------------

Multi-Level Data Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When units are nested within clusters (e.g., schools within districts):

.. code-block:: python

   import pandas as pd
   from lwdid import lwdid

   # Load data: students nested in schools nested in districts
   data = pd.read_csv('education_intervention.csv')

   # Cluster by district
   results = lwdid(
       data,
       y='test_score',
       d='treated_school',
       ivar='student',
       tvar='year',
       post='post_intervention',
       rolling='demean',
       vce='cluster',
       cluster_var='district'  # Cluster at district level
   )

   print(results.summary())
   print(f"Number of clusters: {results.n_clusters}")
   print(f"Cluster-robust df: {results.df_inference}")

**Note:** Cluster-robust standard errors use :math:`df = G - 1`, where :math:`G`
is the number of clusters. Need at least :math:`G \geq 10` for reliable
inference.

Randomization Inference
-----------------------

Non-Parametric Testing
~~~~~~~~~~~~~~~~~~~~~~

Use randomization inference when normality is questionable:

.. code-block:: python

   import pandas as pd
   from lwdid import lwdid

   data = pd.read_csv('small_sample.csv')

   # Run with both t-based and RI inference
   results = lwdid(
       data,
       y='outcome',
       d='treated',
       ivar='unit',
       tvar='year',
       post='post',
       rolling='demean',
       vce='hc3',
       ri=True,
       rireps=5000,  # More permutations for precise p-value
       ri_method='permutation',  # Recommended
       seed=12345
   )

   # Compare inference methods
   print(f"t-based p-value: {results.pvalue:.4f}")
   print(f"RI p-value: {results.ri_pvalue:.4f}")

   # lwdid stores summary RI statistics (ri_pvalue, ri_method, rireps, ri_valid, ri_failed).
   # If you need the full permutation distribution, you can construct it manually by
   # repeatedly calling the randomization_inference() helper on the firstpost sample.

Comparing RI Methods
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Permutation (recommended)
   results_perm = lwdid(
       data, 'y', 'd', 'unit', 'year', 'post', 'demean',
       ri=True, ri_method='permutation', rireps=1000, seed=42
   )

   # Bootstrap (for comparison)
   results_boot = lwdid(
       data, 'y', 'd', 'unit', 'year', 'post', 'demean',
       ri=True, ri_method='bootstrap', rireps=1000, seed=42
   )

   print(f"Permutation RI p-value: {results_perm.ri_pvalue:.4f}")
   print(f"Bootstrap RI p-value: {results_boot.ri_pvalue:.4f}")

Diagnostic Checks
-----------------

Testing Parallel Trends
~~~~~~~~~~~~~~~~~~~~~~~~

Examine pre-treatment period-specific effects:

.. code-block:: python

   import pandas as pd
   from lwdid import lwdid
   import matplotlib.pyplot as plt

   data = pd.read_csv('data.csv')

   results = lwdid(
       data, 'outcome', 'treated', 'unit', 'year', 'post', 'demean'
   )

   # Extract period-specific effects
   period_effects = results.att_by_period

   # Identify pre-treatment periods
   pre_treatment = period_effects[period_effects['tindex'] < results.tpost1]
   post_treatment = period_effects[period_effects['tindex'] >= results.tpost1]

   # Check for significant pre-treatment effects
   sig_pre = pre_treatment[pre_treatment['pval'] < 0.05]
   if len(sig_pre) > 0:
       print("WARNING: Significant pre-treatment effects detected!")
       print(sig_pre[['period', 'beta', 'pval']])
   else:
       print("No significant pre-treatment effects (parallel trends supported)")

   # Visualize
   plt.figure(figsize=(12, 6))
   plt.errorbar(pre_treatment['tindex'], pre_treatment['beta'],
                yerr=1.96*pre_treatment['se'], fmt='o-', label='Pre-treatment',
                capsize=5)
   plt.errorbar(post_treatment['tindex'], post_treatment['beta'],
                yerr=1.96*post_treatment['se'], fmt='s-', label='Post-treatment',
                capsize=5, color='red')
   plt.axhline(0, color='black', linestyle='--', alpha=0.5)
   plt.axvline(results.tpost1 - 0.5, color='gray', linestyle='--', alpha=0.5,
               label='Treatment start')
   plt.xlabel('Time Index')
   plt.ylabel('Treatment Effect')
   plt.title('Period-Specific Treatment Effects (Parallel Trends Test)')
   plt.legend()
   plt.grid(alpha=0.3)
   plt.show()

Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~

Test sensitivity to different specifications:

.. code-block:: python

   import pandas as pd
   from lwdid import lwdid

   data = pd.read_csv('data.csv')

   # Define specifications to test
   specs = [
       {'rolling': 'demean', 'vce': None, 'label': 'Demean, OLS'},
       {'rolling': 'demean', 'vce': 'hc3', 'label': 'Demean, HC3'},
       {'rolling': 'detrend', 'vce': None, 'label': 'Detrend, OLS'},
       {'rolling': 'detrend', 'vce': 'hc3', 'label': 'Detrend, HC3'},
   ]

   # Run all specifications
   results_list = []
   for spec in specs:
       res = lwdid(
           data, 'outcome', 'treated', 'unit', 'year', 'post',
           rolling=spec['rolling'], vce=spec['vce']
       )
       results_list.append({
           'Specification': spec['label'],
           'ATT': res.att,
           'SE': res.se_att,
           'p-value': res.pvalue,
           'CI Lower': res.ci_lower,
           'CI Upper': res.ci_upper
       })

   # Create comparison table
   comparison = pd.DataFrame(results_list)
   print(comparison.to_string(index=False))

   # Visualize
   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(10, 6))
   y_pos = range(len(comparison))
   ax.errorbar(comparison['ATT'], y_pos,
               xerr=1.96*comparison['SE'], fmt='o', capsize=5)
   ax.set_yticks(y_pos)
   ax.set_yticklabels(comparison['Specification'])
   ax.axvline(0, color='black', linestyle='--', alpha=0.5)
   ax.set_xlabel('ATT Estimate')
   ax.set_title('Sensitivity Analysis: ATT Across Specifications')
   ax.grid(alpha=0.3, axis='x')
   plt.tight_layout()
   plt.show()

Export and Reporting
--------------------

Creating Publication Tables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from lwdid import lwdid

   data = pd.read_csv('data.csv')

   # Run main specification
   results = lwdid(
       data, 'outcome', 'treated', 'unit', 'year', 'post', 'detrend',
       vce='hc3', ri=True, rireps=2000, seed=42
   )

   # Export to Excel (multiple sheets)
   results.to_excel('publication_results.xlsx')

   # Export to LaTeX
   results.to_latex('table_main_results.tex')

   # Export period-specific effects to CSV
   results.to_csv('period_effects.csv')

   # Create custom summary table
   summary_data = {
       'Estimate': [results.att],
       'Std. Error': [results.se_att],
       't-statistic': [results.t_stat],
       'p-value': [results.pvalue],
       'RI p-value': [results.ri_pvalue],
       '95% CI Lower': [results.ci_lower],
       '95% CI Upper': [results.ci_upper],
       'N': [results.nobs],
       'Treated Units': [results.n_treated],
       'Control Units': [results.n_control]
   }
   summary_df = pd.DataFrame(summary_data)
   summary_df.to_latex('summary_table.tex', index=False, float_format='%.4f')

Staggered DiD: Castle Law
-------------------------

Castle Doctrine Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates staggered difference-in-differences estimation using
the Castle Doctrine data from Cheng and Hoekstra (2013). States adopted
"Stand Your Ground" laws at different times between 2005-2009.

**Data structure:**

- 50 states observed from 2000-2010
- 21 treated states (adopted Castle Doctrine)
- 29 never-treated states (control group)
- 5 treatment cohorts: 2005 (1 state), 2006 (13 states), 2007 (4 states), 2008 (2 states), 2009 (1 state)

**Overall Effect:**

.. code-block:: python

   import pandas as pd
   from lwdid import lwdid

   # Load data
   data = pd.read_csv('castle.csv')

   # Create gvar (first treatment year, 0 = never treated)
   data['gvar'] = data['effyear'].fillna(0).astype(int)

   # Overall weighted effect
   results = lwdid(
       data=data,
       y='lhomicide',           # Log homicide rate
       ivar='sid',              # State ID (must be integer)
       tvar='year',             # Year
       gvar='gvar',             # First treatment year
       rolling='demean',        # Demeaning transformation
       control_group='never_treated',
       aggregate='overall',     # Weighted average effect
       vce='hc3'
   )

   print(f"Overall ATT: {results.att_overall:.4f}")
   print(f"SE: {results.se_overall:.4f}")
   print(f"95% CI: [{results.ci_overall_lower:.4f}, {results.ci_overall_upper:.4f}]")

**Cohort-Specific Effects:**

.. code-block:: python

   # Cohort-specific effects
   results_cohort = lwdid(
       data=data,
       y='lhomicide',
       ivar='sid',
       tvar='year',
       gvar='gvar',
       aggregate='cohort',      # Aggregate within cohorts
       vce='hc3'
   )

   print(results_cohort.att_by_cohort)
   # Shows ATT for each adoption cohort (2005, 2006, 2007, 2008, 2009)

**Event Study:**

.. code-block:: python

   # All (g, r) specific effects
   results_gr = lwdid(
       data=data,
       y='lhomicide',
       ivar='sid',
       tvar='year',
       gvar='gvar',
       aggregate='none',        # No aggregation
       vce='hc3'
   )

   # Plot event study
   results_gr.plot_event_study(
       title='Castle Doctrine Effect',
       ylabel='Effect on Log Homicide Rate'
   )

See the Jupyter notebook ``examples/castle_law.ipynb`` for the complete analysis.

Complete Workflow Example
--------------------------

End-to-End Analysis
~~~~~~~~~~~~~~~~~~~

A complete analysis from data loading to reporting:

.. code-block:: python

   import pandas as pd
   import matplotlib.pyplot as plt
   from lwdid import lwdid

   # 1. Load and prepare data
   data = pd.read_csv('policy_data.csv')

   # 2. Descriptive statistics
   print("Sample composition:")
   print(data.groupby(['treated', 'post']).size().unstack())

   # 3. Main estimation
   results_main = lwdid(
       data,
       y='outcome',
       d='treated',
       ivar='unit',
       tvar='year',
       post='post_policy',
       rolling='detrend',
       controls=['baseline_x1', 'baseline_x2'],
       vce='hc3',
       ri=True,
       rireps=2000,
       seed=42
   )

   # 4. Print results
   print("\n" + "="*80)
   print("MAIN RESULTS")
   print("="*80)
   results_main.summary()

   # 5. Robustness checks
   print("\n" + "="*80)
   print("ROBUSTNESS CHECKS")
   print("="*80)

   # Without controls
   results_no_controls = lwdid(
       data, 'outcome', 'treated', 'unit', 'year', 'post_policy',
       'detrend', vce='hc3'
   )

   # Demean instead of detrend
   results_demean = lwdid(
       data, 'outcome', 'treated', 'unit', 'year', 'post_policy',
       'demean', controls=['baseline_x1', 'baseline_x2'], vce='hc3'
   )

   # Compare
   robustness = pd.DataFrame({
       'Specification': ['Main', 'No Controls', 'Demean'],
       'ATT': [results_main.att, results_no_controls.att, results_demean.att],
       'SE': [results_main.se_att, results_no_controls.se_att, results_demean.se_att],
       'p-value': [results_main.pvalue, results_no_controls.pvalue, results_demean.pvalue]
   })
   print(robustness.to_string(index=False))

   # 6. Parallel trends test
   print("\n" + "="*80)
   print("PARALLEL TRENDS TEST")
   print("="*80)

   period_effects = results_main.att_by_period
   pre_treatment = period_effects[period_effects['tindex'] < results_main.tpost1]
   sig_pre = pre_treatment[pre_treatment['pval'] < 0.05]

   if len(sig_pre) == 0:
       print("✓ No significant pre-treatment effects detected")
   else:
       print(f"✗ {len(sig_pre)} significant pre-treatment effects detected:")
       print(sig_pre[['period', 'beta', 'pval']])

   # 7. Visualizations
   # Plot treatment effects over time
   results_main.plot()

   # Save a separate figure if needed, for example:
   # results_main.plot(graph_options={'savefig': 'analysis_results.png'})

   # 8. Export results
   results_main.to_excel('main_results.xlsx')
   results_main.to_latex('main_results.tex')
   robustness.to_csv('robustness_checks.csv', index=False)

   print("\n" + "="*80)
   print("Analysis complete. Results saved to:")
   print("  - main_results.xlsx")
   print("  - main_results.tex")
   print("  - robustness_checks.csv")
   print("  - analysis_results.png")
   print("="*80)

See Also
--------

- :doc:`../user_guide` - Comprehensive usage guide
- :doc:`../quickstart` - Quick start tutorial
- :doc:`../api/index` - Complete API reference
- :doc:`../methodological_notes` - Theoretical background

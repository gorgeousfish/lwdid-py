"""
Lee and Wooldridge (2025) Difference-in-Differences Estimator

Implements the Lee and Wooldridge (2025) method for difference-in-differences
estimation with small cross-sectional samples.

Authors: Xuanyu Cai, Wenli Xu
"""

from typing import List, Optional, Union
import random
import numpy as np

import pandas as pd

from . import estimation, transformations, validation
from .randomization import randomization_inference
from .results import LWDIDResults


def lwdid(
    data: pd.DataFrame,
    y: str,
    d: str,
    ivar: str,
    tvar: Union[str, List[str]],
    post: str,
    rolling: str,
    vce: Optional[str] = None,
    controls: Optional[List[str]] = None,
    cluster_var: Optional[str] = None,
    ri: bool = False,
    rireps: int = 1000,
    seed: Optional[int] = None,
    ri_method: str = 'bootstrap',
    graph: bool = False,
    gid: Optional[Union[str, int]] = None,
    graph_options: Optional[dict] = None,
    **kwargs,
) -> LWDIDResults:
    """
    Difference-in-Differences Estimator for Small Cross-Sectional Samples

    Implements the Lee and Wooldridge (2025) method for difference-in-differences
    estimation with small numbers of treated or control units. Transforms panel
    data into cross-sectional form by removing unit-specific pre-treatment patterns,
    enabling exact t-based inference under classical linear model assumptions or
    randomization inference without distributional assumptions.

    Confidence intervals use t-distribution critical values with degrees of freedom
    equal to N-k for homoskedastic standard errors (N units, k parameters) or G-1
    for cluster-robust standard errors (G clusters), providing exact coverage under
    classical linear model assumptions.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format with one row per unit-time observation. Must
        contain columns for outcome, treatment indicator, unit identifier, time
        variable, and post-treatment indicator. Requires at least 3 cross-sectional
        units. Each (unit, time) combination must be unique.
    y : str
        Outcome variable column name.
    d : str
        Treatment indicator column name. Must be a unit-level (time-invariant)
        indicator where non-zero values denote treated units (Dᵢ = 1) and zero
        values denote control units (Dᵢ = 0). Do not use a time-varying treatment
        indicator Wᵢₜ = Dᵢ × postₜ.
    ivar : str
        Unit identifier column name. Accepts numeric or string identifiers;
        string identifiers are internally converted to numeric codes.
    tvar : str or list of str
        Time variable specification:

        - Annual data: single string (e.g., 'year')
        - Quarterly data: list [year_var, quarter_var] where quarter_var
          contains values in {1, 2, 3, 4}

    post : str
        Post-treatment indicator column name. Internally binarized as ``post != 0``:
        non-zero values indicate post-treatment periods (1), zero values indicate
        pre-treatment periods (0). Must be a function of time only (common treatment
        timing) and must be persistent (no treatment reversals).
    rolling : {'demean', 'detrend', 'demeanq', 'detrendq'}
        Transformation method (case-insensitive):

        - 'demean': Unit-specific demeaning (Procedure 2.1)
        - 'detrend': Unit-specific detrending (Procedure 3.1)
        - 'demeanq': Demeaning with quarterly fixed effects
        - 'detrendq': Detrending with quarterly fixed effects

    vce : {None, 'robust', 'hc1', 'hc3', 'cluster'}, optional
        Variance estimator (default=None, case-insensitive):

        - None: Homoskedastic standard errors (exact inference under normality)
        - 'robust' or 'hc1': HC1 heteroskedasticity-robust standard errors
        - 'hc3': HC3 small-sample adjusted standard errors
        - 'cluster': Cluster-robust standard errors (requires cluster_var)

    controls : list of str, optional
        Time-invariant control variable names. Controls are included only if both
        N_treated > K+1 and N_control > K+1, where K is the number of controls.
        Otherwise, a warning is issued and controls are excluded.
    cluster_var : str, optional
        Clustering variable name (required when vce='cluster').
    ri : bool, default=False
        Whether to perform randomization inference. Returns a p-value for H₀: ATT=0
        based on resampling.
    rireps : int, default=1000
        Number of randomization inference replications.
    seed : int, optional
        Random seed for reproducibility. If not specified, a random seed is
        generated. Also accepts 'riseed' as an alias via **kwargs.
    ri_method : {'bootstrap', 'permutation'}, default='bootstrap'
        Randomization inference resampling method (used only if ri=True):

        - 'bootstrap': With-replacement sampling
        - 'permutation': Without-replacement permutation (Fisher randomization
          inference)

    graph : bool, default=False
        Whether to generate a plot of transformed outcomes over time.
    gid : str or int, optional
        Specific unit identifier to plot. If omitted, plots the treated group
        average.
    graph_options : dict, optional
        Plotting options (figsize, title, xlabel, ylabel, legend_loc, savefig).

    **kwargs : dict, optional
        Reserved for future extensions.

    Returns
    -------
    LWDIDResults
        Results object containing:

        **Attributes:**

        - att : float
            Average treatment effect on the treated
        - se_att : float
            Standard error of ATT
        - t_stat : float
            t-statistic for H0: ATT=0
        - pvalue : float
            Two-sided p-value
        - ci_lower, ci_upper : float
            95% confidence interval bounds
        - df_inference : int
            Degrees of freedom for inference (N-k for non-clustered, G-1 for clustered)
        - nobs : int
            Number of observations in cross-sectional regression
        - n_treated, n_control : int
            Number of treated and control units
        - att_by_period : pd.DataFrame
            Period-specific ATT estimates
        - ri_pvalue : float
            Randomization inference p-value (if ri=True)

        **Methods:**

        - summary() : Formatted results summary
        - plot() : Visualization of residualized outcomes
        - to_excel(path) : Export to Excel
        - to_csv(path) : Export period-specific effects to CSV
        - to_latex(path) : Export to LaTeX table

    Raises
    ------
    MissingRequiredColumnError
        Required columns not found in data.
    InvalidRollingMethodError
        Invalid rolling method or quarterly method used without quarterly tvar.
    InsufficientDataError
        Sample size N < 3 or no pre-/post-treatment observations.
    NoTreatedUnitsError
        No units with d==1.
    NoControlUnitsError
        No units with d==0.
    InsufficientPrePeriodsError
        Insufficient pre-treatment periods for chosen transformation.

    Notes
    -----
    This implementation requires common treatment timing: all treated units begin
    treatment in the same period, treatment is persistent, and the time index forms
    a contiguous sequence. Staggered adoption is not supported in this version. See
    Lee and Wooldridge (2025, Section 7) for methods accommodating staggered rollouts.
    
    See Also
    --------
    LWDIDResults : Results class documentation
    
    Examples
    --------
    **Basic usage with demeaning:**
    
    >>> import pandas as pd
    >>> from lwdid import lwdid
    >>> data = pd.read_csv('smoking.csv')
    >>> results = lwdid(
    ...     data=data, 
    ...     y='lcigsale', 
    ...     d='treated', 
    ...     ivar='state',
    ...     tvar='year', 
    ...     post='post', 
    ...     rolling='demean'
    ... )
    >>> print(results.summary())
    >>> print(f"ATT: {results.att:.4f}, SE: {results.se_att:.4f}, p={results.pvalue:.3f}")
    
    **Detrending with HC3 standard errors:**
    
    >>> results = lwdid(
    ...     data=data,
    ...     y='lcigsale',
    ...     d='treated',
    ...     ivar='state',
    ...     tvar='year',
    ...     post='post',
    ...     rolling='detrend',
    ...     vce='hc3'
    ... )
    >>> print(f"ATT (detrend): {results.att:.4f}")
    
    **Quarterly data with seasonal effects:**
    
    >>> data_q = pd.read_csv('quarterly_data.csv')
    >>> results = lwdid(
    ...     data=data_q,
    ...     y='gdp',
    ...     d='policy',
    ...     ivar='country',
    ...     tvar=['year', 'quarter'],
    ...     post='post',
    ...     rolling='detrendq',
    ...     vce='hc3'
    ... )
    
    **Randomization inference:**
    
    >>> results = lwdid(
    ...     data=data,
    ...     y='lcigsale',
    ...     d='treated',
    ...     ivar='state',
    ...     tvar='year',
    ...     post='post',
    ...     rolling='demean',
    ...     ri=True,
    ...     rireps=5000,
    ...     seed=12345
    ... )
    >>> print(f"RI p-value: {results.ri_pvalue:.4f}")
    
    **With control variables:**
    
    >>> results = lwdid(
    ...     data=data,
    ...     y='lcigsale',
    ...     d='treated',
    ...     ivar='state',
    ...     tvar='year',
    ...     post='post',
    ...     rolling='demean',
    ...     controls=['income', 'population']
    ... )
    
    **Visualization:**
    
    >>> results = lwdid(
    ...     data=data,
    ...     y='lcigsale',
    ...     d='treated',
    ...     ivar='state',
    ...     tvar='year',
    ...     post='post',
    ...     rolling='demean',
    ...     graph=True,
    ...     gid='California',
    ...     graph_options={'title': 'CA Cigarette Sales', 'ylabel': 'Log Sales'}
    ... )
    
    **Export results:**
    
    >>> results.to_excel('did_results.xlsx')
    >>> results.att_by_period.to_csv('period_effects.csv', index=False)
    """
    if vce is not None:
        vce = vce.lower()

    data_clean, metadata = validation.validate_and_prepare_data(
        data=data,
        y=y,
        d=d,
        ivar=ivar,
        tvar=tvar,
        post=post,
        rolling=rolling,
        controls=controls,
    )

    rolling = metadata['rolling']

    data_transformed = transformations.apply_rolling_transform(
        data=data_clean,
        y=y,
        ivar=ivar,
        tindex='tindex',
        post='post_',
        rolling=rolling,
        tpost1=metadata['tpost1'],
        quarter=tvar[1] if not isinstance(tvar, str) else None,
    )

    results_dict = estimation.estimate_att(
        data=data_transformed,
        y_transformed='ydot_postavg',
        d='d_',
        ivar=ivar,
        controls=controls,
        vce=vce,
        cluster_var=cluster_var,
        sample_filter=data_transformed['firstpost'],
    )

    # Build period labels
    if isinstance(tvar, str):
        period_labels = {
            t: str(int(year))
            for t, year in data_transformed.groupby('tindex')[tvar].first().items()
        }
    else:
        year_var, quarter_var = tvar[0], tvar[1]
        period_labels = {}
        for t in data_transformed['tindex'].unique():
            row = data_transformed[data_transformed['tindex'] == t].iloc[0]
            year_val = int(row[year_var])
            quarter_val = int(row[quarter_var])
            period_labels[t] = f"{year_val}q{quarter_val}"

    Tmax = int(data_transformed['tindex'].max())
    controls_spec = results_dict.get('controls_spec', None)

    period_df = estimation.estimate_period_effects(
        data=data_transformed,
        ydot='ydot',
        d='d_',
        tindex='tindex',
        tpost1=metadata['tpost1'],
        Tmax=Tmax,
        controls_spec=controls_spec,
        vce=vce,
        cluster_var=cluster_var,
        period_labels=period_labels
    )

    # Assemble period-specific effects table
    avg_row = pd.DataFrame([{
        'period': 'average',
        'tindex': '-',  # Fixed: Use string '-' instead of integer -1
        'beta': results_dict['att'],
        'se': results_dict['se_att'],
        'ci_lower': results_dict['ci_lower'],
        'ci_upper': results_dict['ci_upper'],
        'tstat': results_dict['t_stat'],
        'pval': results_dict['pvalue'],
        'N': results_dict['nobs']
    }])

    avg_row['is_avg'] = True
    period_df['is_avg'] = False

    att_by_period = pd.concat([avg_row, period_df], ignore_index=True)
    att_by_period = att_by_period.sort_values(
        ['is_avg', 'tindex'], ascending=[False, True]
    )

    att_by_period = att_by_period.drop(columns=['is_avg']).reset_index(drop=True)
    
    # Ensure tindex column is string type for consistency
    att_by_period['tindex'] = att_by_period['tindex'].astype(str)
    
    att_by_period = att_by_period[[
        'period', 'tindex', 'beta', 'se', 'ci_lower', 'ci_upper', 'tstat', 'pval', 'N'
    ]]

    if ri:
        if seed is None and 'riseed' in kwargs:
            seed = kwargs['riseed']

        actual_seed = seed if seed is not None else random.randint(1000, 1001000)

        firstpost_df = data_transformed.loc[data_transformed['firstpost']].copy()
        if metadata.get('id_mapping') is not None:
            firstpost_df.attrs['id_mapping'] = metadata['id_mapping']

        ri_result = randomization_inference(
            firstpost_df=firstpost_df,
            y_col='ydot_postavg',
            d_col='d_',
            ivar=ivar,
            rireps=rireps,
            seed=actual_seed,
            att_obs=results_dict['att'],
            ri_method=ri_method,
            controls=controls,
        )

    results = LWDIDResults(results_dict, metadata, att_by_period)

    results.data = data_transformed
    if metadata.get('id_mapping') is not None:
        results.data.attrs['id_mapping'] = metadata['id_mapping']

    if ri:
        results.ri_pvalue = ri_result['p_value']
        results.rireps = int(rireps)
        results.ri_seed = int(actual_seed)
        results.ri_method = ri_result['ri_method']
        results.ri_valid = ri_result['ri_valid']
        results.ri_failed = ri_result['ri_failed']

    if graph:
        try:
            results.plot(gid=gid, graph_options=graph_options)
        except Exception as e:
            import warnings
            warnings.warn(
                f"Plotting failed: {type(e).__name__}: {str(e)}. "
                f"The estimation results are unaffected.",
                UserWarning,
                stacklevel=2
            )

    return results

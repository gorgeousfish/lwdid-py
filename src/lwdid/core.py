"""
Lee and Wooldridge (2025) Difference-in-Differences Estimator

Implements the Lee and Wooldridge (2025) method for difference-in-differences
estimation with small cross-sectional samples.
"""

from typing import Dict, List, Optional, Union
import logging
import random
import warnings
import numpy as np

import pandas as pd

from . import estimation, transformations, validation
from .randomization import randomization_inference
from .results import LWDIDResults
from .validation import validate_staggered_data, is_never_treated

# Configure logging
logger = logging.getLogger('lwdid')


def lwdid(
    data: pd.DataFrame,
    y: str,
    d: str = None,
    ivar: str = None,
    tvar: Union[str, List[str]] = None,
    post: str = None,
    rolling: str = 'demean',
    *,  # Force keyword-only for staggered parameters
    # === Staggered parameters (keyword-only) ===
    gvar: Optional[str] = None,
    control_group: str = 'not_yet_treated',
    estimator: str = 'ra',
    aggregate: str = 'cohort',
    # === Existing parameters ===
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
    
    **Staggered setting (Castle Law example):**
    
    >>> data = pd.read_csv('castle.csv')
    >>> data['gvar'] = data['effyear'].fillna(0).astype(int)
    >>> results = lwdid(
    ...     data=data,
    ...     y='lhomicide',
    ...     ivar='sid',
    ...     tvar='year',
    ...     gvar='gvar',
    ...     control_group='never_treated',
    ...     aggregate='overall'
    ... )
    >>> print(f"Overall ATT: {results.att_overall:.4f}")
    """
    if vce is not None:
        vce = vce.lower()

    # === Mode Detection ===
    if gvar is not None:
        # Staggered mode: warn if d/post also provided
        if d is not None or post is not None:
            warnings.warn(
                f"同时提供了gvar和d/post参数，优先使用staggered模式。"
                f"Staggered模式下将忽略d和post参数。",
                UserWarning,
                stacklevel=2
            )
        
        # Call staggered implementation
        return _lwdid_staggered(
            data=data, y=y, ivar=ivar, tvar=tvar, gvar=gvar,
            rolling=rolling, control_group=control_group,
            estimator=estimator, aggregate=aggregate,
            vce=vce, controls=controls, cluster_var=cluster_var,
            ri=ri, rireps=rireps, seed=seed, ri_method=ri_method,
            graph=graph, gid=gid, graph_options=graph_options,
            **kwargs
        )
    else:
        # Common timing mode: validate d and post
        if d is None:
            raise ValueError(
                "Common timing模式需要提供'd'参数（单位级处理指示符）。\n"
                "如果您的数据是staggered设定（不同单位在不同时期开始处理），"
                "请使用gvar参数指定首次处理时期列。"
            )
        if post is None:
            raise ValueError(
                "Common timing模式需要提供'post'参数（后处理期指示符）。\n"
                "如果您的数据是staggered设定，请使用gvar参数。"
            )
        if ivar is None:
            raise ValueError("需要提供'ivar'参数（单位标识符列名）。")
        if tvar is None:
            raise ValueError("需要提供'tvar'参数（时间变量列名）。")

    # === Common Timing Mode (existing logic) ===
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
            warnings.warn(
                f"Plotting failed: {type(e).__name__}: {str(e)}. "
                f"The estimation results are unaffected.",
                UserWarning,
                stacklevel=2
            )

    return results


# =============================================================================
# Staggered DiD Implementation
# =============================================================================

def _validate_control_group_for_aggregate(
    aggregate: str,
    control_group: str,
    has_never_treated: bool,
    n_never_treated: int = 0
) -> tuple:
    """
    Validate and auto-switch control group strategy based on aggregate level.
    
    Rules (from PRD Section 1.3):
    1. aggregate='cohort' or 'overall' requires never_treated control group
    2. No NT units makes cohort/overall effects impossible to estimate
    3. Warn if NT units are too few (N_NT < 2)
    
    Returns
    -------
    tuple[str, str]
        (control_group_used, warning_message or None)
    """
    warning_msg = None
    control_group_used = control_group

    if aggregate in ('cohort', 'overall'):
        # Force switch to never_treated
        if control_group != 'never_treated':
            warning_msg = (
                f"{aggregate}效应估计要求never_treated控制组（论文公式7.10/7.18），"
                f"已自动从'{control_group}'切换到'never_treated'。"
            )
            logger.info(warning_msg)
            warnings.warn(warning_msg, UserWarning, stacklevel=4)
            control_group_used = 'never_treated'

        # Validate NT units exist
        if not has_never_treated:
            raise ValueError(
                f"无法估计{aggregate}效应: 数据中没有never treated单位。\n"
                f"原因: {aggregate}效应需要NT单位作为统一参照基准。\n"
                f"  - Cohort效应 (公式7.10): 不同cohort的变换使用不同pre-treatment时期，只有NT单位能提供一致参照。\n"
                f"  - 整体效应 (公式7.18): NT单位需计算所有cohort的加权变换。\n"
                f"建议: 使用aggregate='none'估计(g,r)特定效应，可使用not-yet-treated控制组。"
            )

        # Warn if NT units are too few
        if n_never_treated < 2:
            warnings.warn(
                f"Never treated单位数量过少 (N={n_never_treated})，"
                f"推断结果可能不可靠。建议N_NT >= 2。",
                UserWarning,
                stacklevel=4
            )

    return control_group_used, warning_msg


def _lwdid_staggered(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: Union[str, List[str]],
    gvar: str,
    rolling: str,
    control_group: str,
    estimator: str,
    aggregate: str,
    vce: Optional[str],
    controls: Optional[List[str]],
    cluster_var: Optional[str],
    ri: bool,
    rireps: int,
    seed: Optional[int],
    ri_method: str,
    graph: bool,
    gid: Optional[Union[str, int]],
    graph_options: Optional[dict],
    **kwargs
) -> LWDIDResults:
    """
    Staggered DiD estimation implementation.
    
    This implements the Lee & Wooldridge (2023, 2025) staggered DiD method.
    """
    from .staggered import (
        transformations as stag_trans,
        control_groups as stag_ctrl,
        estimation as stag_est,
        aggregation as stag_agg
    )
    
    # === Step 0: Parameter validation ===
    if ivar is None:
        raise ValueError("Staggered模式需要提供'ivar'参数（单位标识符列名）。")
    if tvar is None:
        raise ValueError("Staggered模式需要提供'tvar'参数（时间变量列名）。")
    
    rolling_lower = rolling.lower()
    if rolling_lower not in ('demean', 'detrend'):
        raise ValueError(
            f"Staggered模式不支持rolling='{rolling}'。\n"
            f"有效值: 'demean', 'detrend'。\n"
            f"注意: 季度变换('demeanq', 'detrendq')在staggered模式下尚未支持。"
        )
    
    # === Estimator validation ===
    estimator_lower = estimator.lower()
    VALID_ESTIMATORS = ('ra', 'ipwra', 'psm')
    if estimator_lower not in VALID_ESTIMATORS:
        raise ValueError(
            f"无效的estimator='{estimator}'。\n"
            f"有效值: {VALID_ESTIMATORS}"
        )
    
    # IPWRA和PSM需要controls参数
    if estimator_lower in ('ipwra', 'psm') and not controls:
        raise ValueError(
            f"estimator='{estimator}'需要提供controls参数。\n"
            f"IPWRA需要控制变量来构建倾向得分模型和结果模型。\n"
            f"PSM需要控制变量来构建倾向得分模型。"
        )
    
    # RI validation will be done after estimation if ri=True
    
    # === Step 1: Data validation ===
    validation_result = validate_staggered_data(
        data=data,
        gvar=gvar,
        ivar=ivar,
        tvar=tvar,
        y=y,
        controls=controls
    )
    cohorts = validation_result['cohorts']
    has_never_treated = validation_result['n_never_treated'] > 0
    n_never_treated = validation_result['n_never_treated']
    T_max = validation_result['T_max']
    T_min = validation_result['T_min']
    cohort_sizes = validation_result['cohort_sizes']
    
    # Print validation warnings
    for warning in validation_result.get('warnings', []):
        warnings.warn(warning, UserWarning, stacklevel=3)
    
    # === Step 2: Control group strategy validation and auto-switch ===
    control_group_used, switch_warning = _validate_control_group_for_aggregate(
        aggregate=aggregate,
        control_group=control_group,
        has_never_treated=has_never_treated,
        n_never_treated=n_never_treated
    )
    
    # === Step 3: Data transformation ===
    tvar_str = tvar if isinstance(tvar, str) else tvar[0]
    
    transform_func = (
        stag_trans.transform_staggered_demean 
        if rolling_lower == 'demean' 
        else stag_trans.transform_staggered_detrend
    )
    
    data_transformed = transform_func(
        data=data,
        y=y,
        ivar=ivar,
        tvar=tvar_str,
        gvar=gvar,
    )
    
    # === Step 4: (g,r) effect estimation ===
    cohort_time_effects = stag_est.estimate_cohort_time_effects(
        data_transformed=data_transformed,
        gvar=gvar,
        ivar=ivar,
        tvar=tvar_str,
        controls=controls,
        vce=vce,
        cluster_var=cluster_var,
        control_strategy=control_group_used,
        estimator=estimator_lower,  # 支持ra/ipwra/psm
        transform_type=rolling_lower,  # 'demean' or 'detrend'
    )
    
    # Convert to DataFrame
    att_by_cohort_time = pd.DataFrame([
        {
            'cohort': e.cohort,
            'period': e.period,
            'event_time': e.event_time,
            'att': e.att,
            'se': e.se,
            'ci_lower': e.ci_lower,
            'ci_upper': e.ci_upper,
            't_stat': e.t_stat,
            'pvalue': e.pvalue,
            'n_treated': e.n_treated,
            'n_control': e.n_control,
            'n_total': e.n_total
        }
        for e in cohort_time_effects
    ])
    
    # === Step 5: Effect aggregation ===
    att_by_cohort = None
    att_overall = None
    se_overall = None
    cohort_weights = {}
    t_stat_overall = None
    pvalue_overall = None
    ci_overall = (None, None)
    
    if aggregate in ('cohort', 'overall'):
        # Cohort effect aggregation
        cohort_effects = stag_agg.aggregate_to_cohort(
            data_transformed=data_transformed,
            gvar=gvar,
            ivar=ivar,
            tvar=tvar_str,
            cohorts=cohorts,
            T_max=T_max,
            transform_type='demean' if rolling_lower == 'demean' else 'detrend',
            vce=vce,
            cluster_var=cluster_var,
        )
        
        att_by_cohort = pd.DataFrame([
            {
                'cohort': c.cohort,
                'att': c.att,
                'se': c.se,
                'ci_lower': c.ci_lower,
                'ci_upper': c.ci_upper,
                't_stat': c.t_stat,
                'pvalue': c.pvalue,
                'n_periods': c.n_periods,
                'n_units': c.n_units
            }
            for c in cohort_effects
        ])
    
    if aggregate == 'overall':
        # Overall effect estimation (implements equation 7.18-7.19)
        overall_effect = stag_agg.aggregate_to_overall(
            data_transformed=data_transformed,
            gvar=gvar,
            ivar=ivar,
            tvar=tvar_str,
            transform_type='demean' if rolling_lower == 'demean' else 'detrend',
            vce=vce,
            cluster_var=cluster_var,
        )
        
        att_overall = overall_effect.att
        se_overall = overall_effect.se
        cohort_weights = overall_effect.cohort_weights
        t_stat_overall = overall_effect.t_stat
        pvalue_overall = overall_effect.pvalue
        ci_overall = (overall_effect.ci_lower, overall_effect.ci_upper)
    
    # === Step 6: Build results ===
    # Compute unit statistics
    unit_gvar = data.groupby(ivar)[gvar].first()
    n_treated = int(sum(cohort_sizes.values()))
    n_control = n_never_treated
    
    # Build results dict compatible with LWDIDResults
    results_dict = {
        # Mode identifier
        'is_staggered': True,
        
        # Cohort information
        'cohorts': cohorts,
        'cohort_sizes': cohort_sizes,
        
        # Effect estimates
        'att_by_cohort_time': att_by_cohort_time,
        'att_by_cohort': att_by_cohort,
        'att_overall': att_overall,
        'se_overall': se_overall,
        
        # Overall effect extra info
        'cohort_weights': cohort_weights,
        'ci_overall_lower': ci_overall[0],
        'ci_overall_upper': ci_overall[1],
        't_stat_overall': t_stat_overall,
        'pvalue_overall': pvalue_overall,
        
        # Unit statistics
        'n_treated': n_treated,
        'n_control': n_control,
        'nobs': n_treated + n_control,
        
        # Configuration
        'control_group': control_group,
        'control_group_used': control_group_used,
        'aggregate': aggregate,
        'estimator': estimator,
        'rolling': rolling,
        'n_never_treated': n_never_treated,
        
        # For LWDIDResults compatibility (use overall or first cohort effect)
        'att': att_overall if att_overall is not None else (
            att_by_cohort_time['att'].mean() if len(att_by_cohort_time) > 0 else None
        ),
        'se_att': se_overall if se_overall is not None else (
            att_by_cohort_time['se'].mean() if len(att_by_cohort_time) > 0 else None
        ),
        't_stat': t_stat_overall if t_stat_overall is not None else None,
        'pvalue': pvalue_overall if pvalue_overall is not None else None,
        'ci_lower': ci_overall[0],
        'ci_upper': ci_overall[1],
        'df_resid': n_treated + n_control - 2,
        'vce_type': vce if vce else 'ols',
        'params': None,
        'bse': None,
        'vcov': None,
        'resid': None,
    }
    
    metadata = {
        'is_staggered': True,
        'rolling': rolling,
        'control_group': control_group,
        'control_group_used': control_group_used,
        'aggregate': aggregate,
        'estimator': estimator,
        'cohorts': cohorts,
        'T_max': T_max,
        'T_min': T_min,
        'has_never_treated': has_never_treated,
        'n_never_treated': n_never_treated,
        'n_cohorts': len(cohorts),
        'vce': vce,
        'depvar': y,
        'K': 0,  # Not applicable for staggered
        'tpost1': cohorts[0] if cohorts else 0,
        'N_treated': n_treated,
        'N_control': n_control,
    }
    
    # Build LWDIDResults object
    results = LWDIDResults(results_dict, metadata, att_by_cohort_time)
    results.data = data_transformed
    
    # === Step 7: Randomization Inference (if requested) ===
    if ri:
        from .staggered.randomization import randomization_inference_staggered
        
        # Determine target and observed ATT
        if aggregate == 'overall' and att_overall is not None:
            ri_target = 'overall'
            ri_observed = att_overall
            target_cohort_ri = None
            target_period_ri = None
        elif aggregate == 'cohort' and att_by_cohort is not None and len(att_by_cohort) > 0:
            # Use first cohort effect
            ri_target = 'cohort'
            ri_observed = att_by_cohort.iloc[0]['att']
            target_cohort_ri = int(att_by_cohort.iloc[0]['cohort'])
            target_period_ri = None
        else:
            # Use first (g,r) effect
            ri_target = 'cohort_time'
            if len(cohort_time_effects) > 0:
                first_effect = cohort_time_effects[0]
                ri_observed = first_effect.att
                target_cohort_ri = first_effect.cohort
                target_period_ri = first_effect.period
            else:
                warnings.warn(
                    "无可用的效应估计值，跳过随机化推断。",
                    UserWarning,
                    stacklevel=3
                )
                return results
        
        actual_seed = seed if seed is not None else random.randint(1000, 1001000)
        
        try:
            ri_result = randomization_inference_staggered(
                data=data,
                gvar=gvar,
                ivar=ivar,
                tvar=tvar_str,
                y=y,
                cohorts=cohorts,
                observed_att=ri_observed,
                target=ri_target,
                target_cohort=target_cohort_ri,
                target_period=target_period_ri,
                ri_method=ri_method,
                rireps=rireps,
                seed=actual_seed,
                rolling=rolling,
                controls=controls,
                vce=vce,
                cluster_var=cluster_var,
                n_never_treated=n_never_treated,
            )
            
            # Store RI results
            results.ri_pvalue = ri_result.p_value
            results.rireps = rireps
            results.ri_seed = actual_seed
            results.ri_method = ri_result.ri_method
            results.ri_valid = ri_result.ri_valid
            results.ri_failed = ri_result.ri_failed
            results.ri_target = ri_target
            
        except Exception as e:
            warnings.warn(
                f"随机化推断失败: {type(e).__name__}: {e}",
                UserWarning,
                stacklevel=3
            )
    
    return results

"""
Result container for difference-in-differences estimation outputs.

This module provides the LWDIDResults class for encapsulating estimation
outputs from rolling transformation DiD methodology, supporting three scenarios:

1. **Small-sample common timing**: Results include exact t-based inference
   statistics under classical linear model assumptions.

2. **Large-sample common timing**: Results include asymptotic inference with
   heteroskedasticity-robust standard errors.

3. **Staggered adoption**: Results include cohort-time specific effects,
   cohort-level aggregations, and overall weighted effects with flexible
   control group strategies.

The class implements immutable core attributes via properties to ensure
result integrity, provides multiple summary formats (text, LaTeX, Excel,
CSV), supports event study visualization for staggered designs, and
includes period-specific effects for common timing designs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    import matplotlib.axes
    from lwdid.staggered.parallel_trends import ParallelTrendsTestResult


class LWDIDResults:
    """
    Container for difference-in-differences estimation results.

    Stores all estimation outputs from the lwdid() function implementing
    the rolling transformation methodology. Supports three scenarios:

    1. **Small-sample common timing**: Exact t-based inference under classical
       linear model assumptions.

    2. **Large-sample common timing**: Asymptotic inference with
       heteroskedasticity-robust standard errors.

    3. **Staggered adoption**: Cohort-time specific effects with flexible
       control group strategies.

    All core attributes are read-only properties to ensure result integrity.
    Provides methods for displaying, visualizing, and exporting results.

    Attributes
    ----------
    att : float
        Average treatment effect on the treated (ATT) point estimate.
    se_att : float
        Standard error of ATT.
    t_stat : float
        t-statistic for H0: ATT = 0.
    pvalue : float
        Two-sided p-value for t-test.
    ci_lower : float
        Lower bound of 95% confidence interval.
    ci_upper : float
        Upper bound of 95% confidence interval.
    nobs : int
        Number of observations in the regression.
    n_treated : int
        Number of treated units.
    n_control : int
        Number of control units.
    K : int
        Last pre-treatment period index.
    tpost1 : int
        First post-treatment period index.
    df_resid : int
        Residual degrees of freedom from the main regression.
    df_inference : int
        Degrees of freedom used for inference. For cluster-robust standard
        errors, this is G - 1 (number of clusters minus 1). For other variance
        estimators, this equals df_resid.
    rolling : str
        Transformation method used ('demean', 'detrend', 'demeanq', or
        'detrendq').
    vce_type : str
        Variance estimator type ('ols', 'robust', 'hc1', 'hc3', or 'cluster').
    cluster_var : str or None
        Clustering variable name (if vce='cluster').
    n_clusters : int or None
        Number of clusters (if vce='cluster').
    controls_used : bool
        Whether control variables were included.
    controls : list
        List of control variable names used.
    params : array-like
        Full vector of regression coefficients.
    bse : array-like
        Standard errors of all coefficients.
    vcov : array-like
        Variance-covariance matrix.
    att_by_period : pd.DataFrame
        Period-specific ATT estimates with columns: period, tindex, beta, se,
        ci_lower, ci_upper, tstat, pval, N.
    ri_pvalue : float or None
        Randomization inference p-value (if ri=True was specified).
    ri_seed : int or None
        Random seed used for RI.
    rireps : int or None
        Number of RI permutations.
    ri_method : str or None
        Randomization inference method used ('bootstrap' or 'permutation').
        Only available if ri=True was specified.
    ri_valid : int or None
        Number of valid (successful) RI replications. Only available if
        ri=True.
    ri_failed : int or None
        Number of failed RI replications. Only available if ri=True.
    data : pd.DataFrame
        Transformed data used for regression. Contains the rolling-transformed
        outcome variable and other regression variables. If the original ivar
        was string type, this DataFrame contains numeric IDs (1, 2, 3, ...).
        The mapping between original and numeric IDs is stored in
        data.attrs['id_mapping'].
    is_staggered : bool
        Whether this is a staggered DiD estimation.
    cohorts : list of int
        Sorted list of treatment cohorts (first treatment periods).
        Only available when is_staggered=True.
    cohort_sizes : dict
        Number of units in each cohort. Only available when is_staggered=True.
    att_by_cohort_time : pd.DataFrame or None
        (g,r)-specific ATT estimates with columns: cohort, period, event_time,
        att, se, ci_lower, ci_upper, t_stat, pvalue, n_treated, n_control.
        Only available when is_staggered=True.
    att_by_cohort : pd.DataFrame or None
        Cohort-specific ATT estimates (if aggregate='cohort' or 'overall').
        Only available when is_staggered=True.
    att_overall : float or None
        Overall weighted ATT (if aggregate='overall').
        Only available when is_staggered=True.
    se_overall : float or None
        Standard error of overall ATT.
        Only available when is_staggered=True.
    cohort_weights : dict
        Cohort weights used for overall effect (omega_g = N_g / N_treat).
        Only available when is_staggered=True.
    control_group : str or None
        User-specified control group strategy.
        Only available when is_staggered=True.
    control_group_used : str or None
        Actual control group strategy used (may differ due to auto-switching).
        Only available when is_staggered=True.
    aggregate : str or None
        Aggregation level ('none', 'cohort', 'overall').
        Only available when is_staggered=True.
    estimator : str or None
        Estimation method ('ra', 'ipw', 'ipwra', 'psm').
        Only available when is_staggered=True.

    Methods
    -------
    summary()
        Returns formatted results summary string.
    plot(gid=None, graph_options=None)
        Generates plot of residualized outcomes (control vs. treated).
    plot_event_study(**kwargs)
        Generates event study diagram for staggered designs.
    to_excel(path)
        Exports results to Excel file with multiple sheets.
    to_csv(path)
        Exports period-specific effects to CSV.
    to_latex(path)
        Exports results to LaTeX table format.

    See Also
    --------
    lwdid : Main estimation function that produces LWDIDResults objects.
    """
    
    def __init__(
        self,
        results_dict: dict[str, Any],
        metadata: dict[str, Any],
        att_by_period: pd.DataFrame | None = None,
        cohort_time_effects: list | None = None,
    ):
        """
        Initialize LWDIDResults container with estimation outputs.

        Parameters
        ----------
        results_dict : dict
            Estimation results containing keys: 'att', 'se_att', 't_stat',
            'pvalue', 'ci_lower', 'ci_upper', 'nobs', 'df_resid', 'params',
            'bse', 'vcov', 'resid', 'vce_type'.
        metadata : dict
            Metadata containing keys: 'K', 'tpost1', 'depvar', 'N_treated',
            'N_control'.
        att_by_period : pd.DataFrame, optional
            Period-specific effect estimates for common timing design.
        cohort_time_effects : list, optional
            List of CohortTimeEffect objects for staggered designs.
        """
        self._att = results_dict['att']
        self._se_att = results_dict['se_att']
        self._t_stat = results_dict['t_stat']
        self._pvalue = results_dict['pvalue']
        self._ci_lower = results_dict['ci_lower']
        self._ci_upper = results_dict['ci_upper']
        
        self._K = metadata['K']
        self._tpost1 = metadata['tpost1']
        self._nobs = results_dict['nobs']
        self._n_treated = results_dict.get('n_treated_sample', metadata['N_treated'])
        self._n_control = results_dict.get('n_control_sample', metadata['N_control'])
        self._df_resid = results_dict['df_resid']
        self._df_inference = results_dict.get('df_inference', results_dict['df_resid'])
        
        self._cmd = 'lwdid'
        self._depvar = metadata['depvar']
        self._rolling = metadata.get('rolling', 'demean')
        self._vce_type = results_dict['vce_type']
        self._cluster_var = results_dict.get('cluster_var', None)
        self._n_clusters = results_dict.get('n_clusters', None)
        
        self._controls_used = results_dict.get('controls_used', False)
        self._controls = results_dict.get('controls', [])
        
        self._params = results_dict['params']
        self._bse = results_dict['bse']
        self._vcov = results_dict['vcov']
        
        self._att_by_period = att_by_period
        self._cohort_time_effects = cohort_time_effects
        
        self._resid = results_dict['resid']
        self._metadata = metadata

        self._ri_pvalue: float | None = None
        self._ri_seed: int | None = None
        self._rireps: int | None = None
        self._ri_method: str | None = None
        self._ri_valid: int | None = None
        self._ri_failed: int | None = None
        self._data: pd.DataFrame | None = None
        
        # === Pre-treatment dynamics attributes ===
        self._att_pre_treatment: pd.DataFrame | None = results_dict.get('att_pre_treatment', None)
        self._parallel_trends_test: ParallelTrendsTestResult | None = results_dict.get('parallel_trends_test', None)
        self._include_pretreatment: bool = results_dict.get('include_pretreatment', False)
        
        # === Staggered-specific attributes ===
        self._is_staggered: bool = results_dict.get('is_staggered', False)
        
        if self._is_staggered:
            # Cohort information
            self._cohorts = results_dict.get('cohorts', [])
            self._cohort_sizes = results_dict.get('cohort_sizes', {})
            
            # Effect estimates
            self._att_by_cohort_time = results_dict.get('att_by_cohort_time', None)
            self._att_by_cohort = results_dict.get('att_by_cohort', None)
            self._att_overall = results_dict.get('att_overall', None)
            self._se_overall = results_dict.get('se_overall', None)
            self._ci_overall_lower = results_dict.get('ci_overall_lower', None)
            self._ci_overall_upper = results_dict.get('ci_overall_upper', None)
            self._t_stat_overall = results_dict.get('t_stat_overall', None)
            self._pvalue_overall = results_dict.get('pvalue_overall', None)
            
            # Cohort weights
            self._cohort_weights = results_dict.get('cohort_weights', {})
            
            # Configuration
            self._control_group = results_dict.get('control_group', 'not_yet_treated')
            self._control_group_used = results_dict.get('control_group_used', 'not_yet_treated')
            self._aggregate = results_dict.get('aggregate', 'cohort')
            self._estimator = results_dict.get('estimator', 'ra')
            # n_never_treated attribute
            self._n_never_treated = results_dict.get('n_never_treated', 0)
        else:
            # Set defaults for non-staggered case
            self._cohorts = []
            self._cohort_sizes = {}
            self._att_by_cohort_time = None
            self._att_by_cohort = None
            self._att_overall = None
            self._se_overall = None
            self._ci_overall_lower = None
            self._ci_overall_upper = None
            self._t_stat_overall = None
            self._pvalue_overall = None
            self._cohort_weights = {}
            self._control_group = None
            self._control_group_used = None
            self._aggregate = None
            self._estimator = None
            self._n_never_treated = None
    
    @property
    def att(self) -> float:
        """ATT point estimate."""
        return self._att

    @property
    def se_att(self) -> float:
        """Standard error of ATT."""
        return self._se_att

    @property
    def t_stat(self) -> float:
        """t-statistic for ATT."""
        return self._t_stat

    @property
    def pvalue(self) -> float:
        """Two-sided p-value."""
        return self._pvalue

    @property
    def ci_lower(self) -> float:
        """95% CI lower bound."""
        return self._ci_lower

    @property
    def ci_upper(self) -> float:
        """95% CI upper bound."""
        return self._ci_upper

    @property
    def nobs(self) -> int:
        """Number of observations."""
        return self._nobs

    @property
    def n_treated(self) -> int:
        """Number of treated units."""
        return self._n_treated

    @property
    def n_control(self) -> int:
        """Number of control units."""
        return self._n_control

    @property
    def df_resid(self) -> int:
        """Residual degrees of freedom."""
        return self._df_resid

    @property
    def df_inference(self) -> int:
        """Degrees of freedom for inference."""
        return self._df_inference

    @property
    def K(self) -> int:
        """Last pre-treatment period index."""
        return self._K

    @property
    def tpost1(self) -> int:
        """First post-treatment period index."""
        return self._tpost1

    @property
    def cmd(self) -> str:
        """Command name."""
        return self._cmd

    @property
    def depvar(self) -> str:
        """Dependent variable name."""
        return self._depvar

    @property
    def rolling(self) -> str:
        """Transformation method used."""
        return self._rolling

    @property
    def vce_type(self) -> str | None:
        """Variance estimator type."""
        return self._vce_type

    @property
    def cluster_var(self) -> str | None:
        """Clustering variable name."""
        return self._cluster_var

    @property
    def n_clusters(self) -> int | None:
        """Number of clusters."""
        return self._n_clusters

    @property
    def controls_used(self) -> bool:
        """Whether control variables were included."""
        return self._controls_used

    @property
    def controls(self) -> list:
        """List of control variable names."""
        return list(self._controls)

    @property
    def params(self):
        """Full coefficient vector."""
        return self._params

    @property
    def bse(self):
        """Standard errors of coefficients."""
        return self._bse

    @property
    def vcov(self):
        """Variance-covariance matrix."""
        return self._vcov

    @property
    def att_by_period(self) -> pd.DataFrame | None:
        """Period-specific ATT estimates (returns copy)."""
        if self._att_by_period is None:
            return None
        return self._att_by_period.copy()

    @property
    def ri_pvalue(self) -> float | None:
        """Randomization inference p-value."""
        return self._ri_pvalue

    @ri_pvalue.setter
    def ri_pvalue(self, value: float | None) -> None:
        self._ri_pvalue = value

    @property
    def ri_seed(self) -> int | None:
        """Random seed used for RI."""
        return self._ri_seed

    @ri_seed.setter
    def ri_seed(self, value: int | None) -> None:
        self._ri_seed = value

    @property
    def rireps(self) -> int | None:
        """Number of RI replications."""
        return self._rireps

    @rireps.setter
    def rireps(self, value: int | None) -> None:
        self._rireps = value

    @property
    def ri_method(self) -> str | None:
        """Randomization inference method."""
        return self._ri_method

    @ri_method.setter
    def ri_method(self, value: str | None) -> None:
        self._ri_method = value

    @property
    def ri_valid(self) -> int | None:
        """Number of valid RI replications."""
        return self._ri_valid

    @ri_valid.setter
    def ri_valid(self, value: int | None) -> None:
        self._ri_valid = value

    @property
    def ri_failed(self) -> int | None:
        """Number of failed RI replications."""
        return self._ri_failed

    @ri_failed.setter
    def ri_failed(self, value: int | None) -> None:
        self._ri_failed = value

    @property
    def data(self) -> pd.DataFrame | None:
        """Transformed data used for regression."""
        return self._data

    @data.setter
    def data(self, value: pd.DataFrame | None) -> None:
        self._data = value

    # === Staggered-specific Properties ===

    @property
    def is_staggered(self) -> bool:
        """Whether this is a staggered DiD estimation."""
        return self._is_staggered

    @property
    def cohorts(self) -> list:
        """List of treatment cohorts."""
        return list(self._cohorts)

    @property
    def cohort_sizes(self) -> dict:
        """Number of units in each cohort."""
        return dict(self._cohort_sizes)

    @property
    def att_by_cohort_time(self) -> pd.DataFrame | None:
        """Cohort-time specific ATT estimates (returns copy)."""
        if self._att_by_cohort_time is None:
            return None
        return self._att_by_cohort_time.copy()

    @property
    def att_by_cohort(self) -> pd.DataFrame | None:
        """Cohort-specific ATT estimates (returns copy)."""
        if self._att_by_cohort is None:
            return None
        return self._att_by_cohort.copy()

    @property
    def att_overall(self) -> float | None:
        """Overall weighted ATT estimate."""
        return self._att_overall

    @property
    def se_overall(self) -> float | None:
        """Standard error of overall ATT."""
        return self._se_overall

    @property
    def ci_overall_lower(self) -> float | None:
        """95% CI lower bound for overall ATT."""
        return self._ci_overall_lower

    @property
    def ci_overall_upper(self) -> float | None:
        """95% CI upper bound for overall ATT."""
        return self._ci_overall_upper

    @property
    def t_stat_overall(self) -> float | None:
        """t-statistic for overall ATT."""
        return self._t_stat_overall

    @property
    def pvalue_overall(self) -> float | None:
        """p-value for overall ATT."""
        return self._pvalue_overall

    @property
    def cohort_weights(self) -> dict:
        """Cohort weights for overall effect."""
        return dict(self._cohort_weights)

    @property
    def control_group(self) -> str | None:
        """User-specified control group strategy."""
        return self._control_group

    @property
    def control_group_used(self) -> str | None:
        """Actual control group strategy used."""
        return self._control_group_used

    @property
    def aggregate(self) -> str | None:
        """Aggregation level."""
        return self._aggregate

    @property
    def estimator(self) -> str | None:
        """Estimation method."""
        return self._estimator

    @property
    def n_never_treated(self) -> int | None:
        """Number of never-treated units."""
        return self._n_never_treated

    # === Pre-treatment Dynamics Properties ===

    @property
    def att_pre_treatment(self) -> pd.DataFrame | None:
        """
        Pre-treatment ATT estimates (returns copy).
        
        DataFrame with columns: cohort, period, event_time, att, se,
        ci_lower, ci_upper, t_stat, pvalue, n_treated, n_control,
        is_anchor, rolling_window_size.
        
        Only available when include_pretreatment=True was specified
        during estimation.
        """
        if self._att_pre_treatment is None:
            return None
        return self._att_pre_treatment.copy()

    @att_pre_treatment.setter
    def att_pre_treatment(self, value: pd.DataFrame | None) -> None:
        self._att_pre_treatment = value

    @property
    def parallel_trends_test(self) -> ParallelTrendsTestResult | None:
        """
        Parallel trends test results.
        
        Contains individual t-tests for each pre-treatment period and
        joint F-test for H0: all pre-treatment ATT = 0.
        
        Only available when include_pretreatment=True and
        pretreatment_test=True were specified during estimation.
        """
        return self._parallel_trends_test

    @parallel_trends_test.setter
    def parallel_trends_test(self, value: ParallelTrendsTestResult | None) -> None:
        self._parallel_trends_test = value

    @property
    def include_pretreatment(self) -> bool:
        """Whether pre-treatment dynamics were computed."""
        return self._include_pretreatment
    
    def summary(self) -> str:
        """
        Generate a formatted summary of estimation results.

        For staggered designs, dispatches to summary_staggered(). For common
        timing designs, displays ATT estimate, standard error, t-statistic,
        p-value, confidence interval, and period-specific effects if available.

        Returns
        -------
        str
            Formatted results summary string suitable for console output.
        """
        if self.is_staggered:
            return self.summary_staggered()
        
        sep_line = "=" * 80
        sub_line = "-" * 80
        
        output = []
        output.append(sep_line)
        output.append("                          lwdid Results")
        output.append(sep_line)
        
        output.append(f"Transformation: {self.rolling}")
        
        vce_desc = {
            'ols': 'OLS (Homoskedastic)',
            'robust': 'HC1 (Heteroskedasticity-robust)',
            'hc1': 'HC1 (Heteroskedasticity-robust)',
            'hc3': 'HC3 (Small-sample adjusted)',
            'cluster': f'Cluster-robust (clustered by {self.cluster_var})' if self.cluster_var else 'Cluster-robust'
        }
        vce_display = vce_desc.get(self.vce_type, self.vce_type)
        output.append(f"Variance Type: {vce_display}")
        
        if self.vce_type == 'cluster' and self.n_clusters is not None:
            output.append(f"Number of clusters: {self.n_clusters}")
        
        output.append(f"Dependent Variable: {self.depvar}")
        output.append("")
        
        output.append(f"Number of observations: {self.nobs}")
        output.append(f"Number of treated units: {self.n_treated}")
        output.append(f"Number of control units: {self.n_control}")
        output.append(f"Pre-treatment periods: {self.K} (K={self.K})")
        output.append(f"Post-treatment periods: {self.tpost1} to end (tpost1={self.tpost1})")
        output.append("")
        
        output.append(sub_line)
        output.append("Average Treatment Effect on the Treated")
        output.append(sub_line)
        output.append(f"ATT:        {self.att:>10.4f}")
        output.append(f"Std. Err.:  {self.se_att:>10.4f}  ({self.vce_type})")
        output.append(f"t-stat:     {self.t_stat:>10.2f}")
        output.append(f"P>|t|:      {self.pvalue:>10.3f}")
        output.append(f"df:         {self.df_inference:>10}")
        output.append(f"[95% Conf. Interval]:  {self.ci_lower:>8.4f}   {self.ci_upper:>8.4f}")

        if self.ri_pvalue is not None:
            output.append("")
            output.append("Randomization Inference:")
            method_str = f"method={self.ri_method}" if self.ri_method else ""
            valid_str = f", valid={self.ri_valid}/{self.rireps}" if self.ri_valid is not None else ""
            output.append(f"RI P-value: {self.ri_pvalue:>10.3f}  ({method_str}, seed={self.ri_seed}{valid_str})")

        output.append(sep_line)
        
        if self.att_by_period is not None:
            output.append("")
            output.append("=== Period-by-period post-treatment effects ===")
            output.append(self.att_by_period.head(5).to_string(index=False))
            if len(self.att_by_period) > 5:
                output.append(f"... ({len(self.att_by_period) - 5} more periods)")
            output.append("")
            output.append("Use results.att_by_period to view all period-specific estimates")
        
        return "\n".join(output)
    
    def summary_staggered(self) -> str:
        """
        Generate a formatted summary for staggered DiD estimation results.

        Displays treatment cohorts, sample sizes, control group strategy,
        overall weighted effect (if aggregate='overall'), and cohort-specific
        effects (if aggregate='cohort' or 'overall').

        Returns
        -------
        str
            Formatted results summary string suitable for console output.

        Raises
        ------
        ValueError
            If called on non-staggered estimation results.

        Notes
        -----
        The summary output varies by aggregation level. When aggregate='none',
        only (g,r)-specific effects are available. When aggregate='cohort',
        cohort-specific effects are shown. When aggregate='overall', both
        cohort-specific and overall weighted effects are displayed. If the
        control group strategy was automatically switched from the user-specified
        value, a notification is included.
        """
        if not self.is_staggered:
            raise ValueError(
                "summary_staggered() requires staggered DiD results. "
                "Use summary() for common timing results."
            )
        
        sep_line = "=" * 70
        sub_line = "-" * 70
        
        output = []
        output.append(sep_line)
        output.append("LWDID Staggered DiD Results")
        output.append(sep_line)
        
        # === Basic information ===
        output.append(f"Treatment Cohorts: {', '.join(map(str, self.cohorts))}")
        output.append(f"Number of Treated Units: {self.n_treated}")
        output.append(f"Number of Control Units: {self.n_control}")
        if self.n_never_treated is not None:
            output.append(f"Number of Never Treated Units: {self.n_never_treated}")
        output.append(f"Control Group Strategy: {self.control_group_used}")
        
        # If control group was auto-switched, show notification
        if self.control_group != self.control_group_used:
            output.append(
                f"  Note: Auto-switched from '{self.control_group}' "
                f"for {self.aggregate} effect estimation"
            )
        
        output.append(f"Transformation: {self.rolling}")
        output.append(f"Estimator: {self.estimator}")
        output.append(f"Aggregation: {self.aggregate}")
        output.append(sub_line)
        
        # === Overall effect (shown when aggregate='overall') ===
        if self.att_overall is not None:
            output.append("")
            output.append("Overall Weighted Effect (τ_ω):")
            output.append(f"  ATT_ω   = {self.att_overall:.4f}")
            output.append(f"  SE      = {self.se_overall:.4f}")
            if self.t_stat_overall is not None:
                output.append(f"  t-stat  = {self.t_stat_overall:.3f}")
            if self.pvalue_overall is not None:
                output.append(f"  P>|t|   = {self.pvalue_overall:.3f}")
            if self.ci_overall_lower is not None and self.ci_overall_upper is not None:
                output.append(
                    f"  95% CI: [{self.ci_overall_lower:.4f}, {self.ci_overall_upper:.4f}]"
                )
            output.append("")
            
            # Cohort weights
            if self.cohort_weights:
                output.append("Cohort Weights:")
                for g in sorted(self.cohort_weights.keys()):
                    w = self.cohort_weights[g]
                    n = self.cohort_sizes.get(g, '?')
                    output.append(f"  Cohort {g}: ω = {w:.3f} (N = {n})")
            output.append(sub_line)
        
        # === Cohort effects (shown when aggregate='cohort' or 'overall') ===
        if self.att_by_cohort is not None and not self.att_by_cohort.empty:
            output.append("")
            output.append("Cohort-Specific Effects (τ_g):")
            
            # Header
            header = (
                f"  {'Cohort':>6}  {'ATT':>8}  {'SE':>8}  {'t-stat':>7}  "
                f"{'P>|t|':>6}  {'[95% CI]':>18}  {'N_units':>7}  {'N_periods':>9}"
            )
            output.append(header)
            
            for _, row in self.att_by_cohort.iterrows():
                cohort = int(row['cohort'])
                att = row['att']
                se = row['se']
                # Use column check for optional fields
                ci_l = (
                    row['ci_lower'] if 'ci_lower' in row.index
                    else (att - 1.96 * se)
                )
                ci_u = (
                    row['ci_upper'] if 'ci_upper' in row.index
                    else (att + 1.96 * se)
                )
                t_stat = (
                    row['t_stat'] if 't_stat' in row.index
                    else (att / se if se > 0 else float('nan'))
                )
                pval = row['pvalue'] if 'pvalue' in row.index else float('nan')
                n_units = row['n_units'] if 'n_units' in row.index else '?'
                n_periods = row['n_periods'] if 'n_periods' in row.index else '?'
                
                ci_str = f"[{ci_l:>6.3f}, {ci_u:>6.3f}]"
                line = (
                    f"  {cohort:>6}  {att:>8.4f}  {se:>8.4f}  {t_stat:>7.2f}  "
                    f"{pval:>6.3f}  {ci_str:>18}  {n_units:>7}  {n_periods:>9}"
                )
                output.append(line)
            
            output.append(sub_line)
        
        # === Hint information ===
        output.append("")
        if self.att_by_cohort_time is not None:
            output.append("Use results.att_by_cohort_time for (g,r)-specific effects")
        output.append("Use results.plot_event_study() for Event Study visualization")
        
        # === Pre-treatment Dynamics (shown when include_pretreatment=True) ===
        if self.include_pretreatment and self.att_pre_treatment is not None:
            output.append(sub_line)
            output.append("")
            output.append("Pre-treatment Dynamics")
            output.append("-" * 40)
            
            # Parallel trends test results
            if self.parallel_trends_test is not None:
                pt = self.parallel_trends_test
                output.append("")
                output.append("Parallel Trends Test (H0: all pre-treatment ATT = 0):")
                output.append(f"  F-statistic:  {pt.joint_f_stat:.4f}")
                output.append(f"  P-value:      {pt.joint_pvalue:.4f}")
                output.append(f"  DF (num, den): ({pt.joint_df1}, {pt.joint_df2})")
                output.append(f"  Reject H0:    {'Yes' if pt.reject_null else 'No'} (α={pt.alpha:.2f})")
                output.append("")
            
            # Pre-treatment ATT summary
            pre_df = self.att_pre_treatment
            non_anchor = pre_df[~pre_df['is_anchor']]
            
            if len(non_anchor) > 0:
                output.append("Pre-treatment ATT Estimates:")
                header = (
                    f"  {'e':>4}  {'ATT':>10}  {'SE':>8}  {'t-stat':>7}  "
                    f"{'P>|t|':>6}  {'[95% CI]':>20}  {'Anchor':>6}"
                )
                output.append(header)
                
                # Sort by event_time descending (anchor first, then earlier periods)
                for _, row in pre_df.sort_values('event_time', ascending=False).iterrows():
                    e = int(row['event_time'])
                    att = row['att']
                    se = row['se']
                    t_stat = row['t_stat']
                    pval = row['pvalue']
                    ci_l = row['ci_lower']
                    ci_u = row['ci_upper']
                    is_anchor = row['is_anchor']
                    
                    anchor_str = "  *" if is_anchor else ""
                    
                    if is_anchor:
                        # Anchor point: show as reference
                        ci_str = "[  0.0000,   0.0000]"
                        line = (
                            f"  {e:>4}  {att:>10.4f}  {se:>8.4f}  {'---':>7}  "
                            f"{'---':>6}  {ci_str:>20}{anchor_str}"
                        )
                    else:
                        ci_str = f"[{ci_l:>8.4f}, {ci_u:>8.4f}]"
                        line = (
                            f"  {e:>4}  {att:>10.4f}  {se:>8.4f}  {t_stat:>7.2f}  "
                            f"{pval:>6.3f}  {ci_str:>20}{anchor_str}"
                        )
                    output.append(line)
                
                output.append("")
                output.append("  * Anchor point (e=-1): ATT=0 by construction")
            
            output.append("")
            output.append("Use results.att_pre_treatment for full pre-treatment effects DataFrame")
            output.append("Use results.plot_event_study(include_pre_treatment=True) for visualization")
        
        output.append(sep_line)
        
        return "\n".join(output)
    
    def __repr__(self) -> str:
        """Return a concise string representation of the results object."""
        if self.is_staggered:
            pre_info = ", pre_treatment=True" if self.include_pretreatment else ""
            if self.att_overall is not None:
                return (
                    f"LWDIDResults(staggered=True, att_overall={self.att_overall:.4f}, "
                    f"se={self.se_overall:.4f}, cohorts={len(self.cohorts)}, "
                    f"N_treated={self.n_treated}, N_control={self.n_control}{pre_info})"
                )
            else:
                return (
                    f"LWDIDResults(staggered=True, cohorts={len(self.cohorts)}, "
                    f"aggregate='{self.aggregate}', N_treated={self.n_treated}{pre_info})"
                )
        else:
            return (
                f"LWDIDResults(att={self.att:.4f}, se={self.se_att:.4f}, "
                f"method='{self.rolling}', N={self.nobs})"
            )
    
    def __str__(self) -> str:
        """Return the formatted summary as the string representation."""
        return self.summary()

    @property
    def metadata(self) -> dict[str, Any]:
        """Internal metadata dictionary (returns copy)."""
        return dict(self._metadata)

    def plot(self, gid: str | int | None = None, graph_options: dict | None = None):
        """
        Generate a plot of residualized outcomes for treated and control groups.

        Creates a time series plot comparing the average residualized outcomes
        between treated and control units across all time periods. A vertical
        line indicates the treatment start period.

        Parameters
        ----------
        gid : str or int, optional
            Specific unit ID to highlight. If provided, plots the individual
            unit trajectory along with group averages.
        graph_options : dict, optional
            Matplotlib customization options including 'title', 'xlabel',
            'ylabel', 'figsize', 'colors', and other styling parameters.

        Returns
        -------
        matplotlib.figure.Figure
            The generated matplotlib figure object.

        Raises
        ------
        ValueError
            If results.data is not set (plotting requires transformed data).

        See Also
        --------
        plot_event_study : Event study visualization for staggered designs.
        """
        from .visualization import prepare_plot_data, plot_results

        if self.data is None:
            raise ValueError("results.data is not set; plotting requires transformed data")

        if isinstance(self.data.attrs, dict) and 'id_mapping' not in self.data.attrs:
            self.data.attrs['id_mapping'] = self._metadata.get('id_mapping')

        tindex = 'tindex'
        ivar = self._metadata['ivar']
        tvar = self._metadata['tvar']
        if isinstance(tvar, str):
            period_labels = {
                int(t): str(int(year))
                for t, year in self.data.groupby(tindex)[tvar].first().items()
            }
        else:
            year_var, quarter_var = tvar[0], tvar[1]
            period_labels = {}
            for t in self.data[tindex].unique():
                row = self.data[self.data[tindex] == t].iloc[0]
                year_val = int(row[year_var])
                quarter_val = int(row[quarter_var])
                period_labels[int(t)] = f"{year_val}q{quarter_val}"

        Tmax = int(self.data[tindex].max())
        tpost1 = int(self._metadata['tpost1'])

        plot_data = prepare_plot_data(
            data=self.data,
            ydot_var='ydot',
            d_var='d_',
            tindex_var='tindex',
            ivar_var=ivar,
            gid=gid,
            tpost1=tpost1,
            Tmax=Tmax,
            period_labels=period_labels,
        )
        fig = plot_results(plot_data, graph_options=graph_options)
        return fig

    def plot_event_study(
        self,
        ref_period: int | None = 0,
        show_ci: bool = True,
        aggregation: str = 'mean',
        include_pre_treatment: bool = True,
        alpha: float = 0.05,
        df_strategy: str = 'conservative',
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        figsize: tuple = (10, 6),
        savefig: str | None = None,
        dpi: int = 150,
        ax: matplotlib.axes.Axes | None = None,
        return_data: bool = False,
        **kwargs
    ):
        """
        Generate an event study diagram for staggered DiD results.

        Aggregates cohort-time specific effects by event time (e = r - g) and
        visualizes dynamic treatment effects relative to a reference period.

        Parameters
        ----------
        ref_period : int or None, optional
            Reference period for normalization (event time). Default is 0
            (first treatment period). If None, no normalization is performed.
        show_ci : bool, optional
            Whether to display confidence interval shading. Default True.
        aggregation : {'mean', 'weighted'}, optional
            Cross-cohort aggregation method. 'mean' computes simple average
            with SE = sqrt(sum(se^2))/n. 'weighted' uses cohort weights with
            SE = sqrt(sum(w^2 * se^2)). Default 'mean'.
        include_pre_treatment : bool, optional
            Whether to include pre-treatment periods (e < 0). Default True.
        alpha : float, optional
            Significance level for confidence intervals. Default 0.05 (95% CI).
        df_strategy : {'conservative', 'weighted', 'fallback'}, optional
            Strategy for selecting degrees of freedom for t-distribution:
            - 'conservative': min(df_g) across cohorts (default)
            - 'weighted': weighted average of df_g
            - 'fallback': n_cohorts - 1
        title : str, optional
            Plot title. Default 'Event Study: Dynamic Treatment Effects'.
        xlabel : str, optional
            X-axis label. Default 'Event Time (Periods Since Treatment)'.
        ylabel : str, optional
            Y-axis label. Default 'Treatment Effect'.
        figsize : tuple of int, optional
            Figure size in inches (width, height). Default (10, 6).
        savefig : str, optional
            File path to save the figure. If provided, saves automatically.
        dpi : int, optional
            Resolution for saved figure. Default 150.
        ax : matplotlib.axes.Axes, optional
            Existing axes object to plot on. If None, creates new figure.
        return_data : bool, optional
            If True, also returns the aggregated event study DataFrame.
            Default False.
        **kwargs
            Additional keyword arguments passed to matplotlib plotting functions.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object.
        ax : matplotlib.axes.Axes
            The matplotlib axes object.
        event_df : pd.DataFrame
            Aggregated event study data. Only returned if return_data=True.

        Raises
        ------
        ValueError
            If called on non-staggered results or if att_by_cohort_time is
            empty or None.

        Notes
        -----
        Confidence intervals use t-distribution rather than normal distribution
        for proper small-sample inference. The degrees of freedom are selected
        based on the df_strategy parameter.

        See Also
        --------
        plot : Residualized outcomes plot for common timing designs.
        """
        if not self.is_staggered:
            raise ValueError("Event study plot requires staggered DiD results")
        
        if self.att_by_cohort_time is None or self.att_by_cohort_time.empty:
            raise ValueError("att_by_cohort_time is empty; cannot create event study plot")
        
        import matplotlib.pyplot as plt
        import numpy as np
        import warnings
        
        # pandas version compatibility: include_groups only available in pandas 2.0+
        _pandas_version = tuple(int(x) for x in pd.__version__.split('.')[:2])
        _groupby_apply_kwargs = {'include_groups': False} if _pandas_version >= (2, 0) else {}
        
        # Aggregate by event_time
        df = self.att_by_cohort_time.copy()
        
        # Auto-compute event_time column if not present
        if 'event_time' not in df.columns:
            df['event_time'] = df['period'] - df['cohort']
        
        # Add source column to distinguish post-treatment effects
        df['_source'] = 'post_treatment'
        
        # Merge pre-treatment effects if available and requested
        has_pre_treatment_data = (
            include_pre_treatment and 
            self.include_pretreatment and 
            self.att_pre_treatment is not None and 
            len(self.att_pre_treatment) > 0
        )
        
        if has_pre_treatment_data:
            pre_df = self.att_pre_treatment.copy()
            pre_df['_source'] = 'pre_treatment'
            
            # Ensure consistent columns for merging
            common_cols = ['cohort', 'period', 'event_time', 'att', 'se', 
                          'ci_lower', 'ci_upper', 't_stat', 'pvalue', 
                          'n_treated', 'n_control', '_source']
            
            # Add is_anchor column if not present in post-treatment
            if 'is_anchor' not in df.columns:
                df['is_anchor'] = False
            if 'is_anchor' in pre_df.columns:
                common_cols.append('is_anchor')
            
            # Select only common columns that exist in both DataFrames
            df_cols = [c for c in common_cols if c in df.columns]
            pre_cols = [c for c in common_cols if c in pre_df.columns]
            
            # Combine pre and post treatment effects
            df = pd.concat([
                df[df_cols],
                pre_df[pre_cols]
            ], ignore_index=True)
        
        # Filter pre-treatment if needed (only from post-treatment data)
        if not include_pre_treatment:
            df = df[df['event_time'] >= 0]
        
        if aggregation == 'weighted' and self.cohort_sizes:
            # Weighted aggregation using aggregate_to_event_time for proper t-distribution CI
            from lwdid.staggered.aggregation import aggregate_to_event_time, event_time_effects_to_dataframe
            
            # Use actual cohort_sizes for proper weight computation
            cohort_sizes = self.cohort_sizes
            
            try:
                watt_effects = aggregate_to_event_time(
                    cohort_time_effects=df,
                    cohort_sizes=cohort_sizes,
                    alpha=alpha,
                    df_strategy=df_strategy,
                    verbose=False,
                )
                event_df = event_time_effects_to_dataframe(watt_effects)
            except (ValueError, KeyError) as e:
                # Fallback to simple weighted aggregation if aggregate_to_event_time fails
                warnings.warn(
                    f"aggregate_to_event_time failed ({e}), using fallback weighted aggregation",
                    UserWarning
                )
                df['weight'] = df['cohort'].map(self.cohort_weights).fillna(0)
                
                def weighted_agg(x):
                    if x['weight'].sum() > 0:
                        att = np.average(x['att'], weights=x['weight'])
                        weights_norm = x['weight'] / x['weight'].sum()
                        se = np.sqrt(np.sum((weights_norm ** 2) * (x['se'] ** 2)))
                    else:
                        att = x['att'].mean()
                        se = np.sqrt((x['se'] ** 2).mean())
                    return pd.Series({'att': att, 'se': se, 'n_cohorts': len(x)})
                
                event_df = df.groupby('event_time').apply(
                    weighted_agg, **_groupby_apply_kwargs
                ).reset_index()
                # Use t-distribution for CI with fallback df
                from scipy.stats import t as t_dist
                df_inference = max(1, len(event_df) - 1)
                t_crit = t_dist.ppf(1 - alpha / 2, df_inference)
                event_df['ci_lower'] = event_df['att'] - t_crit * event_df['se']
                event_df['ci_upper'] = event_df['att'] + t_crit * event_df['se']
                event_df['df_inference'] = df_inference
        else:
            # Simple average aggregation (analytical SE assumes independence across cohorts)
            from scipy.stats import t as t_dist
            
            # Warn about independence assumption for analytical SE
            n_cohorts = len(df['cohort'].unique()) if 'cohort' in df.columns else 0
            if n_cohorts > 1:
                warnings.warn(
                    f"Analytical SE assumes independence across {n_cohorts} cohorts. "
                    f"When cohorts share control units, this may underestimate SE "
                    f"(confidence intervals may be too narrow). "
                    f"Consider using se_method='bootstrap' for more accurate SE.",
                    UserWarning,
                    stacklevel=2
                )
            
            def simple_agg(x):
                att = x['att'].mean()
                # SE aggregation: for independent estimates, variance of mean = Var(ΣX/n) = ΣVar(X)/n²
                # Therefore SE = √(Σse²) / n
                n = len(x)
                se = np.sqrt((x['se'] ** 2).sum()) / n
                # Get df_inference: use min across cohorts (conservative)
                if 'df_inference' in x.columns:
                    valid_dfs = x['df_inference'].dropna()
                    df_inf = int(valid_dfs.min()) if len(valid_dfs) > 0 else max(1, n - 1)
                else:
                    df_inf = max(1, n - 1)
                return pd.Series({'att': att, 'se': se, 'n_cohorts': n, 'df_inference': df_inf})
            
            event_df = df.groupby('event_time').apply(
                simple_agg, **_groupby_apply_kwargs
            ).reset_index()
            
            # Calculate CI using t-distribution (NOT fixed z=1.96)
            event_df['ci_lower'] = event_df.apply(
                lambda row: row['att'] - t_dist.ppf(1 - alpha / 2, row['df_inference']) * row['se'],
                axis=1
            )
            event_df['ci_upper'] = event_df.apply(
                lambda row: row['att'] + t_dist.ppf(1 - alpha / 2, row['df_inference']) * row['se'],
                axis=1
            )
        
        event_df = event_df.sort_values('event_time')
        
        # Normalize to reference period
        if ref_period is not None:
            ref_row = event_df[event_df['event_time'] == ref_period]
            if len(ref_row) > 0:
                ref_att = ref_row['att'].values[0]
                event_df['att'] = event_df['att'] - ref_att
                event_df['ci_lower'] = event_df['ci_lower'] - ref_att
                event_df['ci_upper'] = event_df['ci_upper'] - ref_att
            else:
                warnings.warn(
                    f"Reference period e={ref_period} not found in data. "
                    f"Available event times: {sorted(event_df['event_time'].unique())}. "
                    f"Skipping normalization.",
                    UserWarning
                )
        
        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        # Get colors from kwargs or use defaults
        pre_treatment_color = kwargs.get('pre_treatment_color', 'gray')
        post_treatment_color = kwargs.get('post_treatment_color', 'blue')
        anchor_line = kwargs.get('anchor_line', True)
        
        # Reference lines
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=-0.5, color='red', linestyle='--', alpha=0.7, linewidth=1.5, 
                   label='Treatment Start')
        
        # Add anchor point line at e=-1 if pre-treatment data is shown
        if has_pre_treatment_data and anchor_line:
            ax.axvline(x=-1, color='darkgray', linestyle=':', alpha=0.5, linewidth=1,
                       label='Anchor Point (e=-1)')
        
        # Separate pre and post treatment for different styling
        pre_mask = event_df['event_time'] < 0
        post_mask = event_df['event_time'] >= 0
        
        pre_events = event_df[pre_mask].copy()
        post_events = event_df[post_mask].copy()
        
        # Confidence interval shading
        if show_ci:
            ci_level = int((1 - alpha) * 100)
            
            # Post-treatment CI (blue)
            if len(post_events) > 0:
                ax.fill_between(
                    post_events['event_time'],
                    post_events['ci_lower'],
                    post_events['ci_upper'],
                    alpha=0.2, color=post_treatment_color, label=f'{ci_level}% CI (Post)'
                )
            
            # Pre-treatment CI (gray) - only if we have pre-treatment data
            if len(pre_events) > 0 and has_pre_treatment_data:
                ax.fill_between(
                    pre_events['event_time'],
                    pre_events['ci_lower'],
                    pre_events['ci_upper'],
                    alpha=0.15, color=pre_treatment_color, label=f'{ci_level}% CI (Pre)'
                )
        
        # Point estimates - post-treatment (blue)
        if len(post_events) > 0:
            ax.scatter(post_events['event_time'], post_events['att'], 
                      color=post_treatment_color, s=60, zorder=5, label='Post-treatment')
            ax.plot(post_events['event_time'], post_events['att'], 
                   color=post_treatment_color, alpha=0.7, linewidth=1.5)
        
        # Point estimates - pre-treatment (gray)
        if len(pre_events) > 0:
            # Mark anchor point differently
            anchor_mask = pre_events.get('is_anchor', pd.Series([False] * len(pre_events)))
            if anchor_mask.any():
                anchor_events = pre_events[anchor_mask]
                non_anchor_events = pre_events[~anchor_mask]
                
                # Non-anchor pre-treatment points
                if len(non_anchor_events) > 0:
                    ax.scatter(non_anchor_events['event_time'], non_anchor_events['att'],
                              color=pre_treatment_color, s=60, zorder=5, 
                              marker='o', label='Pre-treatment')
                    ax.plot(non_anchor_events['event_time'], non_anchor_events['att'],
                           color=pre_treatment_color, alpha=0.7, linewidth=1.5, linestyle='--')
                
                # Anchor point (diamond marker)
                if len(anchor_events) > 0:
                    ax.scatter(anchor_events['event_time'], anchor_events['att'],
                              color=pre_treatment_color, s=100, zorder=6,
                              marker='D', edgecolors='black', linewidths=1,
                              label='Anchor (e=-1)')
            else:
                # No anchor info, plot all pre-treatment the same
                ax.scatter(pre_events['event_time'], pre_events['att'],
                          color=pre_treatment_color, s=60, zorder=5,
                          marker='o', label='Pre-treatment')
                ax.plot(pre_events['event_time'], pre_events['att'],
                       color=pre_treatment_color, alpha=0.7, linewidth=1.5, linestyle='--')
        
        # Connect pre and post treatment with a line if both exist
        if len(pre_events) > 0 and len(post_events) > 0:
            # Get the last pre-treatment and first post-treatment points
            last_pre = pre_events.loc[pre_events['event_time'].idxmax()]
            first_post = post_events.loc[post_events['event_time'].idxmin()]
            ax.plot([last_pre['event_time'], first_post['event_time']],
                   [last_pre['att'], first_post['att']],
                   color='gray', alpha=0.4, linewidth=1, linestyle=':')
        
        # Labels and title
        ax.set_xlabel(xlabel or 'Event Time (Periods Since Treatment)', fontsize=11)
        ax.set_ylabel(ylabel or 'Treatment Effect', fontsize=11)
        ax.set_title(title or 'Event Study: Dynamic Treatment Effects', fontsize=13, fontweight='bold')
        
        # X-axis ticks as integers
        ax.set_xticks(sorted(event_df['event_time'].astype(int).unique()))
        
        # Legend
        ax.legend(loc='best', framealpha=0.9)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle=':')
        
        plt.tight_layout()
        
        # Save
        if savefig:
            plt.savefig(savefig, dpi=dpi, bbox_inches='tight')
        
        if return_data:
            return fig, ax, event_df
        return fig, ax

    def to_excel(self, path: str):
        """
        Export estimation results to an Excel file.

        For common timing designs, creates a workbook with Summary sheet
        containing ATT, SE, t-statistic, p-value, CI bounds, and sample sizes.
        If period-specific effects are available, includes a ByPeriod sheet.
        For staggered designs, dispatches to to_excel_staggered().

        Parameters
        ----------
        path : str
            File path for the Excel output (.xlsx extension recommended).

        See Also
        --------
        to_csv : Export period-specific effects to CSV format.
        to_latex : Export results to LaTeX table format.
        """
        if self.is_staggered:
            return self.to_excel_staggered(path)
        
        summary_rows = [
            {"Statistic": "ATT", "Value": self.att},
            {"Statistic": "SE", "Value": self.se_att},
            {"Statistic": "t", "Value": self.t_stat},
            {"Statistic": "p", "Value": self.pvalue},
            {"Statistic": "CI_lower", "Value": self.ci_lower},
            {"Statistic": "CI_upper", "Value": self.ci_upper},
            {"Statistic": "N", "Value": self.nobs},
            {"Statistic": "N_treated", "Value": self.n_treated},
            {"Statistic": "N_control", "Value": self.n_control},
            {"Statistic": "vce", "Value": self.vce_type},
            {"Statistic": "rolling", "Value": self.rolling},
        ]
        if self.ri_pvalue is not None:
            summary_rows.extend([
                {"Statistic": "ri_pvalue", "Value": self.ri_pvalue},
                {"Statistic": "ri_seed", "Value": self.ri_seed},
                {"Statistic": "rireps", "Value": self.rireps},
            ])
        df_summary = pd.DataFrame(summary_rows)

        with pd.ExcelWriter(path) as writer:
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            if isinstance(self.att_by_period, pd.DataFrame) and not self.att_by_period.empty:
                self.att_by_period.to_excel(writer, sheet_name='ByPeriod', index=False)
            if self.ri_pvalue is not None:
                df_ri = pd.DataFrame([
                    {"Parameter": "ri_pvalue", "Value": self.ri_pvalue},
                    {"Parameter": "ri_seed", "Value": self.ri_seed},
                    {"Parameter": "rireps", "Value": self.rireps},
                    {"Parameter": "ATT_obs", "Value": self.att},
                ])
                df_ri.to_excel(writer, sheet_name='RI', index=False)

    def to_excel_staggered(self, path: str):
        """
        Export staggered DiD results to a multi-sheet Excel file.

        Creates an Excel workbook with sheets tailored to the aggregation level.
        The Summary sheet is always included. Additional sheets depend on the
        aggregate parameter used during estimation.

        Parameters
        ----------
        path : str
            File path for the Excel output (.xlsx extension required).

        Raises
        ------
        ValueError
            If called on non-staggered estimation results.
        ImportError
            If openpyxl package is not installed.

        Notes
        -----
        Sheet structure varies by aggregation level:

        - aggregate='overall': Summary, Overall, Cohort, CohortTime, Weights,
          Metadata
        - aggregate='cohort': Summary, Cohort, CohortTime, Weights, Metadata
        - aggregate='none': Summary, CohortTime, Metadata
        """
        if not self.is_staggered:
            raise ValueError("to_excel_staggered requires staggered DiD results")
        
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            raise ImportError(
                "to_excel_staggered requires openpyxl. "
                "Install it with: pip install openpyxl"
            )
        
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            # === Sheet 1: Summary (always present) ===
            summary_rows = [
                {"Item": "Estimation Type", "Value": "Staggered DiD"},
                {"Item": "Treatment Cohorts", "Value": ', '.join(map(str, self.cohorts))},
                {"Item": "Number of Cohorts", "Value": len(self.cohorts)},
                {"Item": "N Treated Units", "Value": self.n_treated},
                {"Item": "N Control Units", "Value": self.n_control},
                {"Item": "N Never Treated", "Value": self.n_never_treated},
                {"Item": "Control Group", "Value": self.control_group_used},
                {"Item": "Transformation", "Value": self.rolling},
                {"Item": "Estimator", "Value": self.estimator},
                {"Item": "Aggregation", "Value": self.aggregate},
                {"Item": "VCE Type", "Value": self.vce_type},
            ]
            
            # Only add overall effect info when aggregate='overall'
            if self.att_overall is not None:
                summary_rows.extend([
                    {"Item": "Overall ATT (τ_ω)", "Value": self.att_overall},
                    {"Item": "Overall SE", "Value": self.se_overall},
                    {"Item": "Overall CI Lower", "Value": self.ci_overall_lower},
                    {"Item": "Overall CI Upper", "Value": self.ci_overall_upper},
                ])
            
            df_summary = pd.DataFrame(summary_rows)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # === Sheet 2: Overall Effect (only when aggregate='overall') ===
            if self.att_overall is not None:
                df_overall = pd.DataFrame([{
                    'att_overall': self.att_overall,
                    'se': self.se_overall,
                    't_stat': self.t_stat_overall,
                    'pvalue': self.pvalue_overall,
                    'ci_lower': self.ci_overall_lower,
                    'ci_upper': self.ci_overall_upper,
                }])
                df_overall.to_excel(writer, sheet_name='Overall', index=False)
            
            # === Sheet 3: Cohort Effects (only when aggregate∈{'cohort','overall'}) ===
            if self.att_by_cohort is not None and not self.att_by_cohort.empty:
                self.att_by_cohort.to_excel(writer, sheet_name='Cohort', index=False)
            
            # === Sheet 4: Cohort-Time Effects (always present) ===
            if self.att_by_cohort_time is not None and not self.att_by_cohort_time.empty:
                self.att_by_cohort_time.to_excel(writer, sheet_name='CohortTime', index=False)
            
            # === Sheet 5: Cohort Weights (only when aggregate∈{'cohort','overall'}) ===
            if self.cohort_weights:
                df_weights = pd.DataFrame([
                    {'cohort': g, 'weight': w, 'n_units': self.cohort_sizes.get(g, None)}
                    for g, w in sorted(self.cohort_weights.items())
                ])
                df_weights.to_excel(writer, sheet_name='Weights', index=False)
            
            # === Sheet 6: Metadata (always present) ===
            metadata_rows = [
                {"Parameter": "is_staggered", "Value": True},
                {"Parameter": "control_group", "Value": self.control_group},
                {"Parameter": "control_group_used", "Value": self.control_group_used},
                {"Parameter": "aggregate", "Value": self.aggregate},
                {"Parameter": "estimator", "Value": self.estimator},
                {"Parameter": "rolling", "Value": self.rolling},
                {"Parameter": "vce_type", "Value": self.vce_type},
                {"Parameter": "n_never_treated", "Value": self.n_never_treated},
            ]
            df_metadata = pd.DataFrame(metadata_rows)
            df_metadata.to_excel(writer, sheet_name='Metadata', index=False)

    def to_csv(self, path: str):
        """
        Export period-specific treatment effects to a CSV file.

        Parameters
        ----------
        path : str
            File path for the CSV output.

        Raises
        ------
        ValueError
            If att_by_period is not available (None or empty DataFrame).

        See Also
        --------
        to_excel : Export comprehensive results to Excel format.
        """
        if not isinstance(self.att_by_period, pd.DataFrame) or self.att_by_period.empty:
            raise ValueError("att_by_period is not available for CSV export")
        self.att_by_period.to_csv(path, index=False)

    def to_latex(self, path: str):
        """
        Export estimation results to a LaTeX table file.

        Generates a LaTeX document containing summary statistics (ATT, SE,
        t-statistic, p-value, CI bounds, sample sizes) and period-specific
        effects if available.

        Parameters
        ----------
        path : str
            File path for the LaTeX output (.tex extension recommended).

        See Also
        --------
        to_excel : Export comprehensive results to Excel format.
        to_csv : Export period-specific effects to CSV format.
        """
        summary_rows = [
            ["ATT", f"{self.att:.6g}"],
            ["SE", f"{self.se_att:.6g}"],
            ["t", f"{self.t_stat:.6g}"],
            ["p", f"{self.pvalue:.6g}"],
            ["CI_lower", f"{self.ci_lower:.6g}"],
            ["CI_upper", f"{self.ci_upper:.6g}"],
            ["N", f"{self.nobs}"],
            ["N_treated", f"{self.n_treated}"],
            ["N_control", f"{self.n_control}"],
            ["vce", f"{self.vce_type}"],
            ["rolling", f"{self.rolling}"],
        ]
        if self.ri_pvalue is not None:
            summary_rows.extend([
                ["ri_pvalue", f"{self.ri_pvalue:.6g}"],
                ["ri_seed", f"{self.ri_seed}"],
                ["rireps", f"{self.rireps}"],
            ])
        df_summary = pd.DataFrame(summary_rows, columns=["Statistic", "Value"])

        content = []
        content.append(df_summary.to_latex(index=False, escape=True))
        if isinstance(self.att_by_period, pd.DataFrame) and not self.att_by_period.empty:
            content.append(self.att_by_period.to_latex(index=False, escape=True, na_rep='--'))
        with open(path, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(content))


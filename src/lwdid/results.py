"""
Results Container Module

Defines the LWDIDResults class for storing, displaying, and exporting estimation
results.

"""

from typing import Any, Dict, Optional, Union

import pandas as pd


class LWDIDResults:
    """
    Container for lwdid estimation results
    
    This class stores all estimation outputs from the lwdid() function and provides
    methods for displaying, visualizing, and exporting results. All core attributes
    are read-only properties to ensure result integrity.
    
    Supports both common timing and staggered DiD settings.
    
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
    ci_lower, ci_upper : float
        Lower and upper bounds of 95% confidence interval.
    nobs : int
        Number of observations in the regression.
    n_treated, n_control : int
        Number of treated and control units.
    K : int
        Last pre-treatment period index.
    tpost1 : int
        First post-treatment period index.
    df_resid : int
        Residual degrees of freedom from the main regression.
    df_inference : int
        Degrees of freedom used for inference. For cluster-robust standard errors,
        this is G - 1 (number of clusters minus 1). For other variance estimators,
        this equals df_resid. This is the df value displayed in summary() and used
        for computing t-statistics and confidence intervals.
    rolling : str
        Transformation method used ('demean', 'detrend', 'demeanq', or 'detrendq').
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
        Number of valid (successful) RI replications. Only available if ri=True.
    ri_failed : int or None
        Number of failed RI replications. Only available if ri=True.
    data : pd.DataFrame
        Transformed data used for regression (internal attribute, not a public property).
        Contains the rolling-transformed outcome variable and other regression variables.
        If the original ivar was string type, this DataFrame contains numeric IDs (1, 2, 3, ...).
        The mapping between original and numeric IDs is stored in data.attrs['id_mapping'].

        Access via: results.data (not recommended for typical use)

        String ID mapping (if applicable):

        - results.data.attrs['id_mapping']['original_to_numeric']: dict mapping original IDs to numeric
        - results.data.attrs['id_mapping']['numeric_to_original']: dict mapping numeric to original IDs

        Example:
        
        If ivar='state' with values ['CA', 'TX', 'NY'], they are encoded as 1, 2, 3.
        results.data.attrs['id_mapping'] = {
            'original_to_numeric': {'CA': 1, 'TX': 2, 'NY': 3},
            'numeric_to_original': {1: 'CA', 2: 'TX', 3: 'NY'}
        }

    Staggered-specific attributes (only available when is_staggered=True):
    
    is_staggered : bool
        Whether this is a staggered DiD estimation.
    cohorts : List[int]
        Sorted list of treatment cohorts (first treatment periods).
    cohort_sizes : Dict[int, int]
        Number of units in each cohort.
    att_by_cohort_time : pd.DataFrame
        (g,r)-specific ATT estimates with columns: cohort, period, event_time,
        att, se, ci_lower, ci_upper, t_stat, pvalue, n_treated, n_control.
    att_by_cohort : pd.DataFrame or None
        Cohort-specific ATT estimates (if aggregate='cohort' or 'overall').
    att_overall : float or None
        Overall weighted ATT (if aggregate='overall').
    se_overall : float or None
        Standard error of overall ATT.
    cohort_weights : Dict[int, float]
        Cohort weights used for overall effect (omega_g = N_g / N_treat).
    control_group : str
        User-specified control group strategy.
    control_group_used : str
        Actual control group strategy used (may differ due to auto-switching).
    aggregate : str
        Aggregation level ('none', 'cohort', 'overall').
    estimator : str
        Estimation method ('ra', 'ipwra', 'psm').

    Methods
    -------
    summary() : str
        Returns formatted results summary string.
    plot(gid=None, graph_options=None) : matplotlib.Figure
        Generates plot of residualized outcomes (control vs. treated).
    to_excel(path) : None
        Exports results to Excel file with multiple sheets.
    to_csv(path) : None
        Exports period-specific effects to CSV.
    to_latex(path) : None
        Exports results to LaTeX table format.
    
    Examples
    --------
    >>> from lwdid import lwdid
    >>> import pandas as pd
    >>> data = pd.read_csv('smoking.csv')
    >>> results = lwdid(data, y='lcigsale', d='treated', ivar='state',
    ...                 tvar='year', post='post', rolling='demean')
    >>> 
    >>> # View summary
    >>> print(results.summary())
    >>> 
    >>> # Access attributes
    >>> print(f"ATT: {results.att:.4f}")
    >>> print(f"SE: {results.se_att:.4f}")
    >>> print(f"p-value: {results.pvalue:.3f}")
    >>> 
    >>> # Period-specific effects
    >>> print(results.att_by_period)
    >>> 
    >>> # Visualization
    >>> fig = results.plot()
    >>> fig.savefig('did_plot.png')
    >>> 
    >>> # Export
    >>> results.to_excel('results.xlsx')
    """
    
    def __init__(
        self,
        results_dict: Dict[str, Any],
        metadata: Dict[str, Any],
        att_by_period: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize results object
        
        Parameters:
            results_dict: Results from estimate_att()
            metadata: Metadata from validate_and_prepare_data()
            att_by_period: Period-specific effects (optional)
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
        
        self._resid = results_dict['resid']
        self._metadata = metadata

        self._ri_pvalue: Optional[float] = None
        self._ri_seed: Optional[int] = None
        self._rireps: Optional[int] = None
        self._ri_method: Optional[str] = None
        self._ri_valid: Optional[int] = None
        self._ri_failed: Optional[int] = None
        self._data: Optional[pd.DataFrame] = None
        
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
        """ATT estimate"""
        return self._att
    
    @property
    def se_att(self) -> float:
        """Standard error"""
        return self._se_att
    
    @property
    def t_stat(self) -> float:
        """t-statistic"""
        return self._t_stat
    
    @property
    def pvalue(self) -> float:
        """p-value"""
        return self._pvalue
    
    @property
    def ci_lower(self) -> float:
        """95% CI lower bound"""
        return self._ci_lower
    
    @property
    def ci_upper(self) -> float:
        """95% CI upper bound"""
        return self._ci_upper
    
    @property
    def nobs(self) -> int:
        """Number of observations"""
        return self._nobs
    
    @property
    def n_treated(self) -> int:
        """Number of treated units"""
        return self._n_treated
    
    @property
    def n_control(self) -> int:
        """Number of control units"""
        return self._n_control
    
    @property
    def df_resid(self) -> int:
        """Residual degrees of freedom"""
        return self._df_resid

    @property
    def df_inference(self) -> int:
        """Degrees of freedom for inference (G-1 for cluster, df_resid otherwise)"""
        return self._df_inference

    @property
    def K(self) -> int:
        """Last pre-treatment period"""
        return self._K
    
    @property
    def tpost1(self) -> int:
        """First post-treatment period"""
        return self._tpost1
    
    @property
    def cmd(self) -> str:
        """Command name"""
        return self._cmd
    
    @property
    def depvar(self) -> str:
        """Dependent variable"""
        return self._depvar
    
    @property
    def rolling(self) -> str:
        """Transformation method"""
        return self._rolling
    
    @property
    def vce_type(self) -> Optional[str]:
        """Variance estimator"""
        return self._vce_type
    
    @property
    def cluster_var(self) -> Optional[str]:
        """Clustering variable"""
        return self._cluster_var
    
    @property
    def n_clusters(self) -> Optional[int]:
        """Number of clusters"""
        return self._n_clusters
    
    @property
    def controls_used(self) -> bool:
        """Controls included"""
        return self._controls_used
    
    @property
    def controls(self) -> list:
        """Control variables"""
        return list(self._controls)
    
    @property
    def params(self):
        """Coefficient vector"""
        return self._params
    
    @property
    def bse(self):
        """Standard errors"""
        return self._bse
    
    @property
    def vcov(self):
        """Variance-covariance matrix"""
        return self._vcov
    
    @property
    def att_by_period(self) -> Optional[pd.DataFrame]:
        """Period-specific effects (returns copy)"""
        if self._att_by_period is None:
            return None
        return self._att_by_period.copy()
    
    @property
    def ri_pvalue(self) -> Optional[float]:
        """RI p-value"""
        return self._ri_pvalue
    
    @ri_pvalue.setter
    def ri_pvalue(self, value: Optional[float]) -> None:
        self._ri_pvalue = value
    
    @property
    def ri_seed(self) -> Optional[int]:
        """RI seed"""
        return self._ri_seed
    
    @ri_seed.setter
    def ri_seed(self, value: Optional[int]) -> None:
        self._ri_seed = value
    
    @property
    def rireps(self) -> Optional[int]:
        """RI replications"""
        return self._rireps
    
    @rireps.setter
    def rireps(self, value: Optional[int]) -> None:
        self._rireps = value

    @property
    def ri_method(self) -> Optional[str]:
        """RI method"""
        return self._ri_method

    @ri_method.setter
    def ri_method(self, value: Optional[str]) -> None:
        self._ri_method = value

    @property
    def ri_valid(self) -> Optional[int]:
        """Valid RI replications"""
        return self._ri_valid

    @ri_valid.setter
    def ri_valid(self, value: Optional[int]) -> None:
        self._ri_valid = value

    @property
    def ri_failed(self) -> Optional[int]:
        """Failed RI replications"""
        return self._ri_failed

    @ri_failed.setter
    def ri_failed(self, value: Optional[int]) -> None:
        self._ri_failed = value

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Transformed data"""
        return self._data
    
    @data.setter
    def data(self, value: Optional[pd.DataFrame]) -> None:
        self._data = value
    
    # === Staggered-specific Properties ===
    
    @property
    def is_staggered(self) -> bool:
        """Whether this is a staggered DiD estimation"""
        return self._is_staggered
    
    @property
    def cohorts(self) -> list:
        """List of treatment cohorts (first treatment periods)"""
        return list(self._cohorts)
    
    @property
    def cohort_sizes(self) -> dict:
        """Number of units in each cohort"""
        return dict(self._cohort_sizes)
    
    @property
    def att_by_cohort_time(self) -> Optional[pd.DataFrame]:
        """(g,r)-specific ATT estimates (returns copy)"""
        if self._att_by_cohort_time is None:
            return None
        return self._att_by_cohort_time.copy()
    
    @property
    def att_by_cohort(self) -> Optional[pd.DataFrame]:
        """Cohort-specific ATT estimates (returns copy)"""
        if self._att_by_cohort is None:
            return None
        return self._att_by_cohort.copy()
    
    @property
    def att_overall(self) -> Optional[float]:
        """Overall weighted ATT (if aggregate='overall')"""
        return self._att_overall
    
    @property
    def se_overall(self) -> Optional[float]:
        """Standard error of overall ATT"""
        return self._se_overall
    
    @property
    def ci_overall_lower(self) -> Optional[float]:
        """95% CI lower bound for overall ATT"""
        return self._ci_overall_lower
    
    @property
    def ci_overall_upper(self) -> Optional[float]:
        """95% CI upper bound for overall ATT"""
        return self._ci_overall_upper
    
    @property
    def t_stat_overall(self) -> Optional[float]:
        """t-statistic for overall ATT"""
        return self._t_stat_overall
    
    @property
    def pvalue_overall(self) -> Optional[float]:
        """p-value for overall ATT"""
        return self._pvalue_overall
    
    @property
    def cohort_weights(self) -> dict:
        """Cohort weights for overall effect (omega_g = N_g / N_treat)"""
        return dict(self._cohort_weights)
    
    @property
    def control_group(self) -> Optional[str]:
        """User-specified control group strategy"""
        return self._control_group
    
    @property
    def control_group_used(self) -> Optional[str]:
        """Actual control group strategy used (may differ due to auto-switching)"""
        return self._control_group_used
    
    @property
    def aggregate(self) -> Optional[str]:
        """Aggregation level ('none', 'cohort', 'overall')"""
        return self._aggregate
    
    @property
    def estimator(self) -> Optional[str]:
        """Estimation method ('ra', 'ipwra', 'psm')"""
        return self._estimator
    
    @property
    def n_never_treated(self) -> Optional[int]:
        """Never treated units count (staggered mode returns value, otherwise None)"""
        return self._n_never_treated
    
    def summary(self) -> str:
        """
        Formatted results summary
        
        For staggered results, dispatches to summary_staggered();
        for common timing results, uses original logic.
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
        Print formatted summary for staggered estimation results
        
        Returns
        -------
        str
            Formatted results summary string
            
        Raises
        ------
        ValueError
            If not a staggered estimation result
            
        Notes
        -----
        - When aggregate='none', does not show Overall and Cohort effects
        - When aggregate='cohort', does not show Overall effect
        - If control group was auto-switched, shows notification
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
        output.append(sep_line)
        
        return "\n".join(output)
    
    def __repr__(self) -> str:
        if self.is_staggered:
            if self.att_overall is not None:
                return (
                    f"LWDIDResults(staggered=True, att_overall={self.att_overall:.4f}, "
                    f"se={self.se_overall:.4f}, cohorts={len(self.cohorts)}, "
                    f"N_treated={self.n_treated}, N_control={self.n_control})"
                )
            else:
                return (
                    f"LWDIDResults(staggered=True, cohorts={len(self.cohorts)}, "
                    f"aggregate='{self.aggregate}', N_treated={self.n_treated})"
                )
        else:
            return (
                f"LWDIDResults(att={self.att:.4f}, se={self.se_att:.4f}, "
                f"method='{self.rolling}', N={self.nobs})"
            )
    
    def __str__(self) -> str:
        return self.summary()

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    def plot(self, gid: Optional[Union[str, int]] = None, graph_options: Optional[dict] = None):
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
        ref_period: Optional[int] = 0,
        show_ci: bool = True,
        aggregation: str = 'mean',
        include_pre_treatment: bool = True,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figsize: tuple = (10, 6),
        savefig: Optional[str] = None,
        dpi: int = 150,
        ax: Optional['matplotlib.axes.Axes'] = None,
        return_data: bool = False,
        **kwargs
    ):
        """
        Plot Event Study diagram
        
        Aggregates (g,r) effects by event time (e = r - g) and plots dynamic treatment effects.
        
        Parameters
        ----------
        ref_period : int or None, default=0
            Reference period (event time). Default is 0 (first treatment period).
            If None, no normalization is performed.
            
        show_ci : bool, default=True
            Whether to show confidence interval shading.
            
        aggregation : {'mean', 'weighted'}
            Cross-cohort aggregation method:
            - 'mean': Simple average, SE via √(Σse²/n)
            - 'weighted': Weighted by cohort weights (uses cohort_weights)
            
        include_pre_treatment : bool, default=True
            Whether to include pre-treatment periods (e<0).
            
        title : str, optional
            Plot title, default 'Event Study: Dynamic Treatment Effects'
            
        xlabel : str, optional
            X-axis label, default 'Event Time (Periods Since Treatment)'
            
        ylabel : str, optional
            Y-axis label, default 'Treatment Effect'
            
        figsize : tuple, default=(10, 6)
            Figure size
            
        savefig : str, optional
            Save path. If provided, saves figure to this path.
            
        dpi : int, default=150
            DPI for saved figure
            
        ax : matplotlib.axes.Axes, optional
            Existing axes object. If provided, plots on this axes.
            
        return_data : bool, default=False
            If True, return the aggregated event study DataFrame along with plot objects.
            
        **kwargs
            Additional parameters passed to matplotlib
            
        Returns
        -------
        tuple
            If return_data=False: (fig, ax) matplotlib objects
            If return_data=True: (fig, ax, event_df) where event_df is the aggregated DataFrame
            
        Raises
        ------
        ValueError
            If not a staggered result or att_by_cohort_time is empty
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
        
        # Filter pre-treatment if needed
        if not include_pre_treatment:
            df = df[df['event_time'] >= 0]
        
        if aggregation == 'weighted' and self.cohort_weights:
            # Weighted aggregation
            df['weight'] = df['cohort'].map(self.cohort_weights).fillna(0)
            
            def weighted_agg(x):
                if x['weight'].sum() > 0:
                    att = np.average(x['att'], weights=x['weight'])
                    # Weighted SE: √(Σ ω² × se²), weights normalized
                    weights_norm = x['weight'] / x['weight'].sum()
                    se = np.sqrt(np.sum((weights_norm ** 2) * (x['se'] ** 2)))
                else:
                    att = x['att'].mean()
                    se = np.sqrt((x['se'] ** 2).mean())
                return pd.Series({'att': att, 'se': se, 'n_cohorts': len(x)})
            
            event_df = df.groupby('event_time').apply(
                weighted_agg, **_groupby_apply_kwargs
            ).reset_index()
        else:
            # Simple average
            def simple_agg(x):
                att = x['att'].mean()
                # SE aggregation: for independent estimates, variance of mean = Var(ΣX/n) = ΣVar(X)/n²
                # Therefore SE = √(Σse²) / n
                n = len(x)
                se = np.sqrt((x['se'] ** 2).sum()) / n
                return pd.Series({'att': att, 'se': se, 'n_cohorts': n})
            
            event_df = df.groupby('event_time').apply(
                simple_agg, **_groupby_apply_kwargs
            ).reset_index()
        
        event_df = event_df.sort_values('event_time')
        
        # Calculate confidence intervals
        event_df['ci_lower'] = event_df['att'] - 1.96 * event_df['se']
        event_df['ci_upper'] = event_df['att'] + 1.96 * event_df['se']
        
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
        
        # Reference lines
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=-0.5, color='red', linestyle='--', alpha=0.7, linewidth=1.5, 
                   label='Treatment Start')
        
        # Confidence interval shading
        if show_ci:
            ax.fill_between(
                event_df['event_time'],
                event_df['ci_lower'],
                event_df['ci_upper'],
                alpha=0.2, color='blue', label='95% CI'
            )
        
        # Point estimates
        ax.scatter(event_df['event_time'], event_df['att'], color='blue', s=60, zorder=5)
        ax.plot(event_df['event_time'], event_df['att'], color='blue', alpha=0.7, linewidth=1.5)
        
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
        Export results to Excel file
        
        For staggered results, dispatches to to_excel_staggered();
        for common timing results, uses original logic.
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
        Export staggered results to Excel file (multi-sheet format)
        
        Sheet structure varies by aggregate level:
        
        | aggregate | sheets included |
        |-----------|-----------------|
        | 'overall' | Summary, Overall, Cohort, CohortTime, Weights, Metadata |
        | 'cohort'  | Summary, Cohort, CohortTime, Weights, Metadata |
        | 'none'    | Summary, CohortTime, Metadata |
        
        Parameters
        ----------
        path : str
            Excel file save path (.xlsx)
            
        Raises
        ------
        ValueError
            If not a staggered estimation result
        ImportError
            If openpyxl is not installed
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
        if not isinstance(self.att_by_period, pd.DataFrame) or self.att_by_period.empty:
            raise ValueError("att_by_period is not available for CSV export")
        self.att_by_period.to_csv(path, index=False)

    def to_latex(self, path: str):
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


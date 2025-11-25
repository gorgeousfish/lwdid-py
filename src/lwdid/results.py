"""
Results Container Module

Defines the LWDIDResults class for storing, displaying, and exporting estimation
results.

Authors: Xuanyu Cai, Wenli Xu
"""

from typing import Any, Dict, Optional, Union

import pandas as pd


class LWDIDResults:
    """
    Container for lwdid estimation results
    
    This class stores all estimation outputs from the lwdid() function and provides
    methods for displaying, visualizing, and exporting results. All core attributes
    are read-only properties to ensure result integrity.
    
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
    
    def summary(self) -> str:
        """
        Formatted results summary
        """
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
    
    def __repr__(self) -> str:
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

    def to_excel(self, path: str):
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


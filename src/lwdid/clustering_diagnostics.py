"""
Clustering diagnostics and recommendations for difference-in-differences.

This module provides tools for analyzing clustering structure in panel data
and recommending appropriate clustering levels for standard error estimation
in difference-in-differences analysis.

When the policy or treatment varies at a level higher than the unit of
observation, standard errors should be clustered at the policy variation
level. This module helps identify the appropriate clustering level by:

- Analyzing hierarchical relationships between potential clustering variables
- Detecting the level at which treatment assignment varies
- Recommending clustering variables with sufficient cluster counts
- Checking consistency between clustering choice and treatment variation

For reliable cluster-robust inference, a minimum of 20-30 clusters is
generally recommended. When clusters are fewer, wild cluster bootstrap
methods provide more accurate inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# Enumeration Types
# =============================================================================

class ClusteringLevel(Enum):
    """
    Relative level of clustering variable to unit variable.
    
    Attributes
    ----------
    LOWER : str
        Cluster variable is at a lower level than unit (invalid for clustering).
        Example: sub-unit ID when unit is individual.
    SAME : str
        Cluster variable is at the same level as unit.
        Example: individual ID when unit is individual.
    HIGHER : str
        Cluster variable is at a higher level than unit (recommended).
        Example: state when unit is county.
    """
    LOWER = "lower"
    SAME = "same"
    HIGHER = "higher"


class ClusteringWarningLevel(Enum):
    """
    Severity level for clustering warnings.
    
    Attributes
    ----------
    INFO : str
        Informational message, no action required.
    WARNING : str
        Warning that may affect inference reliability.
    ERROR : str
        Critical issue that prevents valid inference.
    """
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ClusterVarStats:
    """
    Statistics for a single potential clustering variable.
    
    This class holds comprehensive statistics about a clustering variable,
    including cluster counts, size distributions, and validity indicators.
    
    Attributes
    ----------
    var_name : str
        Name of the clustering variable.
    n_clusters : int
        Total number of unique clusters.
    n_treated_clusters : int
        Number of clusters containing treated units.
    n_control_clusters : int
        Number of clusters containing only control units.
    min_cluster_size : int
        Minimum number of observations in any cluster.
    max_cluster_size : int
        Maximum number of observations in any cluster.
    mean_cluster_size : float
        Mean cluster size.
    median_cluster_size : float
        Median cluster size.
    cluster_size_cv : float
        Coefficient of variation of cluster sizes (std/mean).
    level_relative_to_unit : ClusteringLevel
        Whether cluster is at higher/same/lower level than unit.
    units_per_cluster : float
        Average number of unique units per cluster.
    is_nested_in_unit : bool
        True if cluster varies within unit (invalid for clustering).
    treatment_varies_within_cluster : bool
        True if treatment status varies within clusters.
    n_clusters_with_treatment_variation : int
        Number of clusters with within-cluster treatment variation.
    
    Properties
    ----------
    is_valid_cluster : bool
        Whether this is a valid clustering variable.
    is_recommended : bool
        Whether this clustering level is recommended.
    reliability_score : float
        Score indicating reliability of cluster-robust inference (0-1).
    """
    var_name: str
    n_clusters: int
    n_treated_clusters: int
    n_control_clusters: int
    min_cluster_size: int
    max_cluster_size: int
    mean_cluster_size: float
    median_cluster_size: float
    cluster_size_cv: float
    level_relative_to_unit: ClusteringLevel
    units_per_cluster: float
    is_nested_in_unit: bool
    treatment_varies_within_cluster: bool
    n_clusters_with_treatment_variation: int = 0
    
    @property
    def is_valid_cluster(self) -> bool:
        """
        Whether this is a valid clustering variable.
        
        A valid clustering variable must:
        1. Not be nested within units (each unit belongs to one cluster)
        2. Have at least 2 clusters
        3. Not be at a lower level than the unit variable
        
        Returns
        -------
        bool
            True if valid for clustering.
        """
        return (
            not self.is_nested_in_unit and
            self.n_clusters >= 2 and
            self.level_relative_to_unit != ClusteringLevel.LOWER
        )
    
    @property
    def is_recommended(self) -> bool:
        """
        Whether this clustering level is recommended.
        
        A recommended clustering variable must:
        1. Be valid (see is_valid_cluster)
        2. Have at least 20 clusters for reliable inference
        3. Treatment should not vary within clusters
        
        Returns
        -------
        bool
            True if recommended for clustering.
        """
        return (
            self.is_valid_cluster and
            self.n_clusters >= 20 and
            not self.treatment_varies_within_cluster
        )
    
    @property
    def reliability_score(self) -> float:
        """
        Score indicating reliability of cluster-robust inference (0-1).
        
        Based on:
        - Number of clusters (more is better, saturates at 50)
        - Balance of treated/control clusters
        - Cluster size variation (less is better)
        
        Returns
        -------
        float
            Reliability score between 0 and 1.
        """
        # Cluster count score (0-1, saturates at 50 clusters).
        cluster_score = min(self.n_clusters / 50, 1.0)
        
        # Balance score (0-1).
        if self.n_clusters > 0:
            balance = min(self.n_treated_clusters, self.n_control_clusters) / (self.n_clusters / 2)
            balance_score = min(balance, 1.0)
        else:
            balance_score = 0.0
        
        # Size variation score (0-1, lower CV is better).
        cv_score = max(0, 1 - self.cluster_size_cv / 2)
        
        # Weighted average.
        return 0.5 * cluster_score + 0.3 * balance_score + 0.2 * cv_score


@dataclass
class ClusteringDiagnostics:
    """
    Diagnostic results for clustering structure analysis.
    
    This class contains the complete results of clustering diagnostics,
    including statistics for each potential clustering variable and
    recommendations.
    
    Attributes
    ----------
    cluster_structure : Dict[str, ClusterVarStats]
        Statistics for each potential clustering variable.
    recommended_cluster_var : Optional[str]
        Recommended clustering variable name.
    recommendation_reason : str
        Explanation for the recommendation.
    treatment_variation_level : str
        Detected level at which treatment varies.
    warnings : List[str]
        Warning messages about clustering choices.
    """
    cluster_structure: Dict[str, ClusterVarStats]
    recommended_cluster_var: Optional[str]
    recommendation_reason: str
    treatment_variation_level: str
    warnings: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """
        Generate human-readable summary of diagnostics.
        
        Returns
        -------
        str
            Formatted summary string.
        """
        lines = [
            "=" * 70,
            "CLUSTERING DIAGNOSTICS",
            "=" * 70,
            "",
            f"Treatment Variation Level: {self.treatment_variation_level}",
            "",
            "Potential Clustering Variables:",
            "-" * 60,
            f"{'Variable':>15} {'N Clusters':>12} {'Treated':>10} {'Control':>10} {'Valid':>8}",
            "-" * 60,
        ]
        
        for var_name, stats in self.cluster_structure.items():
            valid = "✓" if stats.is_valid_cluster else "✗"
            rec = " (rec)" if var_name == self.recommended_cluster_var else ""
            lines.append(
                f"{var_name:>15} {stats.n_clusters:>12} "
                f"{stats.n_treated_clusters:>10} {stats.n_control_clusters:>10} "
                f"{valid:>8}{rec}"
            )
        
        lines.extend([
            "",
            "─" * 70,
            f"RECOMMENDATION: cluster_var='{self.recommended_cluster_var}'",
            f"Reason: {self.recommendation_reason}",
            "─" * 70,
        ])
        
        if self.warnings:
            lines.extend(["", "WARNINGS:"])
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        
        lines.append("=" * 70)
        return "\n".join(lines)


@dataclass
class ClusteringRecommendation:
    """
    Recommendation for clustering level selection.
    
    This class provides a detailed recommendation for which clustering
    variable to use, along with confidence scores and alternatives.
    
    Attributes
    ----------
    recommended_var : str
        Recommended clustering variable name.
    n_clusters : int
        Number of clusters with recommended variable.
    n_treated_clusters : int
        Number of treated clusters.
    n_control_clusters : int
        Number of control clusters.
    confidence : float
        Confidence in recommendation (0-1).
    reasons : List[str]
        List of reasons supporting the recommendation.
    alternatives : List[Dict[str, Any]]
        Alternative clustering options with their statistics.
    warnings : List[str]
        Warning messages.
    use_wild_bootstrap : bool
        Whether to recommend wild cluster bootstrap.
    wild_bootstrap_reason : Optional[str]
        Reason for wild bootstrap recommendation.
    """
    recommended_var: str
    n_clusters: int
    n_treated_clusters: int
    n_control_clusters: int
    confidence: float
    reasons: List[str]
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    use_wild_bootstrap: bool = False
    wild_bootstrap_reason: Optional[str] = None
    
    def summary(self) -> str:
        """
        Generate human-readable summary of recommendation.
        
        Returns
        -------
        str
            Formatted summary string.
        """
        lines = [
            "=" * 70,
            "CLUSTERING LEVEL RECOMMENDATION",
            "=" * 70,
            "",
            f"Recommended: cluster_var='{self.recommended_var}'",
            f"  - Total clusters: {self.n_clusters}",
            f"  - Treated clusters: {self.n_treated_clusters}",
            f"  - Control clusters: {self.n_control_clusters}",
            f"  - Confidence: {self.confidence:.1%}",
            "",
            "Reasons:",
        ]
        for i, reason in enumerate(self.reasons, 1):
            lines.append(f"  {i}. {reason}")
        
        if self.use_wild_bootstrap:
            lines.extend([
                "",
                "⚠ WILD CLUSTER BOOTSTRAP RECOMMENDED",
                f"   Reason: {self.wild_bootstrap_reason}",
            ])
        
        if self.alternatives:
            lines.extend(["", "Alternatives:"])
            for alt in self.alternatives:
                lines.append(
                    f"  - {alt['var']}: {alt['n_clusters']} clusters "
                    f"({alt['reason']})"
                )
        
        if self.warnings:
            lines.extend(["", "WARNINGS:"])
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        
        lines.append("=" * 70)
        return "\n".join(lines)


@dataclass
class ClusteringConsistencyResult:
    """
    Result of clustering consistency check.
    
    This class contains the results of checking whether the chosen
    clustering level is consistent with the treatment variation level.
    
    Attributes
    ----------
    is_consistent : bool
        Whether clustering level is consistent with treatment variation.
    treatment_variation_level : str
        Detected level at which treatment varies.
    cluster_level : str
        Level of the clustering variable.
    n_clusters : int
        Number of clusters.
    n_treatment_changes_within_cluster : int
        Number of clusters with treatment variation within.
    pct_clusters_with_variation : float
        Percentage of clusters with within-cluster treatment variation.
    recommendation : str
        Suggested action if inconsistent.
    details : str
        Detailed explanation of the consistency check.
    """
    is_consistent: bool
    treatment_variation_level: str
    cluster_level: str
    n_clusters: int
    n_treatment_changes_within_cluster: int
    pct_clusters_with_variation: float
    recommendation: str
    details: str
    
    def summary(self) -> str:
        """
        Generate human-readable summary of consistency check.
        
        Returns
        -------
        str
            Formatted summary string.
        """
        status = "✓ Consistent" if self.is_consistent else "⚠ Inconsistent"
        lines = [
            "=" * 50,
            "CLUSTERING CONSISTENCY CHECK",
            "=" * 50,
            "",
            f"Status: {status}",
            "",
            self.details,
            "",
            f"Recommendation: {self.recommendation}",
            "=" * 50,
        ]
        return "\n".join(lines)


@dataclass
class WildClusterBootstrapResult:
    """
    Result of wild cluster bootstrap inference.
    
    This class contains the results of wild cluster bootstrap,
    which provides more reliable inference when the number of
    clusters is small.
    
    Attributes
    ----------
    att : float
        Point estimate of ATT.
    se_bootstrap : float
        Bootstrap standard error.
    ci_lower : float
        Lower bound of bootstrap confidence interval.
    ci_upper : float
        Upper bound of bootstrap confidence interval.
    pvalue : float
        Bootstrap p-value (two-sided).
    n_clusters : int
        Number of clusters.
    n_bootstrap : int
        Number of bootstrap replications.
    weight_type : str
        Type of bootstrap weights used.
    t_stat_original : float
        Original t-statistic.
    t_stats_bootstrap : np.ndarray
        Bootstrap t-statistics.
    rejection_rate : float
        Proportion of bootstrap t-stats exceeding original.
    """
    att: float
    se_bootstrap: float
    ci_lower: float
    ci_upper: float
    pvalue: float
    n_clusters: int
    n_bootstrap: int
    weight_type: str
    t_stat_original: float
    t_stats_bootstrap: Any  # np.ndarray
    rejection_rate: float
    
    def summary(self) -> str:
        """
        Generate human-readable summary of bootstrap results.
        
        Returns
        -------
        str
            Formatted summary string.
        """
        sig = "***" if self.pvalue < 0.01 else "**" if self.pvalue < 0.05 else "*" if self.pvalue < 0.1 else ""
        return (
            f"Wild Cluster Bootstrap Results\n"
            f"{'='*40}\n"
            f"ATT: {self.att:.4f} {sig}\n"
            f"Bootstrap SE: {self.se_bootstrap:.4f}\n"
            f"95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]\n"
            f"P-value: {self.pvalue:.4f}\n"
            f"N clusters: {self.n_clusters}\n"
            f"N bootstrap: {self.n_bootstrap}\n"
            f"Weight type: {self.weight_type}\n"
            f"{'='*40}"
        )


# =============================================================================
# Helper Functions
# =============================================================================

def _validate_clustering_inputs(
    data: pd.DataFrame,
    ivar: str,
    potential_cluster_vars: List[str],
    gvar: Optional[str],
    d: Optional[str],
) -> None:
    """
    Validate inputs for clustering diagnostics.
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    ivar : str
        Unit identifier column name.
    potential_cluster_vars : List[str]
        List of potential clustering variable names.
    gvar : str, optional
        Cohort variable for staggered designs.
    d : str, optional
        Treatment indicator for common timing.
    
    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame")
    
    if ivar not in data.columns:
        raise ValueError(f"Unit variable '{ivar}' not found in data")
    
    for var in potential_cluster_vars:
        if var not in data.columns:
            raise ValueError(f"Cluster variable '{var}' not found in data")
    
    if gvar is not None and gvar not in data.columns:
        raise ValueError(f"Cohort variable '{gvar}' not found in data")
    
    if d is not None and d not in data.columns:
        raise ValueError(f"Treatment variable '{d}' not found in data")
    
    if gvar is None and d is None:
        raise ValueError("Either gvar or d must be specified")
    
    if len(potential_cluster_vars) == 0:
        raise ValueError("At least one potential cluster variable must be specified")


def _analyze_cluster_var(
    data: pd.DataFrame,
    ivar: str,
    cluster_var: str,
    gvar: Optional[str],
    d: Optional[str],
) -> ClusterVarStats:
    """
    Analyze a single potential clustering variable.
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    ivar : str
        Unit identifier.
    cluster_var : str
        Clustering variable to analyze.
    gvar : str, optional
        Cohort variable (staggered).
    d : str, optional
        Treatment indicator (common timing).
    
    Returns
    -------
    ClusterVarStats
        Statistics for the clustering variable.
    """
    # Basic cluster statistics.
    cluster_sizes = data.groupby(cluster_var).size()
    n_clusters = len(cluster_sizes)
    
    # Determine treatment variable.
    if gvar is not None:
        # Staggered design: units with non-never-treated gvar values are treated.
        never_treated_vals = [0, np.inf]
        treated_mask = ~data[gvar].isin(never_treated_vals) & data[gvar].notna()
    elif d is not None:
        treated_mask = data[d] == 1
    else:
        treated_mask = pd.Series(False, index=data.index)
    
    # Count treated and control clusters.
    # Group treated_mask directly by cluster variable to avoid FutureWarning.
    cluster_has_treated = treated_mask.groupby(data[cluster_var]).any()
    n_treated_clusters = int(cluster_has_treated.sum())
    n_control_clusters = n_clusters - n_treated_clusters
    
    # Check if cluster is nested within unit (invalid for clustering).
    units_per_cluster = data.groupby(cluster_var)[ivar].nunique()
    n_unique_units = data[ivar].nunique()
    is_nested_in_unit = (units_per_cluster == 1).all() and n_clusters > n_unique_units
    
    # Determine level relative to unit.
    if is_nested_in_unit:
        level = ClusteringLevel.LOWER
    elif n_clusters == n_unique_units:
        level = ClusteringLevel.SAME
    else:
        level = ClusteringLevel.HIGHER
    
    # Check if treatment varies within clusters.
    if gvar is not None:
        treatment_per_cluster = data.groupby(cluster_var)[gvar].nunique()
    elif d is not None:
        treatment_per_cluster = data.groupby(cluster_var)[d].nunique()
    else:
        treatment_per_cluster = pd.Series(1, index=data[cluster_var].unique())
    
    treatment_varies = (treatment_per_cluster > 1).any()
    n_with_variation = int((treatment_per_cluster > 1).sum())
    
    # Compute coefficient of variation of cluster sizes.
    if cluster_sizes.mean() > 0:
        cluster_size_cv = float(cluster_sizes.std() / cluster_sizes.mean())
    else:
        cluster_size_cv = 0.0
    
    return ClusterVarStats(
        var_name=cluster_var,
        n_clusters=n_clusters,
        n_treated_clusters=n_treated_clusters,
        n_control_clusters=n_control_clusters,
        min_cluster_size=int(cluster_sizes.min()),
        max_cluster_size=int(cluster_sizes.max()),
        mean_cluster_size=float(cluster_sizes.mean()),
        median_cluster_size=float(cluster_sizes.median()),
        cluster_size_cv=cluster_size_cv,
        level_relative_to_unit=level,
        units_per_cluster=float(units_per_cluster.mean()),
        is_nested_in_unit=is_nested_in_unit,
        treatment_varies_within_cluster=treatment_varies,
        n_clusters_with_treatment_variation=n_with_variation
    )


def _detect_treatment_variation_level(
    data: pd.DataFrame,
    ivar: str,
    potential_cluster_vars: List[str],
    gvar: Optional[str],
    d: Optional[str],
) -> str:
    """
    Detect the level at which treatment varies.
    
    Returns the name of the variable at which treatment is constant
    within groups (i.e., the treatment variation level).
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    ivar : str
        Unit identifier.
    potential_cluster_vars : List[str]
        List of potential clustering variables.
    gvar : str, optional
        Cohort variable.
    d : str, optional
        Treatment indicator.
    
    Returns
    -------
    str
        Name of the variable at which treatment varies.
    """
    treatment_var = gvar if gvar is not None else d
    if treatment_var is None:
        return "unknown"
    
    # Check from highest level (fewest unique values) to lowest level.
    sorted_vars = sorted(potential_cluster_vars, key=lambda v: data[v].nunique())
    
    for var in sorted_vars:
        treatment_per_group = data.groupby(var)[treatment_var].nunique()
        if (treatment_per_group == 1).all():
            return var
    
    # If treatment varies at all levels, return the unit level.
    return ivar


def _generate_clustering_recommendation(
    cluster_structure: Dict[str, ClusterVarStats],
    treatment_level: str,
) -> Tuple[Optional[str], str, List[str]]:
    """
    Generate clustering recommendation based on diagnostics.
    
    Parameters
    ----------
    cluster_structure : Dict[str, ClusterVarStats]
        Statistics for each potential clustering variable.
    treatment_level : str
        Detected treatment variation level.
    
    Returns
    -------
    Tuple[Optional[str], str, List[str]]
        Tuple of (recommended_var, reason, warnings).
    """
    warnings = []
    
    # Filter to valid options.
    valid_options = {
        var: stats for var, stats in cluster_structure.items()
        if stats.is_valid_cluster
    }
    
    if not valid_options:
        return None, "No valid clustering options available.", [
            "All potential cluster variables are invalid (nested within units or < 2 clusters)."
        ]
    
    # Prefer clustering at the treatment variation level.
    if treatment_level in valid_options:
        stats = valid_options[treatment_level]
        if stats.n_clusters >= 20:
            return treatment_level, (
                f"Clustering at treatment variation level ({treatment_level}) "
                f"with {stats.n_clusters} clusters."
            ), warnings
        else:
            warnings.append(
                f"Treatment varies at {treatment_level} level but only "
                f"{stats.n_clusters} clusters available."
            )
    
    # Otherwise, select the option with sufficient clusters and highest reliability.
    ranked = sorted(
        valid_options.items(),
        key=lambda x: (x[1].n_clusters >= 20, x[1].reliability_score),
        reverse=True
    )
    
    best_var, best_stats = ranked[0]
    
    if best_stats.n_clusters < 20:
        warnings.append(
            f"Recommended clustering has only {best_stats.n_clusters} clusters. "
            f"Consider wild cluster bootstrap for reliable inference."
        )
    
    if best_stats.treatment_varies_within_cluster:
        warnings.append(
            f"Treatment varies within {best_stats.n_clusters_with_treatment_variation} "
            f"clusters. Standard errors may be conservative."
        )
    
    reason = (
        f"Best available option with {best_stats.n_clusters} clusters "
        f"(reliability score: {best_stats.reliability_score:.2f})."
    )
    
    return best_var, reason, warnings


def _generate_recommendation_reasons(
    var: str,
    stats: ClusterVarStats,
    diag: ClusteringDiagnostics,
) -> List[str]:
    """
    Generate list of reasons for clustering recommendation.
    
    Parameters
    ----------
    var : str
        Recommended variable name.
    stats : ClusterVarStats
        Statistics for the recommended variable.
    diag : ClusteringDiagnostics
        Full diagnostics results.
    
    Returns
    -------
    List[str]
        List of reasons.
    """
    reasons = []
    
    # Treatment variation level.
    if var == diag.treatment_variation_level:
        reasons.append(f"Treatment varies at {var} level - clustering at this level is appropriate")
    
    # Cluster count.
    if stats.n_clusters >= 30:
        reasons.append(f"Sufficient clusters ({stats.n_clusters}) for reliable inference")
    elif stats.n_clusters >= 20:
        reasons.append(f"Adequate clusters ({stats.n_clusters}) for inference")
    else:
        reasons.append(f"Limited clusters ({stats.n_clusters}) - consider wild bootstrap")
    
    # Balance.
    if stats.n_treated_clusters > 0 and stats.n_control_clusters > 0:
        balance_ratio = min(stats.n_treated_clusters, stats.n_control_clusters) / max(stats.n_treated_clusters, stats.n_control_clusters)
        if balance_ratio > 0.5:
            reasons.append(f"Good balance between treated ({stats.n_treated_clusters}) and control ({stats.n_control_clusters}) clusters")
    
    # Hierarchy level.
    if stats.level_relative_to_unit == ClusteringLevel.HIGHER:
        reasons.append(f"Clustering at higher level than unit of observation")
    
    return reasons


def _get_alternative_reason(stats: ClusterVarStats) -> str:
    """
    Get reason string for an alternative clustering option.
    
    Parameters
    ----------
    stats : ClusterVarStats
        Statistics for the alternative.
    
    Returns
    -------
    str
        Reason string.
    """
    if stats.n_clusters < 10:
        return "too few clusters"
    elif stats.n_clusters < 20:
        return "marginal cluster count"
    elif stats.treatment_varies_within_cluster:
        return "treatment varies within clusters"
    else:
        return f"reliability score: {stats.reliability_score:.2f}"


def _generate_clustering_warnings(
    stats: ClusterVarStats,
    diag: ClusteringDiagnostics,
) -> List[str]:
    """
    Generate warning messages for clustering choice.
    
    Parameters
    ----------
    stats : ClusterVarStats
        Statistics for the recommended variable.
    diag : ClusteringDiagnostics
        Full diagnostics results.
    
    Returns
    -------
    List[str]
        List of warning messages.
    """
    warnings = []
    
    if stats.n_clusters < 10:
        warnings.append(
            f"Only {stats.n_clusters} clusters - cluster-robust inference may be unreliable"
        )
    elif stats.n_clusters < 20:
        warnings.append(
            f"Only {stats.n_clusters} clusters - consider wild cluster bootstrap"
        )
    
    if stats.cluster_size_cv > 1.0:
        warnings.append(
            f"Highly variable cluster sizes (CV={stats.cluster_size_cv:.2f})"
        )
    
    if stats.treatment_varies_within_cluster:
        warnings.append(
            f"Treatment varies within {stats.n_clusters_with_treatment_variation} clusters"
        )
    
    return warnings


def _determine_cluster_level(
    data: pd.DataFrame,
    ivar: str,
    cluster_var: str,
) -> str:
    """
    Determine the level of clustering variable relative to unit.
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    ivar : str
        Unit identifier.
    cluster_var : str
        Clustering variable.
    
    Returns
    -------
    str
        Level description: 'higher', 'same', or 'lower'.
    """
    n_units = data[ivar].nunique()
    n_clusters = data[cluster_var].nunique()
    
    # Check how many units belong to each cluster.
    units_per_cluster = data.groupby(cluster_var)[ivar].nunique()
    
    if n_clusters < n_units and (units_per_cluster > 1).any():
        return "higher"
    elif n_clusters == n_units:
        return "same"
    else:
        return "lower"


# =============================================================================
# Main Public Functions
# =============================================================================

def diagnose_clustering(
    data: pd.DataFrame,
    ivar: str,
    potential_cluster_vars: List[str],
    gvar: Optional[str] = None,
    d: Optional[str] = None,
    verbose: bool = True,
) -> ClusteringDiagnostics:
    """
    Diagnose clustering structure and recommend clustering level.
    
    Analyzes the hierarchical structure of potential clustering variables
    relative to the unit of observation and treatment assignment.
    
    This function helps users choose the appropriate clustering level for
    standard error estimation in difference-in-differences analysis.
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    ivar : str
        Unit identifier column name.
    potential_cluster_vars : List[str]
        List of potential clustering variable column names to evaluate.
    gvar : str, optional
        Cohort variable for staggered designs. Use this for staggered
        adoption designs where treatment timing varies across units.
    d : str, optional
        Treatment indicator variable (for common timing). Use this for
        designs where all treated units receive treatment at the same time.
    verbose : bool, default True
        Whether to print diagnostic summary.
    
    Returns
    -------
    ClusteringDiagnostics
        Diagnostic results containing:
        - cluster_structure: Statistics for each potential clustering variable
        - recommended_cluster_var: Recommended clustering variable name
        - recommendation_reason: Explanation for the recommendation
        - treatment_variation_level: Detected level at which treatment varies
        - warnings: Warning messages about clustering choices
    
    Raises
    ------
    ValueError
        If inputs are invalid (missing columns, no treatment variable, etc.)
    
    Notes
    -----
    When the policy or treatment varies at a level higher than the unit of
    observation, standard errors should be clustered at the policy variation
    level to properly account for within-cluster correlation.
    
    The function evaluates each potential clustering variable based on:
    
    - Number of clusters (more is better, 20-30 minimum recommended)
    - Balance between treated and control clusters
    - Whether treatment varies within clusters
    - Cluster size variation (coefficient of variation)
    
    See Also
    --------
    recommend_clustering_level : Get detailed recommendation with alternatives.
    check_clustering_consistency : Validate clustering choice against treatment.
    """
    # Validate inputs.
    _validate_clustering_inputs(data, ivar, potential_cluster_vars, gvar, d)
    
    # Analyze each potential clustering variable.
    cluster_structure = {}
    for var in potential_cluster_vars:
        stats = _analyze_cluster_var(data, ivar, var, gvar, d)
        cluster_structure[var] = stats
    
    # Detect treatment variation level.
    treatment_level = _detect_treatment_variation_level(
        data, ivar, potential_cluster_vars, gvar, d
    )
    
    # Generate recommendation.
    recommended_var, reason, warnings = _generate_clustering_recommendation(
        cluster_structure, treatment_level
    )
    
    result = ClusteringDiagnostics(
        cluster_structure=cluster_structure,
        recommended_cluster_var=recommended_var,
        recommendation_reason=reason,
        treatment_variation_level=treatment_level,
        warnings=warnings
    )
    
    if verbose:
        print(result.summary())
    
    return result


def recommend_clustering_level(
    data: pd.DataFrame,
    ivar: str,
    tvar: str,
    potential_cluster_vars: List[str],
    gvar: Optional[str] = None,
    d: Optional[str] = None,
    min_clusters: int = 20,
    verbose: bool = True,
) -> ClusteringRecommendation:
    """
    Recommend optimal clustering level based on data characteristics.
    
    This function provides a detailed recommendation for which clustering
    variable to use, along with confidence scores, alternatives, and
    guidance on whether wild cluster bootstrap is needed.
    
    Algorithm:
    1. Analyze each potential cluster variable
    2. Detect treatment variation level
    3. Filter to valid clustering options
    4. Rank by reliability score
    5. Check if wild bootstrap is needed (when clusters < min_clusters)
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    potential_cluster_vars : List[str]
        List of potential clustering variable column names.
    gvar : str, optional
        Cohort/treatment timing variable column name (for staggered designs).
    d : str, optional
        Treatment indicator variable (for common timing designs).
    min_clusters : int, default 20
        Minimum recommended number of clusters. If the recommended
        clustering variable has fewer clusters, wild cluster bootstrap
        will be recommended.
    verbose : bool, default True
        Whether to print recommendation summary.
    
    Returns
    -------
    ClusteringRecommendation
        Recommendation containing:
        - recommended_var: Recommended clustering variable name
        - n_clusters: Number of clusters with recommended variable
        - n_treated_clusters: Number of treated clusters
        - n_control_clusters: Number of control clusters
        - confidence: Confidence in recommendation (0-1)
        - reasons: List of reasons supporting the recommendation
        - alternatives: Alternative clustering options
        - warnings: Warning messages
        - use_wild_bootstrap: Whether to recommend wild cluster bootstrap
        - wild_bootstrap_reason: Reason for wild bootstrap recommendation
    
    Raises
    ------
    ValueError
        If no valid clustering options are found.
    
    Notes
    -----
    The reliability score is computed as a weighted combination of:
    
    - Number of clusters (50% weight, saturates at 50 clusters)
    - Balance of treated/control clusters (30% weight)
    - Cluster size variation (20% weight, lower CV is better)
    
    When the number of clusters is below ``min_clusters``, the function
    recommends using wild cluster bootstrap for more reliable inference.
    
    See Also
    --------
    diagnose_clustering : Get detailed diagnostics for clustering structure.
    check_clustering_consistency : Check if clustering is consistent with treatment.
    wild_cluster_bootstrap : Bootstrap inference for small cluster counts.
    """
    # Run diagnostics.
    diag = diagnose_clustering(
        data, ivar, potential_cluster_vars, gvar=gvar, d=d, verbose=False
    )
    
    # Filter to valid options.
    valid_options = [
        (var, stats) for var, stats in diag.cluster_structure.items()
        if stats.is_valid_cluster
    ]
    
    if not valid_options:
        raise ValueError(
            "No valid clustering options found. "
            "All potential cluster variables are either nested within units "
            "or have fewer than 2 clusters."
        )
    
    # Rank by reliability score.
    ranked = sorted(valid_options, key=lambda x: x[1].reliability_score, reverse=True)
    
    # Select the best option.
    best_var, best_stats = ranked[0]
    
    # Generate recommendation reasons.
    reasons = _generate_recommendation_reasons(best_var, best_stats, diag)
    
    # Check if wild bootstrap is needed.
    use_wild_bootstrap = best_stats.n_clusters < min_clusters
    wild_reason = None
    if use_wild_bootstrap:
        wild_reason = (
            f"Only {best_stats.n_clusters} clusters available (< {min_clusters}). "
            f"Wild cluster bootstrap recommended for reliable inference."
        )
    
    # Generate alternatives.
    alternatives = []
    for var, stats in ranked[1:3]:  # Top 2 alternatives.
        alternatives.append({
            'var': var,
            'n_clusters': stats.n_clusters,
            'reliability_score': stats.reliability_score,
            'reason': _get_alternative_reason(stats)
        })
    
    # Generate warnings.
    warnings = _generate_clustering_warnings(best_stats, diag)
    
    result = ClusteringRecommendation(
        recommended_var=best_var,
        n_clusters=best_stats.n_clusters,
        n_treated_clusters=best_stats.n_treated_clusters,
        n_control_clusters=best_stats.n_control_clusters,
        confidence=best_stats.reliability_score,
        reasons=reasons,
        alternatives=alternatives,
        warnings=warnings,
        use_wild_bootstrap=use_wild_bootstrap,
        wild_bootstrap_reason=wild_reason
    )
    
    if verbose:
        print(result.summary())
    
    return result


def check_clustering_consistency(
    data: pd.DataFrame,
    ivar: str,
    cluster_var: str,
    gvar: Optional[str] = None,
    d: Optional[str] = None,
    verbose: bool = True,
) -> ClusteringConsistencyResult:
    """
    Check if clustering level is consistent with treatment variation level.
    
    A consistent clustering choice means:
    1. Treatment does not vary within clusters (or varies minimally)
    2. Cluster level is at or above the treatment variation level
    
    This function helps validate that the chosen clustering variable is
    appropriate for the treatment assignment mechanism.
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    ivar : str
        Unit identifier.
    cluster_var : str
        Clustering variable to check.
    gvar : str, optional
        Treatment timing variable (for staggered designs).
    d : str, optional
        Treatment indicator (for common timing designs).
    verbose : bool, default True
        Whether to print results.
    
    Returns
    -------
    ClusteringConsistencyResult
        Consistency check results containing:
        - is_consistent: Whether clustering level is consistent
        - treatment_variation_level: Detected treatment variation level
        - cluster_level: Level of the clustering variable
        - n_clusters: Number of clusters
        - n_treatment_changes_within_cluster: Clusters with treatment variation
        - pct_clusters_with_variation: Percentage with variation
        - recommendation: Suggested action if inconsistent
        - details: Detailed explanation
    
    Raises
    ------
    ValueError
        If inputs are invalid.
    
    Notes
    -----
    A clustering choice is considered consistent if:
    
    - Less than 5% of clusters have within-cluster treatment variation
    - The cluster level is at the same level or higher than the unit
    
    If treatment varies within clusters, standard errors may be conservative
    (too large), leading to under-rejection of the null hypothesis.
    
    See Also
    --------
    diagnose_clustering : Get detailed diagnostics for clustering structure.
    recommend_clustering_level : Get recommendation for clustering level.
    """
    # Validate inputs.
    if cluster_var not in data.columns:
        raise ValueError(f"Cluster variable '{cluster_var}' not found in data")
    if ivar not in data.columns:
        raise ValueError(f"Unit variable '{ivar}' not found in data")
    if gvar is None and d is None:
        raise ValueError("Either gvar or d must be specified")
    
    treatment_var = gvar if gvar is not None else d
    if treatment_var not in data.columns:
        raise ValueError(f"Treatment variable '{treatment_var}' not found in data")
    
    # Check if treatment varies within clusters.
    cluster_treatment = data.groupby(cluster_var)[treatment_var].nunique()
    n_clusters_with_variation = int((cluster_treatment > 1).sum())
    n_clusters = len(cluster_treatment)
    pct_with_variation = n_clusters_with_variation / n_clusters * 100 if n_clusters > 0 else 0
    
    # Determine treatment variation level.
    treatment_level = _detect_treatment_variation_level(
        data, ivar, [cluster_var], gvar, d
    )
    
    # Determine cluster level.
    cluster_level = _determine_cluster_level(data, ivar, cluster_var)
    
    # Check consistency.
    is_consistent = (
        pct_with_variation < 5 and  # Less than 5% of clusters have variation.
        cluster_level in ['same', 'higher']
    )
    
    # Generate recommendation.
    if is_consistent:
        recommendation = "Clustering choice is appropriate."
    else:
        if pct_with_variation >= 5:
            recommendation = (
                f"Treatment varies within {pct_with_variation:.1f}% of clusters. "
                f"Consider clustering at a higher level where treatment is constant."
            )
        else:
            recommendation = (
                f"Cluster level ({cluster_level}) may be inappropriate. "
                f"Consider clustering at the treatment variation level ({treatment_level})."
            )
    
    # Generate details.
    details = (
        f"Analyzed {n_clusters} clusters.\n"
        f"Treatment varies within {n_clusters_with_variation} clusters "
        f"({pct_with_variation:.1f}%).\n"
        f"Treatment variation level: {treatment_level}\n"
        f"Cluster level: {cluster_level}"
    )
    
    result = ClusteringConsistencyResult(
        is_consistent=is_consistent,
        treatment_variation_level=treatment_level,
        cluster_level=cluster_level,
        n_clusters=n_clusters,
        n_treatment_changes_within_cluster=n_clusters_with_variation,
        pct_clusters_with_variation=pct_with_variation,
        recommendation=recommendation,
        details=details
    )
    
    if verbose:
        print(result.summary())
    
    return result

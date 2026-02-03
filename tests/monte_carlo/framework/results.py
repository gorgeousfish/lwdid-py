# -*- coding: utf-8 -*-
"""
Monte Carlo Results Data Structures

Provides standardized data structures for storing and reporting
Monte Carlo simulation results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class MonteCarloResults:
    """
    Results from Monte Carlo simulation.
    
    Contains complete evaluation metrics and raw data for detailed analysis.
    
    Attributes
    ----------
    estimator_name : str
        Name of the estimator being evaluated.
    dgp_type : str
        Type of DGP used ('large_sample' or 'small_sample').
    scenario : str
        Scenario identifier (e.g., '1C', '2C', 'scenario_1').
    n_reps : int
        Number of Monte Carlo replications requested.
    n_units : int
        Sample size per replication.
    target_period : int, optional
        Target evaluation period (if applicable).
    
    Core Metrics
    ------------
    bias : float
        Mean bias = E[ATT_hat] - ATT_true.
    sd : float
        Standard deviation of estimates.
    rmse : float
        Root mean squared error = sqrt(bias² + sd²).
    coverage : float
        95% CI coverage rate.
    mean_se : float
        Average estimated standard error.
    se_ratio : float
        Ratio of mean_se to sd (should be ~1 for valid inference).
    """
    # Metadata
    estimator_name: str
    dgp_type: str
    scenario: str
    n_reps: int
    n_units: int
    target_period: Optional[int] = None

    # Core evaluation metrics
    bias: float = 0.0
    sd: float = 0.0
    rmse: float = 0.0
    coverage: float = 0.0
    mean_se: float = 0.0
    se_ratio: float = 0.0
    
    # Additional metrics
    rejection_rate: float = 0.0
    mean_att: float = 0.0
    true_att: float = 0.0
    median_att: float = 0.0
    
    # Counts
    n_valid: int = 0
    n_failed: int = 0
    
    # Raw data (for detailed analysis)
    estimates: np.ndarray = field(default_factory=lambda: np.array([]))
    standard_errors: np.ndarray = field(default_factory=lambda: np.array([]))
    ci_lowers: np.ndarray = field(default_factory=lambda: np.array([]))
    ci_uppers: np.ndarray = field(default_factory=lambda: np.array([]))
    true_values: np.ndarray = field(default_factory=lambda: np.array([]))
    p_values: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def summary(self) -> str:
        """Generate formatted summary string."""
        return f"""
{'=' * 60}
Monte Carlo Results: {self.estimator_name}
{'=' * 60}
DGP: {self.dgp_type}, Scenario: {self.scenario}
Replications: {self.n_reps} (valid: {self.n_valid}, failed: {self.n_failed})
Sample size: {self.n_units}

Performance Metrics:
  True ATT:    {self.true_att:.4f}
  Mean ATT:    {self.mean_att:.4f}
  Bias:        {self.bias:.4f}
  SD:          {self.sd:.4f}
  RMSE:        {self.rmse:.4f}
  
Inference Quality:
  Coverage:    {self.coverage:.2%}
  Mean SE:     {self.mean_se:.4f}
  SE Ratio:    {self.se_ratio:.4f} (target: 1.0)
  Rejection:   {self.rejection_rate:.2%}
{'=' * 60}
"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for tabulation."""
        return {
            'estimator': self.estimator_name,
            'dgp_type': self.dgp_type,
            'scenario': self.scenario,
            'n_units': self.n_units,
            'n_reps': self.n_reps,
            'n_valid': self.n_valid,
            'true_att': round(self.true_att, 4),
            'mean_att': round(self.mean_att, 4),
            'bias': round(self.bias, 4),
            'sd': round(self.sd, 4),
            'rmse': round(self.rmse, 4),
            'coverage': round(self.coverage, 4),
            'mean_se': round(self.mean_se, 4),
            'se_ratio': round(self.se_ratio, 4),
            'rejection_rate': round(self.rejection_rate, 4),
        }
    
    def to_paper_format(self) -> Dict[str, Any]:
        """Convert to paper table format (Table 2 style)."""
        return {
            'Estimator': self.estimator_name,
            'Average Effects': round(self.mean_att, 3),
            'Bias': round(self.bias, 3),
            'SD': round(self.sd, 2),
            'RMSE': round(self.rmse, 3),
            'Coverage Rate': round(self.coverage, 2),
            'Avg SE': round(self.mean_se, 2),
        }
    
    def validate_against_paper(
        self,
        paper_values: Dict[str, float],
        tolerances: Dict[str, float] | None = None,
    ) -> Dict[str, bool]:
        """
        Validate results against paper reference values.
        
        Parameters
        ----------
        paper_values : dict
            Expected values from paper (bias, sd, rmse, coverage).
        tolerances : dict, optional
            Tolerance for each metric. Defaults to reasonable values.
        
        Returns
        -------
        dict
            Validation results for each metric.
        """
        if tolerances is None:
            tolerances = {
                'bias': 0.2,
                'sd': 0.3,
                'rmse': 0.3,
                'coverage': 0.05,
            }
        
        results = {}
        for metric, expected in paper_values.items():
            if expected is None:
                continue
            actual = getattr(self, metric, None)
            if actual is not None:
                tol = tolerances.get(metric, 0.1)
                results[metric] = abs(actual - expected) < tol
        
        return results


@dataclass
class MonteCarloComparison:
    """
    Multi-estimator Monte Carlo comparison results.
    
    Aggregates results from multiple estimators for the same DGP/scenario
    to facilitate comparison.
    """
    dgp_type: str
    scenario: str
    n_units: int
    n_reps: int
    results: Dict[str, MonteCarloResults] = field(default_factory=dict)
    
    def add_result(self, result: MonteCarloResults) -> None:
        """Add a single estimator result."""
        self.results[result.estimator_name] = result
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for comparison."""
        rows = [r.to_dict() for r in self.results.values()]
        return pd.DataFrame(rows)
    
    def to_paper_table(self) -> pd.DataFrame:
        """Generate paper-format comparison table."""
        rows = [r.to_paper_format() for r in self.results.values()]
        return pd.DataFrame(rows)
    
    def summary(self) -> str:
        """Generate comparison summary."""
        lines = [
            f"Monte Carlo Comparison: {self.dgp_type}, {self.scenario}",
            f"N={self.n_units}, Reps={self.n_reps}",
            "=" * 80,
            f"{'Estimator':<20} {'Bias':>8} {'SD':>8} {'RMSE':>8} {'Coverage':>10} {'SE Ratio':>10}",
            "-" * 80,
        ]
        
        for name, result in self.results.items():
            lines.append(
                f"{name:<20} {result.bias:>8.4f} {result.sd:>8.4f} "
                f"{result.rmse:>8.4f} {result.coverage:>10.2%} {result.se_ratio:>10.4f}"
            )
        
        lines.append("=" * 80)
        return "\n".join(lines)
    
    def get_best_estimator(self, metric: str = 'rmse') -> str:
        """
        Get the best estimator by a given metric.
        
        Parameters
        ----------
        metric : str
            Metric to compare ('rmse', 'bias', 'coverage').
        
        Returns
        -------
        str
            Name of the best estimator.
        """
        if metric == 'coverage':
            # Best coverage is closest to 0.95
            return min(
                self.results.keys(),
                key=lambda k: abs(self.results[k].coverage - 0.95)
            )
        elif metric == 'bias':
            # Best bias is closest to 0
            return min(
                self.results.keys(),
                key=lambda k: abs(self.results[k].bias)
            )
        else:
            # For rmse, sd: lower is better
            return min(
                self.results.keys(),
                key=lambda k: getattr(self.results[k], metric)
            )


__all__ = ['MonteCarloResults', 'MonteCarloComparison']

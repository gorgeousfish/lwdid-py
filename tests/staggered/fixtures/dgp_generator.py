"""
Data Generating Process (DGP) for staggered Difference-in-Differences simulations.

Implements the Monte Carlo DGP specification from Section 7.2 of
Lee & Wooldridge (2023), designed for validating staggered DiD estimators.

DGP parameters (Paper Section 7.2):
- N = 1000 units, T = 6 periods
- Cohort shares: g4=12%, g5=11%, g6=11%, NT=66%
- True ATT: tau_{g,r} = 1.5 + 0.5*(r-g) + 0.3*(g-4)

Usage:
>>> from fixtures.dgp_generator import StaggeredDGP
>>> dgp = StaggeredDGP(n_units=1000, seed=42)
>>> data = dgp.generate()
>>> true_att = dgp.get_true_att(g=4, r=5)

References
----------
Lee, S. J. & Wooldridge, J. M. (2023). "Simple Difference-in-Differences
    Estimation in Fixed Effects Models." SSRN 5325686.
Lee, S. J. & Wooldridge, J. M. (2025). "A Simple Transformation Approach
    to DiD Estimation for Panel Data." SSRN 4516518.
"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class StaggeredDGP:
    """
    Staggered Difference-in-Differences data generating process.

    Implements the DGP specification from Lee & Wooldridge (2023, Section 7.2)
    for Monte Carlo validation of staggered DiD estimators.

    Attributes
    ----------
    n_units : int
        Number of cross-sectional units.
    n_periods : int
        Number of time periods.
    cohort_shares : Dict[int, float]
        Share of units in each treatment cohort {cohort: share}.
    base_att : float
        Baseline average treatment effect on the treated.
    exposure_coef : float
        Coefficient on treatment exposure duration.
    cohort_coef : float
        Coefficient on cohort timing.
    seed : int
        Random number generator seed.
    """

    def __init__(
        self,
        n_units: int = 1000,
        n_periods: int = 6,
        cohort_shares: Optional[Dict[int, float]] = None,
        base_att: float = 1.5,
        exposure_coef: float = 0.5,
        cohort_coef: float = 0.3,
        seed: Optional[int] = None,
    ):
        """Initialize the staggered DGP with simulation parameters.

        Parameters
        ----------
        n_units : int
            Number of cross-sectional units.
        n_periods : int
            Number of time periods (T).
        cohort_shares : Dict[int, float], optional
            Share of units in each cohort. Default: {0: 0.66, 4: 0.12, 5: 0.11, 6: 0.11}.
            Cohort 0 denotes never-treated units.
        base_att : float
            Baseline treatment effect tau_0.
        exposure_coef : float
            Exposure duration coefficient beta (coefficient on r-g).
        cohort_coef : float
            Cohort timing coefficient gamma (coefficient on g-4).
        seed : int, optional
            Random number generator seed for reproducibility.
        """
        self.n_units = n_units
        self.n_periods = n_periods

        if cohort_shares is None:
            # Default shares per paper specification (Section 7.2)
            self.cohort_shares = {0: 0.66, 4: 0.12, 5: 0.11, 6: 0.11}
        else:
            self.cohort_shares = cohort_shares

        self.base_att = base_att
        self.exposure_coef = exposure_coef
        self.cohort_coef = cohort_coef

        if seed is not None:
            np.random.seed(seed)
        self.seed = seed

        # Validate that cohort shares sum to unity
        total_share = sum(self.cohort_shares.values())
        if abs(total_share - 1.0) > 1e-10:
            raise ValueError(f"cohort_shares must sum to 1, got: {total_share}")

    def get_true_att(self, g: int, r: int) -> float:
        """Compute the true ATT tau_{g,r} for a given cohort-period pair.

        Formula: tau_{g,r} = tau_0 + beta*(r-g) + gamma*(g-4)

        Parameters
        ----------
        g : int
            Treatment cohort (first treatment period).
        r : int
            Evaluation period.

        Returns
        -------
        float
            True average treatment effect on the treated.
        """
        if g == 0:  # Never-treated units
            return 0.0
        if r < g:  # Pre-treatment periods
            return 0.0

        # tau_{g,r} = 1.5 + 0.5*(r-g) + 0.3*(g-4)
        return self.base_att + self.exposure_coef * (r - g) + self.cohort_coef * (g - 4)

    def generate(self) -> pd.DataFrame:
        """Generate simulated staggered DiD panel data.

        Returns
        -------
        pd.DataFrame
            Panel dataset with columns:
            - id: unit identifier
            - year: time period (1, 2, ..., T)
            - y: outcome variable
            - gvar: treatment cohort (0=never-treated, 4, 5, 6)
            - x1, x2: covariates
            - treated: treatment status indicator (0/1)
        """
        # Step 1: Assign treatment cohorts
        cohorts = list(self.cohort_shares.keys())
        probs = list(self.cohort_shares.values())
        unit_cohorts = np.random.choice(cohorts, size=self.n_units, p=probs)

        # Step 2: Generate unit-level covariates
        x1 = np.random.randn(self.n_units)
        x2 = np.random.randn(self.n_units)

        # Step 3: Construct panel observations
        data_list = []

        for i in range(self.n_units):
            g = unit_cohorts[i]

            for t in range(1, self.n_periods + 1):
                # Baseline outcome model: Y = alpha + beta_1*t + beta_2*x1 + beta_3*x2 + epsilon
                y_base = 1.0 + 0.5 * t + 0.3 * x1[i] + 0.2 * x2[i]

                # Individual-specific effect (simulated fixed effect)
                individual_effect = 0.1 * i / self.n_units

                # Idiosyncratic error term
                epsilon = np.random.randn()

                # Treatment effect
                if g > 0 and t >= g:
                    tau = self.get_true_att(g, t)
                    treated = 1
                else:
                    tau = 0.0
                    treated = 0

                y = y_base + individual_effect + tau + epsilon

                data_list.append({
                    'id': i + 1,
                    'year': t,
                    'y': y,
                    'gvar': g,
                    'x1': x1[i],
                    'x2': x2[i],
                    'treated': treated,
                    # Cohort and period indicator variables for Stata cross-validation
                    'g0': 1 if g == 0 else 0,
                    'g4': 1 if g == 4 else 0,
                    'g5': 1 if g == 5 else 0,
                    'g6': 1 if g == 6 else 0,
                    'f04': 1 if t == 4 else 0,
                    'f05': 1 if t == 5 else 0,
                    'f06': 1 if t == 6 else 0,
                })

        return pd.DataFrame(data_list)

    def get_all_true_atts(self) -> Dict[Tuple[int, int], float]:
        """Return true ATT values for all valid (g, r) cohort-period pairs.

        Returns
        -------
        Dict[Tuple[int, int], float]
            Mapping of {(g, r): true_att} for all post-treatment periods.
        """
        true_atts = {}
        for g in [c for c in self.cohort_shares.keys() if c > 0]:
            for r in range(g, self.n_periods + 1):
                true_atts[(g, r)] = self.get_true_att(g, r)
        return true_atts


def generate_staggered_data(
    n_units: int = 1000,
    n_periods: int = 6,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], float]]:
    """Convenience function to generate staggered DiD data with true ATT values.

    Parameters
    ----------
    n_units : int
        Number of cross-sectional units.
    n_periods : int
        Number of time periods.
    seed : int, optional
        Random number generator seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (panel_data, {(g, r): true_att}) tuple.
    """
    dgp = StaggeredDGP(n_units=n_units, n_periods=n_periods, seed=seed)
    data = dgp.generate()
    true_atts = dgp.get_all_true_atts()
    return data, true_atts

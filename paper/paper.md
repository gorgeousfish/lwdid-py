---
title: 'lwdid: Lee–Wooldridge Difference-in-Differences with Rolling Transformations for Panel Data'
tags:
  - Python
  - difference-in-differences
  - causal inference
  - panel data
  - econometrics
  - staggered adoption
  - small samples
authors:
  - name: Xuanyu Cai
    orcid: 0009-0003-0980-8081
    affiliation: "1"
  - name: Wenli Xu
    affiliation: "1"
affiliations:
  - name: City University of Macau, Macau SAR, China
    index: 1
date: 11 February 2026
bibliography: paper.bib
---
# Summary

`lwdid` is a Python package implementing the rolling-transformation difference-in-differences (DiD) methods proposed by Lee and Wooldridge [-@lee:2025; -@lee:2026]. The core innovation is a unit-specific time-series transformation that converts panel DiD into a standard cross-sectional treatment effects problem: for each unit $i$, the pre-treatment outcomes are used to estimate a mean (or a linear trend), and the estimated mean (or the out-of-sample trend prediction) is then subtracted from each post-treatment outcome to produce a transformed dependent variable. Under the demeaning transformation, for an intervention starting at period $S$, the transformed outcome is

$$
\dot{Y}_{it} = Y_{it} - \frac{1}{S-1}\sum_{r=1}^{S-1} Y_{ir}, \quad t = S, \ldots, T
$$

Once transformed, any standard cross-sectional treatment effect estimator can be applied, including regression adjustment (RA), inverse probability weighting (IPW), the doubly robust IPWRA, and propensity score matching (PSM). `lwdid` supports both common timing and staggered adoption designs, provides four rolling transformation methods (demeaning, detrending, and their seasonal-adjustment variants), and implements exact $t$-distribution inference for the RA estimator under classical linear model assumptions, with degrees of freedom $N-K-2$ (so $N \geq 3$ when no additional covariates are included) [@lee:2026].

# Statement of Need

Difference-in-differences (DiD) is one of the most widely used causal inference methods in the social sciences. In recent years, DiD methodology has undergone substantial revision: de Chaisemartin and D'Haultfœuille [-@dechaisemartin:2020] showed that the conventional two-way fixed effects (TWFE) estimator identifies a weighted sum of average treatment effects with potentially negative weights, and Goodman-Bacon [-@goodmanbacon:2021] further decomposed the TWFE estimator as a weighted average of all possible two-group/two-period DiD estimators, revealing that some comparisons use already-treated units as controls and can be severely biased when treatment effects vary over time. These findings spurred the development of new DiD methods including those by Callaway and Sant'Anna [-@callaway:2021], Sun and Abraham [-@sun:2021], and Wooldridge [-@wooldridge:2021].

However, existing DiD implementations share three key limitations. First, inference in most current tools relies on large-sample approximations ($N \to \infty$), which can be unreliable when the number of treated or control units is small — a problem systematically discussed by Donald and Lang [-@donald:2007]. In particular, few user-facing implementations provide the exact $t$-distribution inference developed for small cross-sectional samples in Lee and Wooldridge [-@lee:2026]. Second, many existing methods estimate each group-time effect using a long difference from a single reference period (for example, Callaway and Sant'Anna [-@callaway:2021] use the transformation $Y_{ir} - Y_{i,g-1}$, referencing only period $g-1$), and do not implement unit-specific rolling transformations that exploit the full pre-treatment history to remove heterogeneous intercepts or trends. Third, to our knowledge, the Python ecosystem does not yet provide a dedicated implementation of the Lee and Wooldridge rolling-transformation approach; existing Python packages for DiD and fixed-effects regression (e.g., `differences` [@dionisi:2023] and `pyfixest` [@fischer:2024]) do not implement rolling transformations or exact small-sample inference.

`lwdid` implements the rolling-transformation methods of Lee and Wooldridge [-@lee:2025; -@lee:2026], addressing all three gaps. The package targets applied econometricians, policy evaluation analysts, and instructors and students in econometrics courses, providing a complete analytical pipeline from data transformation to inference.

# State of the Field

A variety of methods and software implementations have emerged to address heterogeneous treatment effects in DiD settings. In the Stata and R ecosystems, `csdid` [implementing @callaway:2021] provides doubly robust estimators based on long differencing, `did_multiplegt` [implementing @dechaisemartin:2020] provides alternative estimators under heterogeneous effects, `eventstudyinteract` [implementing @sun:2021] provides interaction-weighted event-study estimators, and `did_imputation` [implementing @borusyak:2024] provides the imputation approach. In Python, `differences` [@dionisi:2023] implements the Callaway and Sant'Anna method, and `pyfixest` [@fischer:2024] provides high-dimensional fixed effects regression with Sun-Abraham event-study estimators. None of these tools implement the Lee and Wooldridge rolling-transformation approach.

`lwdid` differs from existing tools in several respects. It provides exact small-sample inference: under classical linear model assumptions, the $t$-statistic for the RA estimator follows an exact $\mathcal{T}_{N-K-2}$ distribution [@lee:2026], where $K$ is the number of covariates, enabling valid inference with as few as $N \geq 3$ units (when $K = 0$), whereas most existing tools rely on large-sample approximations. It removes heterogeneous intercepts or trends through unit-specific rolling transformations that estimate pre-treatment patterns using only pre-treatment data and extrapolate out of sample, whereas existing tools typically rely on long differences or within-estimator approaches that do not separate trend estimation from treatment effect estimation in this way. It also supports seasonal adjustment (demeaning-with-deseasonalization and detrending-with-deseasonalization transformations) for panel data with seasonal patterns (quarterly, monthly, or weekly frequencies), and includes built-in diagnostic toolkits for trend assessment, sensitivity analysis, and clustering.

`lwdid` implements a fundamentally new methodology rather than reimplementing existing methods. The rolling-transformation approach requires redesigning the data pipeline from the ground up: panel data are first transformed at the unit level, then cross-sectional regressions are run for each period (or each cohort-period pair). This differs architecturally from existing packages — Callaway and Sant'Anna [-@callaway:2021] use the long difference from period $g-1$ ($\mathring{Y}_{irg} = Y_{ir} - Y_{i,g-1}$), whereas `lwdid` uses the mean over all pre-treatment periods ($\dot{Y}_{irg} = Y_{ir} - \frac{1}{g-1}\sum_{s=1}^{g-1} Y_{is}$). The two pipelines diverge at the transformation step and cannot be reconciled as extensions of one another. Moreover, the cross-sectional regression framework in `lwdid` allows flexibly applying different estimators (RA, IPW, IPWRA, PSM) at each $(g,r)$ pair using the rolling-transformed outcome, a capability not directly available within existing package architectures that are built around long differencing.

# Software Design

The central design decision in `lwdid` is decomposing panel DiD analysis into a three-stage pipeline: panel data → unit-specific transformation → cross-sectional regression. This design follows directly from the theoretical insight of Lee and Wooldridge [-@lee:2025]: after the rolling transformation removes pre-treatment patterns, the treatment (or cohort) assignment satisfies an unconfoundedness condition (in the conditional mean sense) with respect to the transformed potential outcome, conditional on covariates [see @lee:2025, Theorem 4.1], so any standard cross-sectional treatment effect estimator can be applied. The theoretical foundation rests on two key results. First, Lee and Wooldridge [-@lee:2025, Theorem 3.1] prove algebraic equivalence between the rolling-transformation RA estimator and the POLS/TWFE estimator ($\tilde{\tau}_r = \hat{\tau}_r$) in the common timing case. Second, Wooldridge [-@wooldridge:2021] discusses related algebraic equivalences for TWFE, pooled OLS implementations, and imputation-style estimators. These equivalence results motivate implementing the rolling transformation as a preprocessing step that enables the use of IPW [@abadie:2005], IPWRA [@wooldridge:2007; @santanna:2020], and PSM estimators. `lwdid` supports four transformation methods: demeaning [Procedure 2.1 in @lee:2026], detrending [Procedure 3.1 in @lee:2026], and their seasonal-adjustment variants. The detrending transformation fits a unit-specific linear trend via OLS on pre-treatment periods and extrapolates to post-treatment periods:

$$
\ddot{Y}_{it} = Y_{it} - \hat{A}_i - \hat{B}_i \cdot t, \quad t = S, \ldots, T
$$

where $\hat{A}_i$, $\hat{B}_i$ are obtained from regressing $Y_{it}$ on $1, t$ for $t = 1, \ldots, S-1$. In the staggered adoption design, the pre-treatment mean is computed by cohort $g$ (using $s = 1, \ldots, g-1$), independently of calendar time $r$:

$$
\dot{Y}_{irg} = Y_{ir} - \frac{1}{g-1}\sum_{s=1}^{g-1} Y_{is}
$$

For inference, `lwdid` uses the exact $\mathcal{T}_{N-K-2}$ distribution for the RA estimator under classical linear model assumptions (conditional normality and homoskedasticity) [@lee:2026], enabling valid inference even with very small samples. When heteroskedasticity is suspected, HC3 robust standard errors [@mackinnonwhite:1985] are available for all estimators. For IPW, IPWRA, and PSM, inference is based on normal asymptotics. Cluster-robust inference is supported via wild cluster bootstrap [@mackinnon:2018], with general guidance on cluster-robust methods following Cameron and Miller [-@cameron:2015]. A unified `lwdid()` function dispatches between common timing and staggered adoption designs via parameter selection, minimizing the learning curve. The package builds on NumPy [@harris:2020], pandas [@mckinney:2010], SciPy [@virtanen:2020], and statsmodels [@seabold:2010].

# Research Impact Statement

The package provides three reproducible empirical examples with accompanying Jupyter notebooks: the Castle Law dataset (a staggered adoption design with 21 treated and 29 never-treated states) [@cheng:2013; @cunningham:2021], the California Smoking dataset (a common timing design with one treated unit and 38 controls) [@abadie:2010], and the Walmart dataset (a detrending application illustrating unit-specific trend removal, using data from @brown:2025 and @basker:2005).

`lwdid` can be installed via `pip` and includes Sphinx-based documentation with a Read the Docs configuration.

# Acknowledgements

We thank Jeffrey M. Wooldridge and Soo Jeong Lee for developing the rolling-transformation DiD framework. We also thank the open-source community for feedback during development.

# References

"""Synthetic identification demo for Paper 1, §7.2 -- method-validation only.

Shows OLS recovers the elasticities under exogeneity, is biased under
simultaneity, and that an ORACLE conditioning on the true productivity shock
removes the bias. The oracle is infeasible (it observes A, which no econometrician
does); it establishes that the bias is confounding, not that any feasible
control-function estimator solves the multi-latent-factor case. Ground-truth
Monte Carlo, not real data.
"""
import numpy as np

from estimation import simulate_panel, ols

TRUE = (0.35, 0.35, 0.30)
N = 20_000


def _rng():
    return np.random.default_rng(20260710)


def test_ols_recovers_under_exogeneity():
    """Factors uncorrelated with the productivity shock -> OLS is consistent."""
    logY, X, _ = simulate_panel(N, *TRUE, endogenous=False, rng=_rng())
    a, b, g = ols(logY, X)
    for hat, true in zip((a, b, g), TRUE):
        assert abs(hat - true) < 0.03, f"exogenous OLS should recover {true}, got {hat:.3f}"


def test_simultaneity_biases_ols_upward():
    """Factor choice loading on the unobserved shock -> upward bias (Paper 1 §7.2)."""
    logY, X, _ = simulate_panel(N, *TRUE, endogenous=True, rng=_rng())
    a, b, g = ols(logY, X)
    assert (a + b + g) > sum(TRUE) + 0.05, "endogenous factor choice must bias the elasticity sum upward"


def test_oracle_conditioning_removes_bias():
    """Oracle: conditioning on the TRUE shock removes the bias (infeasible, not a proxy)."""
    logY, X, logA = simulate_panel(N, *TRUE, endogenous=True, rng=_rng())
    slopes = ols(logY, np.column_stack([X, logA]))
    a, b, g = slopes[:3]
    for hat, true in zip((a, b, g), TRUE):
        assert abs(hat - true) < 0.03, f"conditioning on the true A should recover {true}, got {hat:.3f}"

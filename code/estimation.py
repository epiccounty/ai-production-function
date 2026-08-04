"""
Paper 1 — synthetic estimation / identification demonstration.

METHOD-VALIDATION ONLY. This module generates panel data from the Stage-1
production function  Y = A * C^alpha * X^beta * W^gamma  with KNOWN elasticities,
then shows three things the paper argues in §7 (Operationalization & Identification):

  (1) OLS on logs recovers the elasticities when factors are exogenous;
  (2) an UPWARD bias appears when factor choice correlates with the unobserved
      productivity shock A -- OLS is biased upward. The simulation below models
      this as *selection on the shock* (firms that draw a high A invest more),
      which is one microfoundation of the endogeneity Paper 1, §7.2 discusses under
      the broader Marschak-Andrews simultaneity banner; both share the upward
      direction and the control-function remedy, though they are not identical
      mechanisms. The bias direction is what this demonstration establishes;
      distinguishing selection from simultaneity proper is not what it does;
  (3) an ORACLE that conditions on the *true* shock A removes the bias -- confirming
      that (2) is a confounding problem.

Row (3) is deliberately an oracle, NOT a feasible estimator: it regresses on the
ground-truth logA that the simulation generates but that no econometrician observes.
A working control function (Olley-Pakes 1996, Levinsohn-Petrin 2003) never sees A;
it PROXIES the shock through monotone investment / intermediate-input demand. Building
that proxy for the multi-latent-factor case (C, X both latent, plus A) is the open
problem stated in Paper 1 §7.2 -- this module does not solve it.

So this demonstrates the identification PROBLEM on ground truth and shows the bias is
confounding, not that any feasible estimator fixes it. It is NOT empirical evidence
about real AI firms -- there is no real data here, by design. Verified by
tests/test_estimation.py.
"""
from __future__ import annotations
import numpy as np


def simulate_panel(n, alpha, beta, gamma, *, endogenous, rng, load=0.8, sigma_A=0.5):
    """Draw a synthetic cross-section. Returns (logY, X, logA).

    ``logA`` is the productivity shock, unobserved by the econometrician. When
    ``endogenous`` is True, factor choices load on logA (firms with a favorable
    shock invest more), so the regressors correlate with the error term -- the
    simultaneity of Paper 1, §7.2.
    """
    logA = rng.normal(0.0, sigma_A, size=n)
    base = load * logA if endogenous else np.zeros(n)
    logC = rng.normal(1.0, 0.4, size=n) + base
    logX = rng.normal(1.0, 0.4, size=n) + base
    logW = rng.normal(1.0, 0.4, size=n) + base
    logY = logA + alpha * logC + beta * logX + gamma * logW
    return logY, np.column_stack([logC, logX, logW]), logA


def ols(logY, X):
    """OLS with intercept; returns the slope vector (intercept dropped)."""
    design = np.column_stack([np.ones(len(logY)), X])
    coef, *_ = np.linalg.lstsq(design, logY, rcond=None)
    return coef[1:]


def _demo():
    true = (0.35, 0.35, 0.30)
    rng = np.random.default_rng(20260710)
    n = 20_000

    logY, X, _ = simulate_panel(n, *true, endogenous=False, rng=rng)
    exo = ols(logY, X)

    logY, X, logA = simulate_panel(n, *true, endogenous=True, rng=rng)
    endo = ols(logY, X)
    # Oracle: conditions on the TRUE shock logA (infeasible in practice -- see module docstring).
    oracle = ols(logY, np.column_stack([X, logA]))[:3]

    print(f"true elasticities      : alpha,beta,gamma = {true}, sum = {sum(true):.2f}")
    print(f"(1) exogenous OLS      : {np.round(exo, 3)}, sum = {exo.sum():.3f}  -> recovered")
    print(f"(2) endogenous OLS     : {np.round(endo, 3)}, sum = {endo.sum():.3f}  -> biased upward (simultaneity)")
    print(f"(3) oracle (shock obs) : {np.round(oracle, 3)}, sum = {oracle.sum():.3f}  -> bias removed (oracle, not feasible)")


if __name__ == "__main__":
    _demo()

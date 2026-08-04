"""Shared estimators: OLS, VIF, rolling-origin CV, model tests, feasible Levinsohn-Petrin.

numpy-only, matching the style of code/paper1/estimation.py. The LP estimator is
the *feasible* counterpart of paper1's oracle demo: it never sees the true shock,
it proxies it with intermediate-input demand (compute spend), which is exactly
what the pilot design (SS3 E1) requires. It is a simplified two-stage LP -- a
third-degree polynomial first stage and a grid-searched second-stage moment
condition -- and reports its own diagnostics rather than pretending to be the
full ACF/GMM machinery. Its stage 1 shares the Ackerberg-Caves-Frazer critique
of pre-ACF LP designs (gamma is identified in stage 1 only under a timing
assumption on W); the paper states this caveat where the estimator is used.

Two generations of statistics coexist here on purpose. The functions the
defective ("as shipped") path used are frozen bit-for-bit so that the historical
artifact stays reproducible: ``vuong`` (misapplied there to a *nested* pair),
``rolling_cv_rmse`` (full-sample demeaning upstream leaked unit means across the
train/test split), ``bootstrap_sum_ci`` (iid residual resampling ignoring the
cluster structure), and ``ols_t`` with ``n_absorbed=0`` (degrees of freedom
uncorrected for absorbed unit means). The corrected path uses the fixed
counterparts added below: ``f_test_nested`` (the multiplicative pooled model is
the restriction alpha == beta of the split model, so the comparison is an F/LR
test, not a Vuong test), ``rolling_cv_rmse_fe`` (train-window-only unit means),
``bootstrap_sum_ci_wild_cluster`` (Rademacher wild cluster bootstrap), and
``ols_t(..., n_absorbed=G-1)``.
"""
from __future__ import annotations

import math

import numpy as np


def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS with intercept. Returns (slopes, residuals)."""
    design = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef[1:], y - design @ coef


def ols_t(y: np.ndarray, X: np.ndarray, n_absorbed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """OLS slopes and their t-statistics (homoskedastic).

    ``n_absorbed`` counts parameters absorbed upstream of this regression (for a
    within transform over G units, G - 1 unit means beyond the intercept). The
    defective path used the default 0, overstating the degrees of freedom on the
    within-demeaned panel; the corrected path passes G - 1.
    """
    design = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    dof = max(len(y) - design.shape[1] - n_absorbed, 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.maximum(np.diag(cov), 1e-30))
    return coef[1:], (coef / se)[1:]


# --------------------------------------------------------------------------- #
# Distribution helpers (numpy/stdlib only): regularized incomplete beta, F and
# t upper-tail probabilities. Continued-fraction evaluation follows the
# standard Lentz scheme.
# --------------------------------------------------------------------------- #
def _betacf(a: float, b: float, x: float) -> float:
    MAXIT, EPS, FPMIN = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betainc_reg(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                + a * math.log(x) + b * math.log(1.0 - x))
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def f_sf(F: float, d1: float, d2: float) -> float:
    """Upper-tail probability P(F_{d1,d2} > F)."""
    if not np.isfinite(F) or F <= 0.0:
        return 1.0
    return _betainc_reg(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * F))


def t_sf_two_sided(t: float, dof: float) -> float:
    """Two-sided p-value for a t statistic."""
    return f_sf(t * t, 1.0, dof)


def f_test_nested(y: np.ndarray, X_unrestricted: np.ndarray,
                  X_restricted: np.ndarray, q: int,
                  n_absorbed: int = 0) -> dict:
    """F test of a linear restriction via restricted vs unrestricted RSS.

    The multiplicative pooled model  Y = A (CX)^theta W^gamma  is the split
    model under the restriction alpha == beta (q = 1), so the pooled-vs-split
    comparison is *nested* and belongs here, not in a Vuong test: Vuong's
    normal statistic is inapplicable to strictly nested pairs (Vuong 1989).
    ``n_absorbed`` is forwarded to the residual degrees of freedom exactly as
    in ``ols_t``.
    """
    _, resid_u = ols(y, X_unrestricted)
    _, resid_r = ols(y, X_restricted)
    rss_u = float(resid_u @ resid_u)
    rss_r = float(resid_r @ resid_r)
    dof = max(len(y) - X_unrestricted.shape[1] - 1 - n_absorbed, 1)
    F = ((rss_r - rss_u) / q) / (rss_u / dof)
    return {"F": float(F), "p": float(f_sf(F, q, dof)), "dof_num": q,
            "dof_den": int(dof), "reject_5pct": bool(f_sf(F, q, dof) < 0.05)}


def vif(X: np.ndarray) -> np.ndarray:
    """Variance inflation factor of each column regressed on the others."""
    k = X.shape[1]
    out = np.empty(k)
    for j in range(k):
        others = np.delete(X, j, axis=1)
        _, resid = ols(X[:, j], others)
        tss = float(((X[:, j] - X[:, j].mean()) ** 2).sum())
        r2 = 1.0 - float(resid @ resid) / max(tss, 1e-30)
        out[j] = 1.0 / max(1.0 - r2, 1e-12)
    return out


def rolling_cv_rmse(y: np.ndarray, X: np.ndarray, min_train: int = 26) -> float:
    """Rolling-origin one-step-ahead out-of-sample RMSE (pilot design SS3 E2).

    Frozen defective-path variant: it assumes the caller's arrays are already
    within-demeaned, so unit means computed on the *full* sample leak future
    information into every "out-of-sample" prediction. Kept for the as-shipped
    reading only; the corrected path uses ``rolling_cv_rmse_fe``.
    """
    errs = []
    for t in range(min_train, len(y)):
        coef, _ = ols(y[:t], X[:t])
        icept = y[:t].mean() - X[:t].mean(axis=0) @ coef
        errs.append(y[t] - (icept + X[t] @ coef))
    return float(np.sqrt(np.mean(np.square(errs))))


def rolling_cv_rmse_fe(y_raw: np.ndarray, X_raw: np.ndarray, ids: np.ndarray,
                       min_train: int = 26) -> dict:
    """Leakage-free rolling-origin CV for a fixed-effects predictor.

    Arrays must be time-ordered and *not* pre-demeaned. At each origin the unit
    means are computed on the training window only; the test observation is
    predicted from its unit's training mean plus the within slopes. Test rows
    whose unit has no training observation are skipped (counted in
    ``n_skipped``), because a fixed-effects model has no prediction for a unit
    it has never seen.
    """
    n = len(y_raw)
    errs, skipped = [], 0
    for t in range(min_train, n):
        tr = slice(0, t)
        ids_tr = ids[tr]
        if ids[t] not in ids_tr:
            skipped += 1
            continue
        y_tr, X_tr = y_raw[tr].astype(float).copy(), X_raw[tr].astype(float).copy()
        my, mX = {}, {}
        for u in np.unique(ids_tr):
            m = ids_tr == u
            my[u] = y_tr[m].mean()
            mX[u] = X_tr[m].mean(axis=0)
            y_tr[m] -= my[u]
            X_tr[m] -= mX[u]
        coef, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        u = ids[t]
        pred = my[u] + (X_raw[t] - mX[u]) @ coef
        errs.append(y_raw[t] - pred)
    return {"rmse": float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan"),
            "n_evaluated": int(len(errs)), "n_skipped": int(skipped)}


def vuong(y: np.ndarray, X_a: np.ndarray, X_b: np.ndarray) -> float:
    """Vuong z for non-nested model comparison under Gaussian likelihoods.

    Positive z favors model A. |z| > 1.96 is the usual 5% cut.
    """
    n = len(y)
    lls = []
    for X in (X_a, X_b):
        _, resid = ols(y, X)
        s2 = max(float(resid @ resid) / n, 1e-30)
        lls.append(-0.5 * (np.log(2 * np.pi * s2) + resid**2 / s2))
    d = lls[0] - lls[1]
    sd = float(d.std(ddof=1))
    return float(np.sqrt(n) * d.mean() / max(sd, 1e-12))


def lp_feasible(logY: np.ndarray, logC: np.ndarray, logX: np.ndarray,
                logW: np.ndarray, logM: np.ndarray,
                grid: int = 61, span: float = 0.9,
                ids: np.ndarray | None = None,
                t: np.ndarray | None = None) -> dict:
    """Simplified feasible Levinsohn-Petrin with a variable-input proxy.

    Stage 1: logY = gamma*logW + Phi(logC, logX, logM) + e, with Phi a full
             third-degree polynomial -> recovers gamma and the composite Phi-hat.
    Stage 2: for candidate (alpha, beta), omega-hat = Phi-hat - alpha*logC -
             beta*logX follows an AR(1); the innovation xi must be orthogonal to
             the quasi-fixed stocks. Grid-search (alpha, beta) minimizing the
             moment norm  || E[xi * (logC_t, logX_t)] ||.

    ``ids`` (and optionally ``t``) carry the panel structure. The stage-2 AR(1)
    is a *time* lag within a productive unit, so on a stacked multi-repo panel
    the lag pairs MUST be formed inside a repo: with ``ids=None`` the pairs are
    simply adjacent rows, which on the repo-major stack ``reproduce_section9``
    builds (it sorts by repo, then week) pairs rows from *different* repos and
    estimates a cross-sectional correlation instead of an AR(1). Pass ``ids``
    on any pooled panel. When ``t`` is given, pairs are further restricted to
    consecutive periods, so a gap left by dropped observations never silently
    becomes a one-period lag.

    Returns estimates plus the achieved moment norm and a boundary flag --
    honest diagnostics, because the multi-latent-factor case (paper1 SS7.2) may
    simply not be identified; documenting the breakdown is a result.
    """
    def poly3(*cols: np.ndarray) -> np.ndarray:
        terms = list(cols)
        k = len(cols)
        for i in range(k):
            for j in range(i, k):
                terms.append(cols[i] * cols[j])
                for l in range(j, k):
                    terms.append(cols[i] * cols[j] * cols[l])
        return np.column_stack(terms)

    Z = poly3(logC, logX, logM)
    stage1 = np.column_stack([logW, Z])
    coef, _ = ols(logY, stage1)
    gamma = float(coef[0])
    icept = logY.mean() - stage1.mean(axis=0) @ coef
    phi = logY - gamma * logW  # Phi-hat + e; poly part absorbs the shock
    phi_fit = icept + Z @ coef[1:]
    phi_hat = phi_fit  # smoothed composite Phi(C, X, omega)

    # Stage-2 lag pairs (cur, prev). Within repo when ids is given; restricted to
    # consecutive periods when t is given.
    if ids is None:
        cur = np.arange(1, len(logY))
        prev = np.arange(0, len(logY) - 1)
    else:
        cur_l, prev_l = [], []
        for u in np.unique(ids):
            idx = np.flatnonzero(ids == u)
            if t is not None:
                idx = idx[np.argsort(t[idx])]
                consec = np.isclose(np.diff(t[idx]), 1.0)
                c, p = idx[1:][consec], idx[:-1][consec]
            else:
                c, p = idx[1:], idx[:-1]
            if len(c):
                cur_l.append(c)
                prev_l.append(p)
        if not cur_l:
            return {"alpha": float("nan"), "beta": float("nan"), "gamma": gamma,
                    "moment_norm": float("nan"), "at_boundary": True, "n_lag_pairs": 0}
        cur, prev = np.concatenate(cur_l), np.concatenate(prev_l)

    lo_a, hi_a = 0.01, span
    best = {"alpha": np.nan, "beta": np.nan, "moment_norm": np.inf}
    a_grid = np.linspace(lo_a, hi_a, grid)
    for a in a_grid:
        for b in np.linspace(lo_a, hi_a, grid):
            omega = phi_hat - a * logC - b * logX
            rho, resid = ols(omega[cur], omega[prev, None])
            xi = resid
            m = np.array([float(xi @ logC[cur]) / len(xi), float(xi @ logX[cur]) / len(xi)])
            norm = float(np.sqrt(m @ m))
            if norm < best["moment_norm"]:
                best = {"alpha": float(a), "beta": float(b),
                        "moment_norm": norm, "rho": float(rho[0])}
    best["n_lag_pairs"] = int(len(cur))
    best["gamma"] = gamma
    best["at_boundary"] = bool(
        min(abs(best["alpha"] - lo_a), abs(hi_a - best["alpha"]),
            abs(best["beta"] - lo_a), abs(hi_a - best["beta"])) < (hi_a - lo_a) / (grid - 1) * 1.5
    )
    return best


def bootstrap_sum_ci(y: np.ndarray, X: np.ndarray, n_boot: int = 400,
                     seed: int = 20260716) -> tuple[float, float, float]:
    """Residual-bootstrap CI for the sum of OLS slopes (returns-to-scale, E3).

    Frozen defective-path variant: iid residual resampling ignores the panel's
    cluster structure. Kept for the as-shipped reading only; the corrected path
    uses ``bootstrap_sum_ci_wild_cluster``.
    """
    coef, resid = ols(y, X)
    icept = y.mean() - X.mean(axis=0) @ coef
    fitted = icept + X @ coef
    rng = np.random.default_rng(seed)
    sums = np.empty(n_boot)
    for b in range(n_boot):
        yb = fitted + rng.choice(resid, size=len(y), replace=True)
        cb, _ = ols(yb, X)
        sums[b] = cb.sum()
    lo, hi = np.percentile(sums, [2.5, 97.5])
    return float(coef.sum()), float(lo), float(hi)


def bootstrap_sum_ci_wild_cluster(y: np.ndarray, X: np.ndarray, ids: np.ndarray,
                                  n_boot: int = 999, seed: int = 20260716
                                  ) -> tuple[float, float, float]:
    """Wild cluster bootstrap CI for the sum of OLS slopes (clusters = repos).

    Rademacher weights are drawn per cluster, so within-repo residual
    dependence survives resampling (Cameron-Gelbach-Miller). With G = 18
    clusters this is small-sample territory; the CI is reported as a
    cluster-aware bound, not an exact interval.
    """
    coef, resid = ols(y, X)
    icept = y.mean() - X.mean(axis=0) @ coef
    fitted = icept + X @ coef
    rng = np.random.default_rng(seed)
    uniq = np.unique(ids)
    sums = np.empty(n_boot)
    for b in range(n_boot):
        w = rng.choice([-1.0, 1.0], size=len(uniq))
        wu = {u: w[i] for i, u in enumerate(uniq)}
        yb = fitted + resid * np.array([wu[u] for u in ids])
        cb, _ = ols(yb, X)
        sums[b] = cb.sum()
    lo, hi = np.percentile(sums, [2.5, 97.5])
    return float(coef.sum()), float(lo), float(hi)

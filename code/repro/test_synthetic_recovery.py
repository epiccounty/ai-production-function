"""Appendix A: estimator validation on synthetic repository-week panels.

Ground-truth checks promised by the paper's reproducibility appendix: panels
generated from the split Cobb-Douglas with KNOWN elasticities must be recovered
by the within estimator; the H4 comparison must prefer the split model on
split-generated data and fail to reject the pooled restriction when the data
are generated with alpha == beta; and a panel whose stocks do not move within
repositories must be flagged as non-identified rather than yielding spurious
stock elasticities.

These are synthetic-data validations of the *machinery*, not evidence about
any real economy (paper, Appendix A).
"""
from __future__ import annotations

import numpy as np

import reproduce_section9 as r9
from estimators import f_test_nested, ols, rolling_cv_rmse_fe

DELTA_C, DELTA_X = 0.02, 0.04


def make_panels(G=18, T=104, alpha=0.35, beta=0.35, gamma=0.30,
                sigma_shock=0.10, static_stocks=False, seed=20260804):
    """Synthetic repo-week panels in the reproduce_section9 format."""
    rng = np.random.default_rng(seed)
    panels = {}
    weeks = [f"W{i:03d}" for i in range(T)]
    for g in range(G):
        base_ic = rng.uniform(50.0, 500.0)
        base_ix = rng.uniform(20.0, 200.0)
        if static_stocks:
            I_c = np.full(T, base_ic)
            I_x = np.full(T, base_ix)
        else:
            # trending, noisy investment so the PIM stocks move within-repo
            trend = np.exp(rng.normal(0.0, 0.02) * np.arange(T))
            I_c = base_ic * trend * rng.lognormal(0.0, 0.4, T)
            I_x = base_ix * trend * rng.lognormal(0.0, 0.4, T)
        C = np.empty(T)
        X = np.empty(T)
        c = 0.0 if not static_stocks else base_ic / DELTA_C
        x = 0.0 if not static_stocks else base_ix / DELTA_X
        for t in range(T):
            c = (1 - DELTA_C) * c + I_c[t]
            x = (1 - DELTA_X) * x + I_x[t]
            C[t], X[t] = c, x
        W = rng.lognormal(rng.normal(1.0, 0.3), 0.5, T)
        fe = rng.normal(0.0, 0.3)
        logA = fe + rng.normal(0.0, sigma_shock, T)
        Y = np.exp(logA) * C**alpha * X**beta * W**gamma
        panels[f"repo{g:02d}"] = {"weeks": list(weeks), "cols": {
            "Y": Y, "C": C, "X": X, "W": W,
            "I_c": I_c, "I_x": I_x,
            "input_tokens": W * 1e6 * rng.lognormal(0.0, 0.2, T),
        }}
    return panels


def _within_fit(panels):
    widx = {w: i for i, w in enumerate(sorted(
        {w for v in panels.values() for w in v["weeks"]}))}
    L, ids, _t, _d, raw = r9._stack(panels, widx, apply_filter=True, proxy="input")
    split = np.column_stack([L["C"], L["X"], L["W"]])
    coef, _ = ols(L["Y"], split)
    return L, ids, raw, split, coef


def test_within_estimator_recovers_known_elasticities():
    panels = make_panels(alpha=0.35, beta=0.35, gamma=0.30)
    _L, _ids, _raw, _split, coef = _within_fit(panels)
    for got, want in zip(coef, (0.35, 0.35, 0.30)):
        assert abs(got - want) < 0.06


def test_h4_prefers_split_on_split_generated_data():
    panels = make_panels(alpha=0.45, beta=0.25)
    L, ids, raw, split, _coef = _within_fit(panels)
    pooled = np.column_stack([L["C"] + L["X"], L["W"]])
    ft = f_test_nested(L["Y"], split, pooled, q=1,
                       n_absorbed=len(np.unique(ids)) - 1)
    assert ft["reject_5pct"] is True
    cv_split = rolling_cv_rmse_fe(raw["Y"], np.column_stack(
        [raw["C"], raw["X"], raw["W"]]), ids)
    cv_pooled = rolling_cv_rmse_fe(raw["Y"], np.column_stack(
        [raw["C"] + raw["X"], raw["W"]]), ids)
    assert cv_split["rmse"] < cv_pooled["rmse"]


def test_h4_does_not_reject_pooling_when_alpha_equals_beta():
    panels = make_panels(alpha=0.35, beta=0.35)
    L, ids, _raw, split, _coef = _within_fit(panels)
    pooled = np.column_stack([L["C"] + L["X"], L["W"]])
    ft = f_test_nested(L["Y"], split, pooled, q=1,
                       n_absorbed=len(np.unique(ids)) - 1)
    assert ft["reject_5pct"] is False


def test_static_stocks_are_flagged_not_estimated():
    panels = make_panels(static_stocks=True)
    L, ids, _raw, _split, _coef = _within_fit(panels)
    # no within variation in the stocks: identification must fail loudly
    for k in ("C", "X"):
        v = L[k].copy()
        for u in np.unique(ids):
            m = ids == u
            v[m] -= v[m].mean()
        assert float(np.abs(v).max()) < 1e-8

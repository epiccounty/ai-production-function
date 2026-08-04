"""Reproduce Paper 1, Section 9's estimation numbers from the frozen panel.

Self-contained: this script imports only ``numpy``, the Python standard library,
and the sibling ``estimators`` module. It re-runs the exact estimators the paper
reports, on the frozen repository-week cells shipped in ``./panels/``, and emits
both readings side by side:

  as_shipped   the pre-fix path: no (W>0)&(Y>0) filter, LP stage-2 AR(1) lagged
               across the time-major stack, LP proxy = log(total tokens).
  corrected    the fixed path: missing-observation cells dropped, LP lags kept
               inside a repository and restricted to consecutive weeks, LP proxy
               = log(input tokens).

The panels are the studied organization's own records with repository identities
replaced by ``repoNN`` and the raw version-control history and session logs left
out (those are private and are not distributed); the estimation is unchanged by
the renaming, so every Section 9 number regenerates here.

Usage (from this directory):
    python3 reproduce_section9.py            # print both readings
    python3 reproduce_section9.py --check    # verify against results_paper1_corrected.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np

from estimators import (bootstrap_sum_ci, bootstrap_sum_ci_wild_cluster,
                        f_test_nested, lp_feasible, ols, ols_t,
                        rolling_cv_rmse, rolling_cv_rmse_fe, t_sf_two_sided,
                        vif, vuong)

EPS = 1e-9
VIF_THRESHOLD = 10.0
DEFAULT_PANELS = Path("panels")


def load_frozen_panels(panels_dir: Path | None = None) -> dict[str, dict]:
    """Read every ``panel_<repo>.csv`` back into {repo: {weeks, cols}}."""
    panels_dir = panels_dir or DEFAULT_PANELS
    out: dict[str, dict] = {}
    for path in sorted(glob.glob(str(panels_dir / "panel_*.csv"))):
        name = os.path.basename(path)[len("panel_"):-len(".csv")]
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        cols = {k: np.array([float(r[k]) for r in rows], dtype=float)
                for k in rows[0] if k != "week"}
        out[name] = {"weeks": [r["week"] for r in rows], "cols": cols}
    return out


def _within(x: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Demean by unit and add back the grand mean (within transform)."""
    out = x.astype(float).copy()
    for u in np.unique(ids):
        m = ids == u
        out[m] -= x[m].mean()
    return out + x.mean()


def _stack(panels: dict[str, dict], widx: dict[str, int], *,
           apply_filter: bool, proxy: str):
    ids_l, t_l = [], []
    raw: dict[str, list] = {k: [] for k in ("Y", "C", "X", "W", "input_tokens")}
    dropped = 0
    for uid, (_name, v) in enumerate(sorted(panels.items())):
        c, weeks = v["cols"], v["weeks"]
        keep = ((c["W"] > 0) & (c["Y"] > 0)) if apply_filter \
            else np.ones(len(weeks), dtype=bool)
        dropped += int((~keep).sum())
        if not keep.any():
            continue
        ids_l += [uid] * int(keep.sum())
        t_l += [widx[w] for w, k in zip(weeks, keep) if k]
        for k in raw:
            raw[k].append(c[k][keep])
    ids = np.array(ids_l)
    t = np.array(t_l, dtype=float)
    R = {k: np.concatenate(v) for k, v in raw.items()}
    logs = {k: np.log(np.maximum(R[k], EPS)) for k in ("Y", "C", "X", "W")}
    if proxy == "total":          # as shipped: log(input + output)
        logs["spend"] = np.log(np.maximum(R["input_tokens"] + R["W"] * 1e6, 1.0))
    else:                         # corrected: the variable input alone
        logs["spend"] = np.log(np.maximum(R["input_tokens"], 1.0))
    order = np.lexsort((ids, t))
    L = {k: _within(v, ids)[order] for k, v in logs.items()}
    raw = {k: v[order] for k, v in logs.items()}
    raw["lvl_C"] = R["C"][order]
    raw["lvl_X"] = R["X"][order]
    return L, ids[order], t[order], dropped, raw


def estimands(logY, logC, logX, logW, log_spend, min_train: int | None = None,
              n_dropped: int = 0, ids=None, t=None) -> dict:
    """E1-E3 on plain (possibly within-demeaned, time-sorted) arrays.

    ``ids``/``t`` carry the panel structure on a stacked multi-repo design; they
    are forwarded to the LP stage-2 AR(1) so its lag pairs stay inside a repo and
    span consecutive periods. Omit them only for a single-unit series.
    """
    n = len(logY)
    min_train = min_train or max(min(26, n // 2), 8)

    split_X = np.column_stack([logC, logX, logW])
    pooled_X = np.column_stack([logC + logX, logW])

    ols_coef, _ = ols(logY, split_X)
    lp = lp_feasible(logY, logC, logX, logW, log_spend, ids=ids, t=t)

    rmse_split = rolling_cv_rmse(logY, split_X, min_train)
    rmse_pooled = rolling_cv_rmse(logY, pooled_X, min_train)
    z = vuong(logY, split_X, pooled_X)
    v = vif(split_X)
    identified = bool(max(v[0], v[1]) < VIF_THRESHOLD)
    split_wins = bool(rmse_split < rmse_pooled and z > 1.96 and identified)

    total, lo, hi = bootstrap_sum_ci(logY, split_X)

    return {
        "n_weeks": n,
        "n_dropped": n_dropped,
        "E1": {
            "ols": {"alpha": float(ols_coef[0]), "beta": float(ols_coef[1]),
                    "gamma": float(ols_coef[2])},
            "lp_feasible": lp,
        },
        "E2_H4": {
            "rmse_split": rmse_split, "rmse_pooled": rmse_pooled,
            "vuong_z": z, "vif_logC": float(v[0]), "vif_logX": float(v[1]),
            "separately_identified": identified, "split_wins": split_wins,
            "decision": ("split model earns its place (H4 supported)" if split_wins
                         else "pooled model is the honest winner (H4 not supported)"),
        },
        "E3_returns": {
            "sum": total, "ci95": [lo, hi],
            "constant_returns_inside_ci": bool(lo <= 1.0 <= hi),
        },
    }


def _compute_only(logY: np.ndarray, logW: np.ndarray) -> dict:
    """Compute-only elasticity of W (the compute-in-the-labor-slot shape)."""
    coef, tstats = ols_t(logY, logW[:, None])
    return {"elasticity_W": float(coef[0]), "t_W": float(tstats[0])}


def _cluster_t(y: np.ndarray, X: np.ndarray, ids: np.ndarray) -> list[float]:
    """CR1 cluster-robust t-statistics (clusters = repos), intercept included."""
    D = np.column_stack([np.ones(len(y)), X])
    XtX_inv = np.linalg.pinv(D.T @ D)
    beta = XtX_inv @ (D.T @ y)
    u = y - D @ beta
    meat = np.zeros((D.shape[1], D.shape[1]))
    for g in np.unique(ids):
        m = ids == g
        s = D[m].T @ u[m]
        meat += np.outer(s, s)
    G, n, k = len(np.unique(ids)), len(y), D.shape[1]
    c = (G / (G - 1)) * ((n - 1) / (n - k))
    V = c * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.maximum(np.diag(V), 1e-30))
    return [float(t) for t in (beta / se)[1:]]


def _partial_r2(y: np.ndarray, X: np.ndarray, j: int) -> float:
    """Share of residual variance a column explains once the others are in."""
    others = [i for i in range(X.shape[1]) if i != j]
    _, r0 = ols(y, X[:, others])
    _, r1 = ols(y, X)
    ss0, ss1 = float(r0 @ r0), float(r1 @ r1)
    return (ss0 - ss1) / ss0 if ss0 > 0 else float("nan")


def reading(panels: dict[str, dict], widx: dict[str, int], *,
            apply_filter: bool, proxy: str, lp_panel_aware: bool,
            corrected_stats: bool = False) -> dict:
    L, ids, t, dropped, raw = _stack(panels, widx, apply_filter=apply_filter,
                                     proxy=proxy)
    est = estimands(
        L["Y"], L["C"], L["X"], L["W"], L["spend"], n_dropped=dropped,
        ids=(ids if lp_panel_aware else None), t=(t if lp_panel_aware else None))
    X = np.column_stack([L["C"], L["X"], L["W"]])
    est["partial_r2"] = {"C": _partial_r2(L["Y"], X, 0),
                         "X": _partial_r2(L["Y"], X, 1),
                         "W": _partial_r2(L["Y"], X, 2)}
    est["compute_only"] = _compute_only(L["Y"], L["W"])
    Xw = np.column_stack([L["W"], L["C"], L["X"]])
    coef_w, t_w = ols_t(L["Y"], Xw)
    est["stocks_controlled"] = {
        "elasticity_W": float(coef_w[0]), "t_naive": float(t_w[0]),
        "t_cluster_repo": _cluster_t(L["Y"], Xw, ids)[0],
    }
    est["proxy_corr_with_logW"] = float(np.corrcoef(L["spend"], L["W"])[0, 1])
    est["n_repos_with_variation"] = int(sum(
        1 for u in np.unique(ids) if (ids == u).sum() >= 2))
    if corrected_stats:
        est.update(_second_correction(L, ids, raw, est))
    return est


def _second_correction(L: dict, ids: np.ndarray, raw: dict, est: dict) -> dict:
    """Second-audit statistics for the corrected reading.

    Four repairs relative to the first correction, each disclosed in the paper
    (Appendix C(vii)): (1) the multiplicative pooled model is *nested* in the
    split model (alpha == beta), so H4's model-comparison leg is an F test, not
    a Vuong z; the Vuong statistic is retained only for the genuinely
    non-nested additive aggregate K = C + X. (2) Rolling-origin CV now computes
    unit means on the training window only (the first correction demeaned on
    the full sample, leaking future information into every "out-of-sample"
    prediction). (3) The returns-to-scale CI uses a wild cluster bootstrap
    (clusters = repos) instead of iid residual resampling. (4) Naive t
    statistics adjust degrees of freedom for the G - 1 absorbed repository
    means. First-correction values remain regenerable via the plain
    ``E2_H4`` / ``E3_returns`` / ``stocks_controlled.t_naive`` fields, which
    this block supersedes.
    """
    n = len(L["Y"])
    G = len(np.unique(ids))
    nab = G - 1
    min_train = max(min(26, n // 2), 8)

    split_w = np.column_stack([L["C"], L["X"], L["W"]])
    pooled_w = np.column_stack([L["C"] + L["X"], L["W"]])
    additive_w = np.column_stack([_within(np.log(raw["lvl_C"] + raw["lvl_X"]), ids), L["W"]])

    ft = f_test_nested(L["Y"], split_w, pooled_w, q=1, n_absorbed=nab)

    raw_split = np.column_stack([raw["C"], raw["X"], raw["W"]])
    raw_pooled = np.column_stack([raw["C"] + raw["X"], raw["W"]])
    raw_add = np.column_stack([np.log(raw["lvl_C"] + raw["lvl_X"]), raw["W"]])
    cv_split = rolling_cv_rmse_fe(raw["Y"], raw_split, ids, min_train)
    cv_pooled = rolling_cv_rmse_fe(raw["Y"], raw_pooled, ids, min_train)
    cv_add = rolling_cv_rmse_fe(raw["Y"], raw_add, ids, min_train)

    z_add = vuong(L["Y"], split_w, additive_w)

    identified = bool(est["E2_H4"]["separately_identified"])
    split_wins = bool(ft["reject_5pct"]
                      and cv_split["rmse"] < min(cv_pooled["rmse"], cv_add["rmse"])
                      and identified)

    total, lo, hi = bootstrap_sum_ci_wild_cluster(L["Y"], split_w, ids)

    Xw = np.column_stack([L["W"], L["C"], L["X"]])
    coef_w, t_w = ols_t(L["Y"], Xw, n_absorbed=nab)
    co_coef, co_t = ols_t(L["Y"], L["W"][:, None], n_absorbed=nab)

    return {
        "E2_H4_corrected": {
            "f_alpha_eq_beta": ft,
            "cv_fe_split": cv_split, "cv_fe_pooled": cv_pooled,
            "cv_fe_additive": cv_add,
            "vuong_z_vs_additive": float(z_add),
            "separately_identified": identified,
            "split_wins": split_wins,
            "decision": ("split model earns its place (H4 supported)" if split_wins
                         else "pooled model is the honest winner (H4 not supported)"),
        },
        "E3_returns_corrected": {
            "sum": total, "ci95_wild_cluster": [lo, hi],
            "constant_returns_inside_ci": bool(lo <= 1.0 <= hi),
            "n_clusters": G,
        },
        "stocks_controlled_corrected": {
            "elasticity_W": float(coef_w[0]),
            "t_dof_adjusted": float(t_w[0]),
            "p_two_sided": t_sf_two_sided(float(t_w[0]), n - 4 - nab),
            "dof": int(n - 4 - nab),
        },
        "compute_only_corrected": {
            "elasticity_W": float(co_coef[0]),
            "t_dof_adjusted": float(co_t[0]),
            "t_cluster_repo": _cluster_t(L["Y"], L["W"][:, None], ids)[0],
            "dof": int(n - 2 - nab),
        },
    }


def run(panels_dir: Path | None = None) -> dict:
    panels = load_frozen_panels(panels_dir)
    if len(panels) < 2:
        raise ValueError("need >= 2 frozen repo panels")
    all_weeks = sorted({w for v in panels.values() for w in v["weeks"]})
    widx = {w: i for i, w in enumerate(all_weeks)}
    n_obs = sum(len(v["weeks"]) for v in panels.values())
    zero_cells = int(sum(((v["cols"]["W"] <= 0) | (v["cols"]["Y"] <= 0)).sum()
                         for v in panels.values()))
    return {
        "source": "frozen anonymized panel (panels/panel_*.csv)",
        "n_repos": len(panels), "n_obs": n_obs,
        "weeks_span": [all_weeks[0], all_weeks[-1]],
        "missing_observation_cells": zero_cells,
        "as_shipped": reading(panels, widx, apply_filter=False,
                              proxy="total", lp_panel_aware=False),
        "corrected": reading(panels, widx, apply_filter=True,
                             proxy="input", lp_panel_aware=True,
                             corrected_stats=True),
    }


def _print(res: dict) -> None:
    print(f"frozen panel: {res['n_repos']} repos / {res['n_obs']} obs "
          f"({res['weeks_span'][0]}..{res['weeks_span'][1]})")
    print(f"missing-observation cells (W==0 or Y==0): "
          f"{res['missing_observation_cells']}/{res['n_obs']}")
    for tag, r in (("as shipped", res["as_shipped"]), ("corrected", res["corrected"])):
        e1, e3, h4 = r["E1"]["ols"], r["E3_returns"], r["E2_H4"]
        print(f"\n[{tag}] n={r['n_weeks']} dropped={r['n_dropped']}")
        print(f"  OLS   alpha={e1['alpha']:+.3f} beta={e1['beta']:+.3f} "
              f"gamma={e1['gamma']:+.3f}")
        print(f"  RTS   sum={e3['sum']:.3f} CI95=[{e3['ci95'][0]:.3f},{e3['ci95'][1]:.3f}] "
              f"constant_returns_inside={e3['constant_returns_inside_ci']}")
        print(f"  H4    rmse split={h4['rmse_split']:.4f} pooled={h4['rmse_pooled']:.4f} "
              f"vuong_z={h4['vuong_z']:.3f} VIF={h4['vif_logC']:.2f}/{h4['vif_logX']:.2f} "
              f"split_wins={h4['split_wins']}")
        lp = r["E1"]["lp_feasible"]
        print(f"  LP    alpha={lp['alpha']:.3f} beta={lp['beta']:.3f} "
              f"gamma={lp['gamma']:+.3f} at_boundary={lp['at_boundary']}")
        print(f"  partial R2  C={r['partial_r2']['C']:.4f} X={r['partial_r2']['X']:.4f} "
              f"W={r['partial_r2']['W']:.4f}")
        print(f"  compute-only elasticity_W={r['compute_only']['elasticity_W']:+.4f} "
              f"(t={r['compute_only']['t_W']:+.3f})   proxy corr with logW="
              f"{r['proxy_corr_with_logW']:.4f}")
        if "E2_H4_corrected" in r:
            h4c, e3c = r["E2_H4_corrected"], r["E3_returns_corrected"]
            ft = h4c["f_alpha_eq_beta"]
            print(f"  [second correction]")
            print(f"  H4    F(a=b)={ft['F']:.3f} (p={ft['p']:.4f}, dof={ft['dof_num']},{ft['dof_den']}) "
                  f"CV-FE split={h4c['cv_fe_split']['rmse']:.4f} "
                  f"pooled={h4c['cv_fe_pooled']['rmse']:.4f} "
                  f"additive={h4c['cv_fe_additive']['rmse']:.4f} "
                  f"vuong_vs_additive={h4c['vuong_z_vs_additive']:.3f} "
                  f"split_wins={h4c['split_wins']}")
            print(f"  RTS   sum={e3c['sum']:.3f} wild-cluster CI95="
                  f"[{e3c['ci95_wild_cluster'][0]:.3f},{e3c['ci95_wild_cluster'][1]:.3f}] "
                  f"constant_returns_inside={e3c['constant_returns_inside_ci']}")
            sc, co = r["stocks_controlled_corrected"], r["compute_only_corrected"]
            print(f"  W     stocks-controlled={sc['elasticity_W']:+.3f} "
                  f"(t_dof={sc['t_dof_adjusted']:+.2f}, p={sc['p_two_sided']:.4f}) "
                  f"compute-only={co['elasticity_W']:+.3f} "
                  f"(t_dof={co['t_dof_adjusted']:+.2f}, t_cluster={co['t_cluster_repo']:+.2f})")


def _flatten(o, prefix=""):
    """Yield (path, float) for every numeric leaf, skipping the source label."""
    if isinstance(o, dict):
        for k, v in o.items():
            if prefix == "" and k == "source":
                continue
            yield from _flatten(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from _flatten(v, f"{prefix}[{i}]")
    elif isinstance(o, bool):
        yield prefix, float(o)
    elif isinstance(o, (int, float)):
        yield prefix, float(o)


def check(res: dict, reference_path: Path, tol: float = 1e-6) -> int:
    ref = json.loads(Path(reference_path).read_text())
    got = dict(_flatten(res))
    want = dict(_flatten(ref))
    mism = []
    for key, wv in want.items():
        gv = got.get(key)
        if gv is None:
            mism.append((key, "missing", wv))
        elif not (np.isnan(wv) and np.isnan(gv)) and abs(gv - wv) > tol:
            mism.append((key, gv, wv))
    if mism:
        print(f"MISMATCH ({len(mism)} field(s), tol={tol}):")
        for key, gv, wv in mism[:20]:
            print(f"  {key}: reproduced={gv} reference={wv}")
        return 1
    print(f"OK: reproduced {len(want)} numeric fields match "
          f"{reference_path} within {tol}.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panels-dir", default=None)
    ap.add_argument("--check", action="store_true",
                    help="verify against results_paper1_corrected.json and exit non-zero on mismatch")
    ap.add_argument("--freeze", action="store_true",
                    help="write the current run to the reference JSON (refreezing the result)")
    ap.add_argument("--reference", default="results_paper1_corrected.json")
    args = ap.parse_args()
    res = run(Path(args.panels_dir) if args.panels_dir else None)
    if args.freeze:
        Path(args.reference).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
        print(f"froze {args.reference}")
        return 0
    if args.check:
        return check(res, Path(args.reference))
    _print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

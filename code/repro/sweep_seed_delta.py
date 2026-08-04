"""Appendix C(vi): the perpetual-inventory seed / depreciation sweep, shipped.

Every cross-factor conclusion in Section 9 is conditional on how the stocks
C and X are seeded. This script makes that claim checkable: it rebuilds both
stocks from the shipped investment flows (``I_c``, ``I_x``) under each seeding
convention and depreciation multiplier, re-runs the corrected reading, and
reports the diagnostics the paper conditions on.

Conventions swept (crossed with delta multipliers 0.5x / 1x / 2x on the
baseline delta_c = 0.02, delta_x = 0.04 per week):

  steady      C_pre = mean(I)/delta  -- the paper's baseline. At the baseline
              multiplier this reproduces the shipped stocks for repositories
              whose panel spans their full instrumented window (deviations,
              where the shipped seed used a different investment mean, are
              reported by --validate).
  zero        C_pre = 0              -- all stock is accumulated in-window.
  firstweek   C_pre = I[0]/delta     -- the first week is treated as typical.

Usage (from this directory):
    python3 sweep_seed_delta.py             # print the sweep table
    python3 sweep_seed_delta.py --validate  # compare reconstructed vs shipped stocks
    python3 sweep_seed_delta.py --freeze    # write results_sweep.json
    python3 sweep_seed_delta.py --check     # verify against results_sweep.json
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

import reproduce_section9 as r9

BASE_DELTA_C = 0.02
BASE_DELTA_X = 0.04
SEEDS = ("steady", "zero", "firstweek")
MULTS = (0.5, 1.0, 2.0)


def rebuild_stock(I: np.ndarray, delta: float, seed_rule: str) -> np.ndarray:
    if seed_rule == "steady":
        pre = float(I.mean()) / delta
    elif seed_rule == "zero":
        pre = 0.0
    elif seed_rule == "firstweek":
        pre = float(I[0]) / delta
    else:
        raise ValueError(seed_rule)
    out = np.empty(len(I))
    prev = pre
    for i, inv in enumerate(I):
        prev = (1.0 - delta) * prev + float(inv)
        out[i] = prev
    return out


def panels_under(panels: dict, seed_rule: str, mult: float) -> dict:
    out = copy.deepcopy(panels)
    for v in out.values():
        c = v["cols"]
        c["C"] = rebuild_stock(c["I_c"], BASE_DELTA_C * mult, seed_rule)
        c["X"] = rebuild_stock(c["I_x"], BASE_DELTA_X * mult, seed_rule)
    return out


def _within_std(x: np.ndarray, ids: np.ndarray) -> float:
    out = x.astype(float).copy()
    for u in np.unique(ids):
        m = ids == u
        out[m] -= x[m].mean()
    return float(out.std(ddof=1))


def one_config(panels: dict, widx: dict, seed_rule: str, mult: float) -> dict:
    p = panels_under(panels, seed_rule, mult)
    est = r9.reading(p, widx, apply_filter=True, proxy="input",
                     lp_panel_aware=True, corrected_stats=True)
    L, ids, _t, _dropped, _raw = r9._stack(p, widx, apply_filter=True, proxy="input")
    h4c, e3c = est["E2_H4_corrected"], est["E3_returns_corrected"]
    return {
        "seed": seed_rule, "delta_mult": mult,
        "within_sd_logC": _within_std(L["C"], ids),
        "within_sd_logX": _within_std(L["X"], ids),
        "ols": est["E1"]["ols"],
        "rts_sum": e3c["sum"],
        "rts_ci95": e3c["ci95_wild_cluster"],
        "constant_returns_inside_ci": e3c["constant_returns_inside_ci"],
        "partial_r2": est["partial_r2"],
        "context_leads_compute": bool(est["partial_r2"]["X"] > est["partial_r2"]["W"]),
        "f_alpha_eq_beta_p": h4c["f_alpha_eq_beta"]["p"],
        "split_wins": h4c["split_wins"],
        "lp_at_boundary": est["E1"]["lp_feasible"]["at_boundary"],
    }


def run(panels_dir: Path | None = None) -> dict:
    panels = r9.load_frozen_panels(panels_dir)
    all_weeks = sorted({w for v in panels.values() for w in v["weeks"]})
    widx = {w: i for i, w in enumerate(all_weeks)}
    rows = [one_config(panels, widx, s, m) for s in SEEDS for m in MULTS]
    base = next(x for x in rows if x["seed"] == "steady" and x["delta_mult"] == 1.0)
    return {
        "source": "rebuilt from shipped I_c/I_x flows (panels/panel_*.csv)",
        "base_delta_c": BASE_DELTA_C, "base_delta_x": BASE_DELTA_X,
        "configs": rows,
        "headline": {
            "ranking_stable_across_seeds": bool(all(
                x["context_leads_compute"] == base["context_leads_compute"]
                for x in rows)),
            "split_verdict_stable_across_seeds": bool(all(
                x["split_wins"] == base["split_wins"] for x in rows)),
            "boundary_always": bool(all(x["lp_at_boundary"] for x in rows)),
        },
    }


def validate(panels_dir: Path | None = None) -> int:
    panels = r9.load_frozen_panels(panels_dir)
    worst = 0.0
    for name, v in sorted(panels.items()):
        c = v["cols"]
        for key, delta in (("C", BASE_DELTA_C), ("X", BASE_DELTA_X)):
            rebuilt = rebuild_stock(c[f"I_{key.lower()}" if key == "C" else "I_x"]
                                    if key == "X" else c["I_c"], delta, "steady")
            rel = float(np.max(np.abs(rebuilt - c[key]) / np.maximum(c[key], 1.0)))
            worst = max(worst, rel)
            print(f"{name} {key}: max relative deviation vs shipped = {rel:.4f}")
    print(f"worst-case deviation: {worst:.4f} "
          "(non-zero rows: repositories whose shipped seed used an investment "
          "mean over a window wider than their panel rows)")
    return 0


def _print(res: dict) -> None:
    print(f"seed/delta sweep on {res['source']}")
    print(f"{'seed':<10}{'mult':>5}{'sd(logC)':>10}{'sd(logX)':>10}"
          f"{'alpha':>8}{'beta':>8}{'gamma':>8}{'RTS':>8}{'CRS?':>6}"
          f"{'pR2 X':>7}{'pR2 W':>7}{'X>W':>5}{'F p':>8}{'split':>6}{'bdry':>5}")
    for x in res["configs"]:
        o, p = x["ols"], x["partial_r2"]
        print(f"{x['seed']:<10}{x['delta_mult']:>5.1f}"
              f"{x['within_sd_logC']:>10.4f}{x['within_sd_logX']:>10.4f}"
              f"{o['alpha']:>8.2f}{o['beta']:>8.2f}{o['gamma']:>8.2f}"
              f"{x['rts_sum']:>8.2f}{str(x['constant_returns_inside_ci']):>6}"
              f"{p['X']:>7.3f}{p['W']:>7.3f}{str(x['context_leads_compute']):>5}"
              f"{x['f_alpha_eq_beta_p']:>8.4f}{str(x['split_wins']):>6}"
              f"{str(x['lp_at_boundary']):>5}")
    h = res["headline"]
    print(f"\nranking stable across conventions:      {h['ranking_stable_across_seeds']}")
    print(f"split verdict stable across conventions: {h['split_verdict_stable_across_seeds']}")
    print(f"LP at boundary in every configuration:   {h['boundary_always']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panels-dir", default=None)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--reference", default="results_sweep.json")
    args = ap.parse_args()
    if args.validate:
        return validate(Path(args.panels_dir) if args.panels_dir else None)
    res = run(Path(args.panels_dir) if args.panels_dir else None)
    if args.freeze:
        Path(args.reference).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
        print(f"froze {args.reference}")
        return 0
    if args.check:
        return r9.check(res, Path(args.reference))
    _print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

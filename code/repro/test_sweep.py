"""Pin the Appendix C(vi) seed/depreciation sweep to its frozen result.

The sweep is the evidence for the paper's claim that every cross-factor
conclusion of Section 9 is conditional on the perpetual-inventory seeding
convention. These tests assert (a) the frozen JSON regenerates exactly, and
(b) the qualitative claims the paper makes about the sweep are what the sweep
actually shows.
"""
from pathlib import Path

import sweep_seed_delta as sw
import reproduce_section9 as r9

HERE = Path(__file__).resolve().parent


def _res():
    return sw.run(HERE / "panels")


def test_matches_frozen_sweep_json():
    assert r9.check(_res(), HERE / "results_sweep.json", tol=1e-6) == 0


def test_ranking_and_split_verdict_are_convention_dependent():
    res = _res()
    assert res["headline"]["ranking_stable_across_seeds"] is False
    assert res["headline"]["split_verdict_stable_across_seeds"] is False


def test_lp_boundary_is_the_invariant_diagnostic():
    assert _res()["headline"]["boundary_always"] is True


def test_zero_seed_amplifies_within_dispersion():
    res = {(x["seed"], x["delta_mult"]): x for x in _res()["configs"]}
    steady, zero = res[("steady", 1.0)], res[("zero", 1.0)]
    assert zero["within_sd_logC"] / steady["within_sd_logC"] > 15   # ~23x
    assert zero["within_sd_logX"] / steady["within_sd_logX"] > 40   # ~55x
    # and the factor ranking inverts: compute overtakes context
    assert steady["context_leads_compute"] is True
    assert zero["context_leads_compute"] is False


def test_zero_seed_excludes_constant_returns_from_below():
    res = {(x["seed"], x["delta_mult"]): x for x in _res()["configs"]}
    lo, hi = res[("zero", 1.0)]["rts_ci95"]
    assert hi < 1.0

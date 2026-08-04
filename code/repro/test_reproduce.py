"""Pin Section 9's headline numbers to the shipped anonymized panels.

Runs the self-contained reproducer on ``./panels/`` and asserts that every
Section 9 statistic — both the corrected reading and the as-shipped reading it
corrects — regenerates to a tight tolerance, and that the shipped
``results_paper1_corrected.json`` is exactly what the panels produce.
"""
import math
from pathlib import Path

import reproduce_section9 as r

HERE = Path(__file__).resolve().parent
TOL = 1e-6


def _res():
    return r.run(HERE / "panels")


def test_panel_shape():
    res = _res()
    assert res["n_repos"] == 18
    assert res["n_obs"] == 123
    assert res["weeks_span"] == ["2026-W15", "2026-W30"]
    assert res["missing_observation_cells"] == 49


def test_corrected_reading():
    c = _res()["corrected"]
    assert c["n_weeks"] == 74 and c["n_dropped"] == 49
    ols = c["E1"]["ols"]
    assert math.isclose(ols["alpha"], -2.882, abs_tol=1e-3)
    assert math.isclose(ols["beta"], 7.742, abs_tol=1e-3)
    assert math.isclose(ols["gamma"], 0.224, abs_tol=1e-3)
    rts = c["E3_returns"]
    assert math.isclose(rts["sum"], 5.085, abs_tol=1e-3)
    assert math.isclose(rts["ci95"][0], 0.215, abs_tol=1e-3)
    assert math.isclose(rts["ci95"][1], 9.895, abs_tol=1e-3)
    assert rts["constant_returns_inside_ci"] is True
    h4 = c["E2_H4"]
    assert math.isclose(h4["rmse_split"], 0.7459, abs_tol=1e-3)
    assert math.isclose(h4["rmse_pooled"], 0.7725, abs_tol=1e-3)
    assert math.isclose(h4["vuong_z"], 1.404, abs_tol=1e-3)
    assert h4["split_wins"] is False
    lp = c["E1"]["lp_feasible"]
    assert lp["at_boundary"] is True
    assert math.isclose(lp["alpha"], 0.9, abs_tol=1e-9)
    assert math.isclose(lp["beta"], 0.9, abs_tol=1e-9)
    assert math.isclose(c["partial_r2"]["X"], 0.2357, abs_tol=1e-3)
    assert math.isclose(c["partial_r2"]["W"], 0.1567, abs_tol=1e-3)
    assert math.isclose(c["compute_only"]["elasticity_W"], 0.2836, abs_tol=1e-3)
    assert math.isclose(c["compute_only"]["t_W"], 4.103, abs_tol=1e-3)


def test_second_correction_reading():
    """The second-audit statistics (Appendix C(vii)): valid nested F test,
    leakage-free FE cross-validation, wild cluster bootstrap, dof-adjusted t."""
    c = _res()["corrected"]
    h4 = c["E2_H4_corrected"]
    ft = h4["f_alpha_eq_beta"]
    assert math.isclose(ft["F"], 5.020, abs_tol=1e-2)
    assert math.isclose(ft["p"], 0.0293, abs_tol=1e-3)
    assert ft["dof_den"] == 53
    assert math.isclose(h4["cv_fe_split"]["rmse"], 1.1050, abs_tol=1e-3)
    assert math.isclose(h4["cv_fe_pooled"]["rmse"], 1.1168, abs_tol=1e-3)
    assert math.isclose(h4["cv_fe_additive"]["rmse"], 1.1368, abs_tol=1e-3)
    assert h4["split_wins"] is True  # on the canonical panel; seed-conditional (C(vi))
    e3 = c["E3_returns_corrected"]
    assert math.isclose(e3["ci95_wild_cluster"][0], 0.237, abs_tol=1e-2)
    assert math.isclose(e3["ci95_wild_cluster"][1], 9.990, abs_tol=1e-2)
    assert e3["constant_returns_inside_ci"] is True
    sc = c["stocks_controlled_corrected"]
    assert math.isclose(sc["t_dof_adjusted"], 3.14, abs_tol=1e-2)
    assert sc["dof"] == 53
    co = c["compute_only_corrected"]
    assert math.isclose(co["t_dof_adjusted"], 3.59, abs_tol=1e-2)
    assert math.isclose(co["t_cluster_repo"], 3.56, abs_tol=1e-2)


def test_as_shipped_reading_is_the_pre_fix_artifact():
    s = _res()["as_shipped"]
    assert s["n_weeks"] == 123 and s["n_dropped"] == 0
    assert math.isclose(s["E1"]["ols"]["alpha"], 30.964, abs_tol=1e-3)
    assert math.isclose(s["E3_returns"]["sum"], 77.714, abs_tol=1e-3)
    assert s["E3_returns"]["constant_returns_inside_ci"] is False
    assert math.isclose(s["proxy_corr_with_logW"], 0.9924, abs_tol=1e-3)


def test_matches_shipped_json_exactly():
    assert r.check(_res(), HERE / "results_paper1_corrected.json", tol=TOL) == 0

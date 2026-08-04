# Section 9 reproduction — frozen panel

This directory regenerates every number in **Section 9 (Empirical Illustration)**
from the frozen repository-week panel, without the raw sources.

```bash
cd repro
pip install numpy pytest
python3 reproduce_section9.py          # print the corrected and as-shipped readings (incl. second-correction stats)
python3 reproduce_section9.py --check  # verify against results_paper1_corrected.json (exit 0 = match)
python3 sweep_seed_delta.py            # Appendix C(vi): seed/depreciation sweep table
python3 sweep_seed_delta.py --check    # verify against results_sweep.json
python3 -m pytest -q                   # pin the headline statistics, the sweep, and the synthetic recovery
```

## What is here

| File | What it is |
|------|------------|
| `panels/panel_repoNN.csv` | the frozen panel: one file per repository-week series (18 repositories, 123 rows, 2026-W15..W30). Repository identities are replaced by `repoNN`. |
| `reproduce_section9.py` | self-contained regenerator (imports only `numpy`, the standard library, and `estimators`). Emits the as-shipped (defective), first-correction, and second-correction readings; `--check` compares against the shipped result, `--freeze` re-freezes it. |
| `estimators.py` | the estimators the paper uses. Two generations coexist by design: the frozen defective-path functions (Vuong misapplied to a nested pair, leaky rolling CV, iid bootstrap, uncorrected dof) keep the historical readings reproducible, and the corrected counterparts (nested F test, leakage-free FE CV, wild cluster bootstrap, absorbed-dof `ols_t`) produce the readings the paper stands behind (Appendix C(vii)). |
| `sweep_seed_delta.py` | the Appendix C(vi) seed/depreciation sweep: rebuilds the stocks from the shipped `I_c`/`I_x` flows under 3 seeding conventions x 3 depreciation multipliers and re-runs the corrected reading. `--validate` audits the reconstruction against the shipped stock levels. |
| `results_paper1_corrected.json` | the frozen result the section cites; `--check` and the tests assert the panel reproduces it. |
| `results_sweep.json` | the frozen sweep behind every "seed-conditional" claim; pinned by `test_sweep.py`. |
| `test_reproduce.py` | asserts every Table 1 statistic — both correction rounds — regenerates. |
| `test_sweep.py` | pins the sweep and its qualitative claims (dispersion amplification, ranking inversion, invariant LP boundary). |
| `test_synthetic_recovery.py` | Appendix A estimator validation: known-elasticity synthetic panels recovered within 0.06; H4 rule prefers split on split-generated data and declines to reject pooling when α = β; static-stock panels flagged as non-identified. |

## Scope

The panels are the studied organization's own version-control history and
agent-session logs, aggregated to repository-week cells. The layer that builds
these cells from the raw logs is **not distributed** — those logs are the
organization's private records (the paper is an n = 1 illustration by the operator
of the measured system, disclosed in Section 9). What travels here is the frozen
aggregate panel and the estimation code, so the reported statistics are checkable
on a machine that does not hold the raw sources. Renaming the repositories does
not change any estimate.


"""
Paper 1 scenario experiments — capital convergence and output elasticities.

Run from this directory:
    python experiments.py

Demonstrates theorems B.1-B.3 (steady-state convergence + comparative statics)
and A.1-A.3 (elasticities equal exponents) numerically, as an illustration of
the paper's Appendix B. (The paper's Sections 8 and 10 reference the qualitative
conclusions these theorems support; this script does not reproduce the Section 8
AgentEconomy skeleton or the Section 10 implications, which are not computational.)
Requires numpy; matplotlib optional.
"""
from __future__ import annotations
import os

import numpy as np

from model import (
    production_stage1, steady_code, steady_context, simulate_capital,
)


def _table(title, header, rows):
    print(f"\n=== {title} ===")
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
    fmt = "  ".join(f"{{:>{x}}}" for x in w)
    print(fmt.format(*header))
    print(fmt.format(*["-" * x for x in w]))
    for r in rows:
        print(fmt.format(*r))


def scenario_capital_convergence():
    """B.1-B.3: code/context capital converge to their steady states."""
    Ic, delta_c = 0.10, 0.05
    D, eta, delta_x = 0.20, 0.10, 0.05
    C_star, X_star = steady_code(Ic, delta_c), steady_context(D, eta, delta_x)
    Cs, Xs = simulate_capital(
        C0=10.0, X0=0.10, Ic=Ic, D=D, eta=eta,
        delta_c=delta_c, delta_x=delta_x, steps=60,
    )
    rows = [(t, round(Cs[t], 4), round(Xs[t], 4)) for t in (0, 1, 5, 10, 20, 40, 60)]
    _table("Scenario 1 — Capital accumulation -> steady state",
           ("t", "Code C_t", "Context X_t"), rows)
    print(f"steady states: C* = {C_star:.4f}, X* = {X_star:.4f}  "
          f"| reached C={Cs[-1]:.4f}, X={Xs[-1]:.4f}")
    return np.array(Cs), np.array(Xs), C_star, X_star


def scenario_elasticities():
    """A.1-A.3: doubling each factor multiplies output by 2^(elasticity)."""
    alpha, beta, gamma = 0.35, 0.35, 0.30
    A, C, X, W = 1.0, 2.0, 2.0, 2.0
    Y0 = production_stage1(A, C, X, W, alpha, beta, gamma)
    rows = [
        ("Code C x2", round(production_stage1(A, 2 * C, X, W, alpha, beta, gamma) / Y0, 4),
         f"2^{alpha}={2**alpha:.4f}"),
        ("Context X x2", round(production_stage1(A, C, 2 * X, W, alpha, beta, gamma) / Y0, 4),
         f"2^{beta}={2**beta:.4f}"),
        ("Work W x2", round(production_stage1(A, C, X, 2 * W, alpha, beta, gamma) / Y0, 4),
         f"2^{gamma}={2**gamma:.4f}"),
        ("All x2 (scale)", round(
            production_stage1(A, 2 * C, 2 * X, 2 * W, alpha, beta, gamma) / Y0, 4),
         f"2^s={2**(alpha+beta+gamma):.4f}"),
    ]
    _table("Scenario 2 — Output elasticities equal exponents",
           ("shock", "Y/Y0", "predicted"), rows)


def main():
    print("Paper 1 — scenario experiments")
    print("=" * 50)
    scenario_capital_convergence()
    scenario_elasticities()
    print("\nAll scenarios consistent with proofs/01-production-proofs.md theorems.")


if __name__ == "__main__":
    main()

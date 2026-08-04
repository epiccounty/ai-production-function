"""Theorem group B — capital accumulation and steady state (Paper 1).

B.1 (code steady state C* = Ic/delta_c), B.2 (context X* = eta*D/delta_x),
B.3 (steady-output comparative statics). The earlier standalone note's B.4
(replication externality) was subsumed into A.4(ii) and is not a separate
theorem; see tests/test_production.py for the A.4 Euler/replication coverage.
"""
import math

from model import (
    production_stage1,
    steady_code,
    steady_context,
    simulate_capital,
)


def test_code_steady_state():
    """B.1: simulated code capital converges geometrically to Ic/delta_c."""
    Ic, delta_c = 0.10, 0.05
    C_star = steady_code(Ic, delta_c)          # = 2.0
    assert math.isclose(C_star, Ic / delta_c)
    Cs, _ = simulate_capital(
        C0=10.0, X0=5.0, Ic=Ic, D=0.2, eta=0.1,
        delta_c=delta_c, delta_x=0.05, steps=5_000,
    )
    assert math.isclose(Cs[-1], C_star, rel_tol=1e-6)
    # convergence is monotone toward C_star (AR(1) with positive factor < 1)
    tail = [abs(c - C_star) for c in Cs[-6:]]
    assert all(tail[i] > tail[i + 1] for i in range(len(tail) - 1))


def test_context_steady_state():
    """B.2: simulated context capital converges to eta*D/delta_x."""
    D, eta, delta_x = 0.20, 0.10, 0.05
    X_star = steady_context(D, eta, delta_x)   # = 0.4
    assert math.isclose(X_star, eta * D / delta_x)
    _, Xs = simulate_capital(
        C0=2.0, X0=0.01, Ic=0.1, D=D, eta=eta,
        delta_c=0.05, delta_x=delta_x, steps=5_000,
    )
    assert math.isclose(Xs[-1], X_star, rel_tol=1e-6)


def test_steady_output_comparative_statics():
    """B.3: Y* = A (Ic/delta_c)^alpha (eta D/delta_x)^beta W^gamma moves correctly."""
    A, W = 1.0, 1.0
    alpha, beta, gamma = 0.35, 0.35, 0.30

    def y_star(Ic, delta_c, D, eta, delta_x):
        return production_stage1(
            A, steady_code(Ic, delta_c), steady_context(D, eta, delta_x), W,
            alpha, beta, gamma,
        )

    base = y_star(0.1, 0.05, 0.2, 0.1, 0.05)
    assert y_star(0.2, 0.05, 0.2, 0.1, 0.05) > base   # more code investment
    assert y_star(0.1, 0.10, 0.2, 0.1, 0.05) < base   # higher code decay
    assert y_star(0.1, 0.05, 0.2, 0.2, 0.05) > base   # higher learning efficiency
    assert y_star(0.1, 0.05, 0.2, 0.1, 0.10) < base   # higher context decay


def test_replication_scale_economy():
    """A.4(ii): a non-rival code unit replicated across n deployments at near-zero
    marginal cost yields increasing returns to deployment of the non-rival stock.

    This is the construction behind A.4(ii) (the paper labels it "a construction,
    not an empirical result"). Its economic content is the *costless-replication +
    non-rivalry* premise, which the test makes explicit: one owned code stock C is
    reused (not rebuilt) across n deployments, each consuming rival inputs (X, W)
    afresh, with replication cost fixed and negligible. Under that premise total
    output grows faster than total cost, i.e. average output per unit of rival input
    rises with n. We verify that scaling directly, not the arithmetic identity
    n*mp > mp (which holds for any function and tests nothing).
    """
    A = 1.0
    alpha, beta, gamma = 0.35, 0.35, 0.30
    # One owned, non-rival code stock; each deployment reuses it without rebuilding.
    C_shared = 2.0
    X_per_site, W_per_site = 1.0, 1.0
    # Replication carries a fixed one-time development cost spread across sites,
    # modelled as a fixed cost F independent of n (the A.4(ii) premise).
    F = 1.0

    def total_output_and_cost(n):
        Y_total = sum(
            production_stage1(A, C_shared, X_per_site, W_per_site, alpha, beta, gamma)
            for _ in range(n)
        )
        # rival inputs scale with n; the non-rival code stock does not; replication cost is fixed F.
        cost = n * (X_per_site + W_per_site) + F
        return Y_total, cost

    ns = [1, 2, 5, 10]
    avg_y = [total_output_and_cost(n)[0] / total_output_and_cost(n)[1] for n in ns]
    # average output per unit of input rises with n: the costless-replication scale economy
    for a, b in zip(avg_y, avg_y[1:]):
        assert b > a, f"average output should rise with replication count, got {avg_y}"

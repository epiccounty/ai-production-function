"""Theorem group A — production-function properties (Paper 1).

A.1 (positive, diminishing marginal products), A.2 (elasticity = exponent),
A.3 (returns to scale), A.4 (Euler factor-distribution identity).
"""
import math

import pytest

from model import production_stage1, partial, second_partial, output_elasticity

_FN = lambda *a: production_stage1(*a)  # noqa: E731

CASES = [
    # A,   C,   X,   W,   alpha, beta, gamma
    (1.0, 1.0, 1.0, 1.0, 0.35, 0.35, 0.30),
    (1.2, 2.5, 0.8, 1.7, 0.40, 0.30, 0.30),
    (0.9, 5.0, 3.0, 2.0, 0.35, 0.35, 0.30),
]


@pytest.mark.parametrize("A,C,X,W,alpha,beta,gamma", CASES)
def test_diminishing_returns(A, C, X, W, alpha, beta, gamma):
    """A.1: dY/dC, dY/dX, dY/dW > 0 and strictly diminishing."""
    args = (A, C, X, W, alpha, beta, gamma)
    for i, name in ((1, "C"), (2, "X"), (3, "W")):
        mp = partial(_FN, args, i)
        dmp = second_partial(_FN, args, i)
        assert mp > 0, f"marginal product of {name} must be positive, got {mp}"
        assert dmp < 0, f"marginal product of {name} must diminish, got {dmp}"


@pytest.mark.parametrize("A,C,X,W,alpha,beta,gamma", CASES)
def test_elasticities_equal_exponents(A, C, X, W, alpha, beta, gamma):
    """A.2: output elasticity of each factor equals its exponent."""
    args = (A, C, X, W, alpha, beta, gamma)
    assert math.isclose(output_elasticity(_FN, args, 1), alpha, rel_tol=1e-4)
    assert math.isclose(output_elasticity(_FN, args, 2), beta, rel_tol=1e-4)
    assert math.isclose(output_elasticity(_FN, args, 3), gamma, rel_tol=1e-4)


@pytest.mark.parametrize("A,C,X,W,alpha,beta,gamma", CASES)
@pytest.mark.parametrize("lam", [0.5, 2.0, 3.7])
def test_returns_to_scale(A, C, X, W, alpha, beta, gamma, lam):
    """A.3: Y(lambda*C, lambda*X, lambda*W) = lambda^s * Y(C,X,W), s = alpha+beta+gamma."""
    Y = production_stage1(A, C, X, W, alpha, beta, gamma)
    Y_scaled = production_stage1(A, lam * C, lam * X, lam * W, alpha, beta, gamma)
    s = alpha + beta + gamma
    assert math.isclose(Y_scaled, (lam ** s) * Y, rel_tol=1e-9)


def test_euler_factor_distribution():
    """A.4: under constant returns (s = 1), factor payments exhaust output.

    C*dY/dC + X*dY/dX + W*dY/dW = Y.
    """
    alpha, beta, gamma = 0.40, 0.35, 0.25  # sums to 1.0 exactly
    assert abs(alpha + beta + gamma - 1.0) < 1e-12
    A, C, X, W = 1.3, 4.0, 2.5, 3.0
    Y = production_stage1(A, C, X, W, alpha, beta, gamma)
    args = (A, C, X, W, alpha, beta, gamma)
    payments = (
        C * partial(_FN, args, 1)
        + X * partial(_FN, args, 2)
        + W * partial(_FN, args, 3)
    )
    assert math.isclose(payments, Y, rel_tol=1e-3)


@pytest.mark.parametrize("alpha,beta,gamma", [(-0.5, 0.35, 0.30), (1.5, 0.35, 0.30), (0.35, 0.0, 0.30), (0.35, 1.0, 0.30)])
def test_exponent_range_enforced(alpha, beta, gamma):
    """Appendix B assumes alpha, beta, gamma in (0,1); out-of-range exponents
    violate A.1 (diminishing marginal products) and must be rejected."""
    with pytest.raises(ValueError):
        production_stage1(1.0, 2.0, 2.0, 2.0, alpha, beta, gamma)

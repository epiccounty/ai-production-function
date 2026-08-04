"""
Paper 1 model — Stage-1 production function and capital accumulation.

Self-contained: no imports from sibling packages. Implements the constructs of
the paper's Appendix B (Theorems A.1-A.3, A.4(i), B.1-B.3, and Proposition
A.4(ii)) and is verified by tests/test_production.py and
tests/test_accumulation.py. A.4(ii) — the replication scale economy — is an
economic proposition (near-costless replication + a pricing structure), not an
algebraic identity, and is therefore exercised by a construction test that
makes its premises explicit rather than by a theorem check.

Notation:
    Production   : Y = A * C^alpha * X^beta * W^gamma            (P1)
    Code capital : C_{t+1} = (1 - delta_c) * C_t + Ic            (B.1)
    Context cap. : X_{t+1} = (1 - delta_x) * X_t + eta * D       (B.2)
"""
from __future__ import annotations
from typing import Callable, Sequence


# --------------------------------------------------------------------------- #
# Production function (P1) and numerical differentiation
# --------------------------------------------------------------------------- #
def production_stage1(
    A: float, C: float, X: float, W: float,
    alpha: float = 0.35, beta: float = 0.35, gamma: float = 0.30,
) -> float:
    """Stage-1 generalized production function  Y = A * C^alpha * X^beta * W^gamma."""
    if min(A, C, X, W) <= 0:
        raise ValueError("factors must be positive")
    if not (0 < alpha < 1 and 0 < beta < 1 and 0 < gamma < 1):
        raise ValueError("exponents must lie in (0,1) for positive, diminishing marginal products (A.1)")
    return A * C**alpha * X**beta * W**gamma


def partial(fn: Callable, args: Sequence[float], i: int, h: float = 1e-6) -> float:
    """Central-difference partial derivative of fn(*args) with respect to args[i]."""
    lo, hi = list(args), list(args)
    lo[i] -= h
    hi[i] += h
    return (fn(*hi) - fn(*lo)) / (2.0 * h)


def second_partial(fn: Callable, args: Sequence[float], i: int, h: float = 1e-3) -> float:
    """Second central-difference partial derivative of fn(*args) wrt args[i]."""
    lo, hi = list(args), list(args)
    lo[i] -= h
    hi[i] += h
    return (fn(*hi) - 2.0 * fn(*args) + fn(*lo)) / (h * h)


def output_elasticity(fn: Callable, args: Sequence[float], i: int, h: float = 1e-6) -> float:
    """Numerical output elasticity  d ln Y / d ln (args[i])  =  (x/Y) * dY/dx."""
    x = args[i]
    return (x / fn(*args)) * partial(fn, args, i, h)


# --------------------------------------------------------------------------- #
# Capital accumulation (B.1, B.2)
# --------------------------------------------------------------------------- #
def step_code(C: float, Ic: float, delta_c: float) -> float:
    """Code capital one-step transition  C_{t+1} = (1 - delta_c) * C_t + Ic."""
    return (1.0 - delta_c) * C + Ic


def step_context(X: float, D: float, eta: float, delta_x: float) -> float:
    """Context capital one-step transition  X_{t+1} = (1 - delta_x) * X_t + eta * D."""
    return (1.0 - delta_x) * X + eta * D


def steady_code(Ic: float, delta_c: float) -> float:
    """Steady-state code capital  C* = Ic / delta_c  (Theorem B.1)."""
    if delta_c <= 0:
        raise ValueError("delta_c must be positive for a finite steady state")
    return Ic / delta_c


def steady_context(D: float, eta: float, delta_x: float) -> float:
    """Steady-state context capital  X* = eta * D / delta_x  (Theorem B.2)."""
    if delta_x <= 0:
        raise ValueError("delta_x must be positive for a finite steady state")
    return eta * D / delta_x


def simulate_capital(
    C0: float, X0: float, *, Ic: float, D: float, eta: float,
    delta_c: float, delta_x: float, steps: int = 10_000, tol: float = 1e-12,
) -> tuple[list[float], list[float]]:
    """Iterate the code/context accumulation laws until convergence."""
    Cs, Xs = [C0], [X0]
    for _ in range(steps):
        Cs.append(step_code(Cs[-1], Ic, delta_c))
        Xs.append(step_context(Xs[-1], D, eta, delta_x))
        if abs(Cs[-1] - Cs[-2]) < tol and abs(Xs[-1] - Xs[-2]) < tol:
            break
    return Cs, Xs

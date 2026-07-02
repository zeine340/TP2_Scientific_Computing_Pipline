import numpy as np
from sympy import tanh

from src.symbolic_derivations import (
    define_symbols,
    analytical_solution,
    compute_derivatives,
    residual_source,
    export_numpy_functions,
)


def test_define_symbols():
    x, t, c, nu = define_symbols()

    assert str(x) == "x"
    assert str(t) == "t"
    assert str(c) == "c"
    assert str(nu) == "nu"


def test_analytical_solution():
    u, *_ = analytical_solution()

    assert u.has(tanh)


def test_compute_derivatives():

    u, du_dt, du_dx, d2u_dx2, *_ = compute_derivatives()

    assert du_dt is not None
    assert du_dx is not None
    assert d2u_dx2 is not None


def test_residual():

    f, *_ = residual_source()

    assert f is not None


def test_lambdify():

    funcs = export_numpy_functions()

    value = funcs["u"](1.0, 0.5, 1.0)

    assert np.isfinite(value)
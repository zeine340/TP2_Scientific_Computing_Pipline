import numpy as np

from src.numerical_core import (
    create_grid,
    compare_memory_layout,
    residual,
)


def test_create_grid():

    X, T = create_grid(20, 10)

    assert X.shape == (20, 10)
    assert T.shape == (20, 10)


def test_memory_layout():

    X, _ = create_grid()

    c_array, f_array = compare_memory_layout(X)

    assert c_array.flags["C_CONTIGUOUS"]
    assert f_array.flags["F_CONTIGUOUS"]


def test_residual_shape():

    X, T = create_grid(30, 40)

    r = residual(X, T, c=1.0, nu=0.1)

    assert r.shape == X.shape


def test_residual_values():

    X, T = create_grid(10, 10)

    r = residual(X, T, 1.0, 0.1)

    assert np.all(np.isfinite(r))
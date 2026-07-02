import numpy as np

from src.analyse_stabilite import (
    hilbert_matrix,
    condition_number,
    solve_system,
    validate_solution,
)


def test_hilbert():

    A = hilbert_matrix(5)

    assert A.shape == (5, 5)


def test_condition_number():

    A = hilbert_matrix(5)

    cond = condition_number(A)

    assert cond > 1


def test_solver():

    A = hilbert_matrix(5)

    x = np.ones(5)

    b = A @ x

    x_hat = solve_system(A, b)

    assert np.allclose(x, x_hat)


def test_validation():

    A = hilbert_matrix(5)

    x = np.ones(5)

    b = A @ x

    assert validate_solution(A, x, b)
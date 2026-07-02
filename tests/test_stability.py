import numpy as np

from src.analyse_stabilite import (
    hilbert_matrix,
    condition_number,
    solve_system,
    reconstruction_error,
    compare_precisions,
    perturbation_analysis,
    validate_solution,
)


def test_hilbert_shape():
    H = hilbert_matrix(6)

    assert H.shape == (6, 6)


def test_hilbert_first_values():
    H = hilbert_matrix(3)

    assert np.isclose(H[0, 0], 1.0)
    assert np.isclose(H[0, 1], 0.5)
    assert np.isclose(H[1, 0], 0.5)


def test_condition_number():
    H = hilbert_matrix(8)

    cond = condition_number(H)

    assert cond > 1


def test_solve_system():
    A = hilbert_matrix(5)

    x = np.ones(5)

    b = A @ x

    solution = solve_system(A, b)

    assert np.allclose(solution, x)


def test_reconstruction_error():
    A = hilbert_matrix(5)

    x = np.ones(5)

    b = A @ x

    err = reconstruction_error(A, x, b)

    assert err < 1e-12


def test_compare_precisions():
    results = compare_precisions([5])

    assert len(results) == 1

    row = results[0]

    assert "n" in row
    assert "condition_number" in row
    assert "float16" in row
    assert "float32" in row
    assert "float64" in row


def test_perturbation_analysis():
    result = perturbation_analysis(8)

    assert "condition_number" in result
    assert "solution_error" in result

    assert result["condition_number"] > 1
    assert result["solution_error"] >= 0


def test_validate_solution():
    A = hilbert_matrix(5)

    x = np.ones(5)

    b = A @ x

    x_hat = solve_system(A, b)

    assert validate_solution(A, x_hat, b)


def test_validate_solution_false():
    A = hilbert_matrix(5)

    x = np.ones(5)

    b = A @ x

    wrong = np.zeros(5)

    assert not validate_solution(A, wrong, b)
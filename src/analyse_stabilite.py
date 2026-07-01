from __future__ import annotations

import numpy as np


def hilbert_matrix(n: int, dtype=np.float64) -> np.ndarray:
    """
    Construct an n×n Hilbert matrix.
    """
    i = np.arange(n)
    j = np.arange(n)
    return (1.0 / (i[:, None] + j + 1)).astype(dtype)


def condition_number(A: np.ndarray) -> float:
    """
    Compute the condition number κ(A).
    """
    return float(np.linalg.cond(A))


def solve_system(
    A: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Solve A x = b.
    """
    return np.linalg.solve(A, b)


def reconstruction_error(
    A: np.ndarray,
    alpha: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Compute ||Aα-b||₂.
    """
    return float(np.linalg.norm(A @ alpha - b))


def compare_precisions(
    n_values=range(5, 26),
):
    """
    Compare float16, float32 and float64.
    """

    results = []

    for n in n_values:

        errors = {}

        for dtype in (np.float16, np.float32, np.float64):

            A = hilbert_matrix(n, dtype=np.float64)

            x_true = np.ones(n)

            b = A @ x_true

            if dtype == np.float16:
                # NumPy does not reliably solve linear systems in float16.
                # We cast to float32 for the solve while keeping the inputs
                # quantized to float16.
                A_num = A.astype(np.float16).astype(np.float32)
                b_num = b.astype(np.float16).astype(np.float32)
            else:
                A_num = A.astype(dtype)
                b_num = b.astype(dtype)

            x = np.linalg.solve(A_num, b_num)

            err = np.linalg.norm(
                x.astype(np.float64) - x_true
            )

            errors[np.dtype(dtype).name] = err

        cond = np.linalg.cond(A)

        results.append(
            {
                "n": n,
                "condition_number": cond,
                **errors,
            }
        )

    return results


def perturbation_analysis(
    n: int = 15,
    epsilon: float = 1e-7,
):
    """
    Study perturbation propagation.
    """

    A = hilbert_matrix(n)

    x_true = np.ones(n)

    b = A @ x_true

    perturbation = epsilon * np.random.randn(n)

    b_perturbed = b + perturbation

    x = np.linalg.solve(A, b)

    x_perturbed = np.linalg.solve(A, b_perturbed)

    amplification = np.linalg.norm(x_perturbed - x)

    return {
        "condition_number": np.linalg.cond(A),
        "solution_error": amplification,
    }


def validate_solution(
    A: np.ndarray,
    alpha: np.ndarray,
    b: np.ndarray,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> bool:
    """
    Robust numerical validation using np.isclose.
    """

    residual = A @ alpha - b

    return bool(
        np.all(
            np.isclose(
                residual,
                0.0,
                atol=atol,
                rtol=rtol,
            )
        )
    )


if __name__ == "__main__":

    A = hilbert_matrix(10)

    print("Condition number:", condition_number(A))

    results = compare_precisions()

    print(results[:3])

    analysis = perturbation_analysis()

    print(analysis)

    x = np.ones(10)

    b = A @ x

    x_hat = solve_system(A, b)

    print(validate_solution(A, x_hat, b))
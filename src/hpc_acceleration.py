from __future__ import annotations

import timeit
import cProfile
import pstats

import numpy as np
from numba import njit, prange
from joblib import Parallel, delayed


# ==========================================================
# Exercise 6.1
# Initial filtering operator (non-optimized)
# ==========================================================

def local_filter(grid: np.ndarray) -> np.ndarray:
    """
    Simple 3x3 mean filter implemented with nested loops.
    """

    n, m = grid.shape
    output = np.zeros_like(grid)

    for i in range(1, n - 1):
        for j in range(1, m - 1):

            s = 0.0

            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    s += grid[i + di, j + dj]

            output[i, j] = s / 9.0

    return output


# ==========================================================
# Profiling
# ==========================================================

def profile_filter() -> None:

    grid = np.random.rand(1000, 1000)

    profiler = cProfile.Profile()

    profiler.enable()
    local_filter(grid)
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumtime")
    stats.print_stats(10)

    execution_time = timeit.timeit(
        lambda: local_filter(grid),
        number=3,
    )

    print(f"Average execution time: {execution_time / 3:.3f} s")


# ==========================================================
# Exercise 6.2
# Numba acceleration
# ==========================================================

@njit(parallel=True, fastmath=True)
def local_filter_numba(grid):

    n, m = grid.shape

    output = np.zeros_like(grid)

    for i in prange(1, n - 1):

        for j in range(1, m - 1):

            s = 0.0

            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    s += grid[i + di, j + dj]

            output[i, j] = s / 9.0

    return output


# ==========================================================
# Parameter study
# ==========================================================

def simulate_parameter(c: float, nu: float) -> float:

    grid = np.random.rand(800, 800)

    result = local_filter_numba(grid)

    return result.mean() + c + nu


def parameter_sweep():

    parameters = [
        (c, nu)
        for c in np.linspace(0.5, 2.0, 10)
        for nu in np.linspace(0.01, 0.10, 10)
    ]

    results = Parallel(
        n_jobs=-1
    )(
        delayed(simulate_parameter)(c, nu)
        for c, nu in parameters
    )

    return results


if __name__ == "__main__":

    profile_filter()

    grid = np.random.rand(1000, 1000)

    local_filter_numba(grid)   # JIT compilation

    t = timeit.timeit(
        lambda: local_filter_numba(grid),
        number=5,
    )

    print(f"Numba execution time: {t / 5:.3f} s")

    parameter_sweep()
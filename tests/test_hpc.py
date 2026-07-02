import numpy as np

from src.hpc_acceleration import (
    local_filter,
    local_filter_numba,
    simulate_parameter,
    parameter_sweep,
)


def test_local_filter_shape():
    grid = np.random.rand(20, 20)

    result = local_filter(grid)

    assert result.shape == grid.shape


def test_local_filter_preserves_boundary():
    grid = np.ones((10, 10))

    result = local_filter(grid)

    # Borders remain zero because they are never updated
    assert np.all(result[0, :] == 0)
    assert np.all(result[-1, :] == 0)
    assert np.all(result[:, 0] == 0)
    assert np.all(result[:, -1] == 0)


def test_local_filter_center():
    grid = np.ones((5, 5))

    result = local_filter(grid)

    # Mean of a 3x3 block of ones is one
    assert np.isclose(result[2, 2], 1.0)


def test_numba_matches_python():
    np.random.seed(0)

    grid = np.random.rand(30, 30)

    python_result = local_filter(grid)
    numba_result = local_filter_numba(grid)

    assert np.allclose(python_result, numba_result)


def test_simulate_parameter_returns_float():
    result = simulate_parameter(1.0, 0.05)

    assert isinstance(result, float)


def test_parameter_sweep():
    results = parameter_sweep()

    # 10 c values × 10 nu values
    assert len(results) == 100

    assert all(isinstance(x, float) for x in results)
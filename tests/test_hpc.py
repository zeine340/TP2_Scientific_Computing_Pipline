import numpy as np

from src.hpc_acceleration import (
    local_filter,
    local_filter_numba,
)


def test_filter_shape():

    grid = np.random.rand(20, 20)

    result = local_filter(grid)

    assert result.shape == grid.shape


def test_numba_filter():

    grid = np.random.rand(20, 20)

    local_filter_numba(grid)

    result = local_filter_numba(grid)

    assert result.shape == grid.shape


def test_filters_close():

    grid = np.random.rand(20, 20)

    a = local_filter(grid)

    b = local_filter_numba(grid)

    assert np.allclose(a, b)
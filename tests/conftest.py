import numpy as np
import pytest

@pytest.fixture
def grid():

    x = np.linspace(-1, 1, 30)

    t = np.linspace(0, 1, 20)

    return np.meshgrid(x, t, indexing="ij")

@pytest.fixture
def random_matrix():

    np.random.seed(42)

    return np.random.rand(20, 20)
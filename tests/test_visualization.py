import numpy as np
from pathlib import Path

from src.visualization import (
    generate_static_figures,
    generate_interactive_surface,
)


def test_static_plot(tmp_path):

    X, T = np.meshgrid(
        np.linspace(0, 1, 20),
        np.linspace(0, 1, 20),
    )

    prediction = np.sin(X)

    reference = np.sin(X)

    output = tmp_path / "figure.pdf"

    generate_static_figures(
        X,
        T,
        prediction,
        reference,
        output,
    )

    assert output.exists()


def test_plotly(tmp_path):

    X, T = np.meshgrid(
        np.linspace(0, 1, 20),
        np.linspace(0, 1, 20),
    )

    prediction = np.sin(X)

    output = tmp_path / "plot.html"

    generate_interactive_surface(
        X,
        T,
        prediction,
        output,
    )

    assert output.exists()
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from pathlib import Path

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})


def generate_static_figures(
    X: np.ndarray,
    T: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
    output_pdf: str | Path ="outputs/figures/results.pdf",
) -> None:
    
    """
    Generate publication-quality figures.
    """

    error = np.abs(prediction - reference)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(
        prediction,
        cmap="viridis",
        ax=axes[0],
    )

    axes[0].set_title(r"PINN approximation $\hat{u}(x,t)$")
    axes[0].set_xlabel(r"$t$")
    axes[0].set_ylabel(r"$x$")

    mean_error = error.mean(axis=0)

    axes[1].plot(T[0], mean_error)

    axes[1].set_xlabel(r"$t$")
    axes[1].set_ylabel(r"$|u-\hat{u}|$")
    axes[1].set_title("Prediction error")

    fig.tight_layout()

    plt.savefig(
        output_pdf,
        format="pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def generate_interactive_surface(
    X: np.ndarray,
    T: np.ndarray,
    prediction: np.ndarray,
    # reference: np.ndarray,
    output_html: str | Path ="outputs/figures/pinn_surface.html",
) -> None:
    """
    Interactive 3D Plotly visualization.
    """

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=T,
                z=prediction,
                colorscale="Viridis",
            )
        ]
    )

    fig.update_layout(
        title="PINN Solution",
        scene=dict(
            xaxis_title="x",
            yaxis_title="t",
            zaxis_title="u(x,t)",
        ),
    )

    fig.write_html(
        output_html,
        include_plotlyjs="cdn",
    )
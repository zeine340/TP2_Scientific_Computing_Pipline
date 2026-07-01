# src/numerical_core.py

from __future__ import annotations

import numpy as np
import polars as pl


def create_grid(
    n: int = 200,
    m: int = 100,
    x_min: float = -5.0,
    x_max: float = 5.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a 2D spatial-temporal grid.

    Returns
    -------
    X, T : ndarray
    """
    x = np.linspace(x_min, x_max, n)
    t = np.linspace(t_min, t_max, m)

    X, T = np.meshgrid(x, t, indexing="ij")
    return X, T


def inspect_array(arr: np.ndarray) -> None:
    """
    Display low-level array information.
    """
    print("Shape   :", arr.shape)
    print("Dtype   :", arr.dtype)
    print("Strides :", arr.strides)
    print("Order C :", arr.flags["C_CONTIGUOUS"])
    print("Order F :", arr.flags["F_CONTIGUOUS"])
    print()


def compare_memory_layout(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compare C- and Fortran-contiguous layouts.
    """
    X_c = np.ascontiguousarray(X)
    X_f = np.asfortranarray(X)

    print("===== C layout =====")
    inspect_array(X_c)

    print("===== Fortran layout =====")
    inspect_array(X_f)

    return X_c, X_f


def demonstrate_view_vs_copy(X: np.ndarray) -> None:
    """
    Demonstrate the difference between views and copies.
    """
    view = X[:, :10]
    copy = X[:, :10].copy()

    print("View shares memory :", view.base is not None)
    print("Copy shares memory :", copy.base is not None)

    print("View base :", view.base is X)
    print("Copy base :", copy.base)


def load_sensor_coordinates(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read Parquet or CSV lazily using Polars and return valid coordinates.
    """
    if path.endswith(".parquet"):
        scan = pl.scan_parquet(path)
    elif path.endswith(".csv"):
        scan = pl.scan_csv(path)
    else:
        raise ValueError("Unsupported file format.")

    df = (
        scan
        .filter(
            pl.col("latitude").is_not_null()
            & pl.col("longitude").is_not_null()
        )
        .select(["latitude", "longitude"])
        .collect()
    )

    latitude = df["latitude"].to_numpy()
    longitude = df["longitude"].to_numpy()

    return latitude, longitude


def residual(
    X: np.ndarray,
    T: np.ndarray,
    c: float,
    nu: float,
) -> np.ndarray:
    """
    Residual obtained in Module 3.

    f(x,t)=2*nu*tanh(x-c*t)*sech²(x-c*t)
    """

    z = X - c * T

    tanh_z = np.tanh(z)
    sech2 = 1.0 - tanh_z**2

    return 2.0 * nu * tanh_z * sech2


def apply_residual_vectorized(
    X: np.ndarray,
    T: np.ndarray,
    c: float = 1.0,
    nu: float = 0.1,
) -> np.ndarray:
    """
    Fully vectorized residual evaluation.
    """
    return residual(X, T, c, nu)


if __name__ == "__main__":

    X, T = create_grid(300, 200)

    inspect_array(X)

    compare_memory_layout(X)

    demonstrate_view_vs_copy(X)

    F = apply_residual_vectorized(X, T)

    print("Residual shape :", F.shape)
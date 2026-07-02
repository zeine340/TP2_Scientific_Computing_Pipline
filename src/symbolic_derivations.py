from typing import Any
from collections.abc import Callable
from sympy import symbols, Symbol, Expr, tanh, diff, lambdify


def define_symbols() -> tuple[Symbol, Symbol, Symbol, Symbol]:
    """
    Define symbolic variables and parameters.

    Returns
    -------
    tuple
        (x, t, c, nu)
    """
    x, t = symbols("x t", real=True)
    c, nu = symbols("c nu", real=True)
    return x, t, c, nu

def analytical_solution() -> tuple[
    Expr,
    Symbol,
    Symbol,
    Symbol,
    Symbol
]:

    """
    Define the analytical solution u(x,t)=tanh(x-c*t).

    Returns
    -------
    tuple
        (u, x, t, c, nu)
    """
    x, t, c, nu = define_symbols()
    u = tanh(x - c * t)
    return u, x, t, c, nu

def compute_derivatives() -> tuple[
    Expr,
    Expr,
    Expr,
    Expr,
    Symbol,
    Symbol,
    Symbol,
    Symbol,
]:
    
    """
    Compute symbolic derivatives.

    Returns
    -------
    tuple
        (u, du_dt, du_dx, d2u_dx2, x, t, c, nu)
    """
    u, x, t, c, nu = analytical_solution()

    du_dt = diff(u, t)
    du_dx = diff(u, x)
    d2u_dx2 = diff(u, x, 2)

    return u, du_dt, du_dx, d2u_dx2, x, t, c, nu

def residual_source() -> tuple[
    Expr,
    Expr,
    Expr,
    Expr,
    Expr
]:
    """
    Compute the residual/source term

        f(x,t)=∂u/∂t + c∂u/∂x − ν∂²u/∂x²

    Returns
    -------
    tuple
        (f, x, t, c, nu)
    """
    u, du_dt, du_dx, d2u_dx2, x, t, c, nu = compute_derivatives()
    

    f = du_dt + c * du_dx - nu * d2u_dx2

    return f, x, t, c, nu

def export_numpy_functions() -> dict[str, Callable[..., Any]]:
    """
    Export NumPy functions using lambdify.

    Returns
    -------
    dict
        Dictionary containing NumPy-callable functions.
    """
    u, du_dt, du_dx, d2u_dx2, x, t, c, nu = compute_derivatives()
    f, _, _, _, _ = residual_source()

    return {
        "u": lambdify((x, t, c), u, "numpy"),
        "du_dt": lambdify((x, t, c), du_dt, "numpy"),
        "du_dx": lambdify((x, t, c), du_dx, "numpy"),
        "d2u_dx2": lambdify((x, t, c), d2u_dx2, "numpy"),
        "f": lambdify((x, t, c, nu), f, "numpy"),
    }


if __name__ == "__main__":
    u, du_dt, du_dx, d2u_dx2, *_ = compute_derivatives()
    f, *_ = residual_source()

    print("u(x,t) =", u)
    print("∂u/∂t =", du_dt)
    print("∂u/∂x =", du_dx)
    print("∂²u/∂x² =", d2u_dx2)
    print("Residual f(x,t) =", f)
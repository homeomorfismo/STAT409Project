"""PML construction for organ pipes.

Provides:
- Radial PML (cf. [1,2,4])
- Cartesian PML (cf. [3])

[1] Gopalakrishnan, Jay, Jacob Grosek, Gabriel Pinochet-Soto,
    and Pieter VandenBerge.
    "Adaptive resolution of fine scales in modes of microstructured optical
    fibers." SIAM Journal on Scientific Computing 47, no. 1 (2025): B108-B130.

[2] Vandenberge, Pieter, Jay Gopalakrishnan, and Jacob Grosek.
    "Sensitivity of confinement losses in optical fibers to modeling approach." 
    Optics Express 31, no. 16 (2023): 26735-26756.

[3] Vaziri Astaneh, Ali, Brendan Keith, and Leszek Demkowicz.
    "On perfectly matched layers for discontinuous Petrov–Galerkin methods."
    Computational Mechanics 63 (2019): 1131-1145.

[4] Kim, Seungil, and Joseph Pasciak. "The computation of resonances in open
    systems using a perfectly matched layer." Mathematics of Computation 78,
    no. 267 (2009): 1375-1398.
"""

from __future__ import annotations
from typing import Tuple
import ngsolve as ng
import sympy as sp


def radial_pml(
    alpha: float,
    pml_begin: float,
    pml_end: float,
) -> Tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """Symbolic radial PML function. Cf. [4].

    Input
    -----
    alpha : float
        PML strength.
    pml_begin : float
        (radial) start of PML region.
    pml_end : float
        (radial) end of PML region.

    Output
    ------
    radial_pml : sympy.Expr
        Radial PML function.
    complete_pml : sympy.Expr
        Complete PML function (1 + 1j * radial_pml).
    radial_pml_dt : sympy.Expr
        Derivative of radial PML function.
    complete_pml_dt : sympy.Expr
        Derivative of complete PML function.
    """
    s_var, t_var, rad_0, rad_1 = sp.symbols("s t R_0 R_1")
    # Integrate on s_var, from rad_0 to t_var; function of t_var
    _unscaled_pml = sp.integrate(
        (s_var - rad_0) ** 2 * (s_var - rad_1) ** 2,
        (s_var, rad_0, t_var),
    ).factor()
    # Total integral from rad_0 to rad_1
    _total = _unscaled_pml.subs(t_var, rad_1).factor()
    # Radial PML function
    pml_func = alpha * _unscaled_pml / _total
    pml_func = pml_func.subs(rad_0, pml_begin).subs(rad_1, pml_end)
    # Remaining term
    complete_pml = 1 + 1j * pml_func
    radial_pml = complete_pml * t_var  # Mapped pml function
    # Derivative of radial PML function
    complete_pml_dt = complete_pml.diff(t_var).factor()
    radial_pml_dt = radial_pml.diff(t_var).factor()
    return radial_pml, complete_pml, radial_pml_dt, complete_pml_dt


def radial_pml_ng(
    alpha: float,
    pml_begin: float,
    pml_end: float,
    dim: int = 2,
) -> Tuple[ng.Expr, ng.Expr]:
    """Radial PML function for NGSolve. Cf. [4].

    Input
    -----
    alpha : float
        PML strength.
    pml_begin : float
        (radial) start of PML region.
    pml_end : float
        (radial) end of PML region.

    Output
    ------
    radial_pml_ng : ngsolve.Expr
        Radial PML function.
    complete_pml_ng : ngsolve.Expr
        Complete PML function (1 + 1j * radial_pml_ng).
    radial_pml_dt_ng : ngsolve.Expr
        Derivative of radial PML function.
    complete_pml_dt_ng : ngsolve.Expr
        Derivative of complete PML function.
    """
    if dim == 2:
        rad = ng.sqrt(ng.x * ng.x + ng.y * ng.y)
    elif dim == 3:
        rad = ng.sqrt(ng.x * ng.x + ng.y * ng.y + ng.z * ng.z)
    else:
        raise ValueError("dim must be 2 or 3")
    radial_pml, complete_pml, radial_pml_dt, complete_pml_dt = radial_pml(
        alpha, pml_begin, pml_end
    )
    # Convert sympy expressions to NGSolve expressions
    _radial_pml_ng = str(radial_pml).replace("I", "1j").replace("t", "rad")
    _complete_pml_ng = str(complete_pml).replace("I", "1j").replace("t", "rad")
    _radial_pml_dt_ng = str(radial_pml_dt).replace("I", "1j").replace("t", "rad")
    _complete_pml_dt_ng = str(complete_pml_dt).replace("I", "1j").replace("t", "rad")

    radial_pml_ng = eval(_radial_pml_ng)
    complete_pml_ng = eval(_complete_pml_ng)
    radial_pml_dt_ng = eval(_radial_pml_dt_ng)
    complete_pml_dt_ng = eval(_complete_pml_dt_ng)
    return radial_pml_ng, complete_pml_ng, radial_pml_dt_ng, complete_pml_dt_ng


def cartesian_pml(
    alpha,
    pml_begin: float,
    pml_end: float,
    order: int = 2,
) -> Tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """Symbolic Cartesian PML function. Cf. [3].

    Assuming all directions dilated equally.

    Input
    -----
    alpha : float
        PML strength.
    pml_begin : float
        (n-directional) start of PML region.
    pml_end : float
        (n-directional) end of PML region.
    order : int
        Order of the PML function. Default is 2.

    Output
    ------
    cartesian_pml : sympy.Expr
        Cartesian PML function.
    complete_pml : sympy.Expr
        Complete PML function (1 + 1j * cartesian_pml).
    TODO
    """
    s_var, t_var, x_0, x_1 = sp.symbols("s t x_0 x_1")
    _directional_pml = ((t_var - x_0) / (x_1 - x_0)) ** order
    pml_func = alpha * _directional_pml
    pml_func = pml_func.subs(x_0, pml_begin).subs(x_1, pml_end)
    complete_pml = t_var + 1j * pml_func
    return complete_pml


def cartesian_pml_ng(
    alpha: float,
    pml_begin: float,
    pml_end: float,
    dim: int = 2,
) -> Tuple[ng.Expr, ng.Expr]:
    """Cartesian PML function for NGSolve. Cf. [3].

    Input
    -----
    alpha : float
        PML strength.
    pml_begin : float
        (n-directional) start of PML region.
    pml_end : float
        (n-directional) end of PML region.

    Output
    ------
    pml_terms : list
        List of PML terms for each direction.
    """
    if dim == 2:
        vars = ng.x, ng.y
    elif dim == 3:
        vars = ng.x, ng.y, ng.z
    else:
        raise ValueError("dim must be 2 or 3")

    cartesian_pml_func = cartesian_pml(alpha, pml_begin, pml_end)
    pml_terms = []
    for var in vars:
        _directional_pml = str(cartesian_pml_func).replace("I", "1j").replace("t", var)
        directional_pml = eval(_directional_pml)
        pml_terms.append(directional_pml)
    return pml_terms


if __name__ == "__main__":
    # Test radial PML
    alpha = 1.0
    pml_begin = 1.0
    pml_end = 2.0
    radial_pml, complete_pml, radial_pml_dt, complete_pml_dt = radial_pml(
        alpha, pml_begin, pml_end
    )
    print(
        f"Radial PML function: {radial_pml},\n"
        f"Complete PML function: {complete_pml},\n"
        f"Radial PML derivative: {radial_pml_dt},\n"
        f"Complete PML derivative: {complete_pml_dt}\n"
    )

    # Test Cartesian PML
    cartesian_pml_func = cartesian_pml(alpha, pml_begin, pml_end)
    print(f"Cartesian PML function: {cartesian_pml_func}\n")

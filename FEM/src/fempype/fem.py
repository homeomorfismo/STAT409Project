"""FEM-formulation of the pipe physics.

Time-harmonic wave equation.
"""

from __future__ import annotations
from typing import Any, Callable, Tuple
import ngsolve as ng

# from .geometry import CoorDict


def tm(callable: Callable) -> Callable:
    """Decorator. Uses task manager to run the function."""

    def wrapper(*args, **kwargs):
        with ng.TaskManager():
            temp = callable(*args, **kwargs)
        return temp

    return wrapper


@tm
def __assemble(*args: Any) -> None:
    """Assemble the given forms."""
    for form in args:
        form.Assemble()


def get_standard_fem(
    mesh: ng.Mesh,
    order: int,
    dim: int,
) -> Tuple[ng.FESpace, ng.BilinearForm]:
    """Create a standard finite element space."""
    h1 = ng.H1(mesh, order=order, dim=dim, dirichlet="dirichlet", complex=True)
    u, v = h1.TnT()

    bil = ng.BilinearForm(h1, symmetric=True)
    bil += ng.grad(u) * ng.grad(v) * ng.dx
    __assemble(bil)

    return h1, bil


if __name__ == "__main__":
    # TODO: Add test cases
    pass

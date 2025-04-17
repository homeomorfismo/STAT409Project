"""FEM-formulation of the pipe physics.

Time-harmonic wave equation.
"""

from __future__ import annotations
from typing import Dict, Any, Union

import ngsolve as ng


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

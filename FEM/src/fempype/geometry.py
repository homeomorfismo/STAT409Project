"""
Define the geometry of the organ pipes.
All pipes have a PML subdomain.
"""

from __future__ import annotations
from typing import Dict, Any, Union
import netgen.occ as occ
import ngsolve as ng
import numpy as np

# type alias
CoorDict = Dict[str, Union[tuple[float, float, float], float]]


def __compute_coordinates(
    length: float,
    width: float,
    height: float,
    diam_toe: float,
    thickness: float,
    air_thickness: float,
    pml_thickness: float,
) -> CoorDict:
    """Compute the coordinates of the square pipe.

    Input
    -----
    length : float
        Length of the pipe.
    width : float
        Width of the pipe.
    height : float
        Height of the pipe.
    thickness : float
        Thickness of the pipe walls.
    air_thickness : float
        Thickness of the air layer.
    pml_thickness : float
        Thickness of the PML layer.
    """
    assert (
        length > 0
        and width > 0
        and height > 0
        and diam_toe > 0
        and thickness > 0
        and air_thickness > 0
        and pml_thickness > 0
    ), "All dimensions must be positive"
    assert length > 2 * thickness, "Length must be larger than twice the thickness"
    assert (
        width > 2 * thickness + diam_toe and height > 2 * thickness + diam_toe
    ), "Toe diameter must be smaller than the dimensions"

    coords = {
        # inner box
        "ibox1": (
            thickness - length / 2,
            thickness - width / 2,
            thickness - height / 2,
        ),
        "ibox2": (
            length / 2,  # accounts for opening
            width / 2 - thickness,
            height / 2 - thickness,
        ),
        # outer box
        "obox1": (-length / 2, -width / 2, -height / 2),
        "obox2": (length / 2, width / 2, height / 2),
        # inner PML box/air box
        "ipml1": (
            -length / 2 - air_thickness,
            -width / 2 - air_thickness,
            -height / 2 - pml_thickness,
        ),
        "ipml2": (
            length / 2 + air_thickness,
            width / 2 + air_thickness,
            height / 2 + pml_thickness,
        ),
        # outer PML box
        "opml1": (
            -length / 2 - pml_thickness - air_thickness,
            -width / 2 - pml_thickness - air_thickness,
            -height / 2 - pml_thickness - air_thickness,
        ),
        "opml2": (
            length / 2 + pml_thickness + air_thickness,
            width / 2 + pml_thickness + air_thickness,
            height / 2 + pml_thickness + air_thickness,
        ),
        # toe
        "toe_ci": (-length / 2 + thickness, 0.0, 0.0),
        "toe_co": (-length / 2, 0.0, 0.0),
        "toe_h": thickness,
        "toe_r": diam_toe / 2,
    }
    return coords


def square_pipe(
    *,
    length: float = 1.0,
    width: float = 1.0,
    height: float = 1.0,
    diam_toe: float = 0.1,
    thickness: float = 0.01,
    air_thickness: float = 1.0,
    pml_thickness: float = 2.0,
) -> Any:
    """Create a square pipe geometry with PML subdomain.

    Input
    -----
    length : float
        Length of the pipe.
    width : float
        Width of the pipe.
    height : float
        Height of the pipe.
    diam_toe : float
        Diameter of the toe.
    thickness : float
        Thickness of the pipe walls.
    air_thickness : float
        Thickness of the air layer.
    pml_thickness : float
        Thickness of the PML layer.

    Returns
    -------
    mesh : ng.Mesh
        The mesh of the square pipe.
    """
    coord: CoorDict = __compute_coordinates(
        length, width, height, diam_toe, thickness, air_thickness, pml_thickness
    )
    inner_box = occ.Box(occ.Pnt(*coord["ibox1"]), occ.Pnt(*coord["ibox2"]))
    outer_box = occ.Box(occ.Pnt(*coord["obox1"]), occ.Pnt(*coord["obox2"]))
    pml_box = occ.Box(occ.Pnt(*coord["ipml1"]), occ.Pnt(*coord["ipml2"]))
    outer_pml_box = occ.Box(occ.Pnt(*coord["opml1"]), occ.Pnt(*coord["opml2"]))
    toe = occ.Cylinder(
        occ.Pnt(*coord["toe_co"]), occ.X, r=coord["toe_r"], h=coord["toe_h"],
    )

    outer_pml_box.bc("dirichlet")

    pml_region = outer_pml_box - pml_box
    air_region = (pml_box - outer_box) + inner_box + toe
    box = (outer_box - inner_box) - toe

    pml_region.mat("pml")
    air_region.mat("air")
    box.mat("solid")

    geo = occ.OCCGeometry(occ.Glue([pml_region, air_region, box]))
    return geo

if __name__ == "__main__":
    # Example usage
    geo = square_pipe(
        length=1.0,
        width=1.0,
        height=1.0,
        diam_toe=0.1,
        thickness=0.01,
        air_thickness=1.0,
        pml_thickness=2.0,
    )
    ng_mesh = geo.GenerateMesh(maxh=0.1)
    mesh = ng.Mesh(ng_mesh)
    mesh.Curve(3)
    ng.Draw(mesh)

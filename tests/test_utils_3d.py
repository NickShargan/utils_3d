# tests/test_utils_3d.py
import os
import math
import vtk
import pytest

from utils_3d import (
    build_sphere,
    build_cone,
    save_mesh,
)

def test_build_sphere_bounds():
    r = 1.25
    cx, cy, cz = 0.0, 0.0, 0.0
    mesh = build_sphere(radius=r, center=(cx, cy, cz), theta_res=24, phi_res=24)

    assert isinstance(mesh, vtk.vtkPolyData)
    assert mesh.GetNumberOfPoints() > 0
    assert mesh.GetNumberOfPolys() > 0

    xmin, xmax, ymin, ymax, zmin, zmax = mesh.GetBounds()
    # Bounds should be about [c - r, c + r] on each axis
    assert xmin == pytest.approx(cx - r, rel=0.02, abs=1e-3)
    assert xmax == pytest.approx(cx + r, rel=0.02, abs=1e-3)
    assert ymin == pytest.approx(cy - r, rel=0.02, abs=1e-3)
    assert ymax == pytest.approx(cy + r, rel=0.02, abs=1e-3)
    assert zmin == pytest.approx(cz - r, rel=0.02, abs=1e-3)
    assert zmax == pytest.approx(cz + r, rel=0.02, abs=1e-3)


def test_build_cone_bounds():
    height = 2.0
    radius = 0.75
    # Keep center at origin and default direction +Z for simple bounds checks
    mesh = build_cone(height=height, radius=radius, center=(0.0, 0.0, 0.0), resolution=48)

    assert isinstance(mesh, vtk.vtkPolyData)
    assert mesh.GetNumberOfPoints() > 0
    assert mesh.GetNumberOfPolys() > 0

    xmin, xmax, ymin, ymax, zmin, zmax = mesh.GetBounds()

    # For a cone along +Z centered at origin, z extent ~ height
    assert (zmax - zmin) == pytest.approx(height, rel=0.03, abs=1e-3)

    # The max radius should show up in x/y bounds roughly at the base: ~±radius
    assert max(abs(xmin), abs(xmax)) == pytest.approx(radius, rel=0.05, abs=1e-2)
    assert max(abs(ymin), abs(ymax)) == pytest.approx(radius, rel=0.05, abs=1e-2)


def test_save_mesh(tmp_path):
    mesh = build_sphere(radius=0.4)
    out_path = tmp_path / "sphere.obj"

    save_mesh(mesh, str(out_path))

    assert out_path.exists()
    # File should be non-trivial size
    assert out_path.stat().st_size > 100

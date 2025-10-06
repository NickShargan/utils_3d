# tests/test_utils_3d.py
import os
import math
import vtk
import pytest

from utils_3d import (
    build_sphere,
    build_cone,
    scale_mesh,
    is_intersected,
    cut_polydata
)

from utils_vtk import merge_polydata

from io_mesh import write_mesh


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


def test_write_mesh(tmp_path):
    mesh = build_sphere(radius=0.4)
    out_path = tmp_path / "sphere.obj"

    write_mesh(mesh, str(out_path))

    assert out_path.exists()
    # File should be non-trivial size
    assert out_path.stat().st_size > 100


def _center_and_halfwidths(bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)
    hx = 0.5 * (xmax - xmin)
    hy = 0.5 * (ymax - ymin)
    hz = 0.5 * (zmax - zmin)
    return (cx, cy, cz), (hx, hy, hz)


def test_scale_mesh_sphere_extents_and_center():
    mesh = build_sphere(radius=1.0, center=(1.0, -2.0, 0.5), theta_res=24, phi_res=24)
    s = 2.5
    out = scale_mesh(mesh, s)

    assert isinstance(out, vtk.vtkPolyData)
    c1, h1 = _center_and_halfwidths(mesh.GetBounds())
    c2, h2 = _center_and_halfwidths(out.GetBounds())

    # Center should remain (approximately) unchanged
    assert c2[0] == pytest.approx(c1[0], abs=1e-6)
    assert c2[1] == pytest.approx(c1[1], abs=1e-6)
    assert c2[2] == pytest.approx(c1[2], abs=1e-6)

    # Half-widths (radii along axes) should scale by s
    assert h2[0] == pytest.approx(h1[0] * s, rel=0.02)
    assert h2[1] == pytest.approx(h1[1] * s, rel=0.02)
    assert h2[2] == pytest.approx(h1[2] * s, rel=0.02)


def test_scale_mesh_rejects_nonpositive():
    mesh = build_sphere(radius=1.0)
    with pytest.raises(ValueError):
        scale_mesh(mesh, 0.0)
    with pytest.raises(ValueError):
        scale_mesh(mesh, -1.0)


def test_is_intersected_bool():
    # prepare geometry with possible multiple intersections
    mesh_cone_1 = build_cone(height=1.5, radius=0.5, center=(1.0, 0.0, 0.0))
    mesh_cone_2 = build_cone(height=1.5, radius=0.5, center=(1.0, 0.5, 0.0))

    mesh_cones = merge_polydata(mesh_cone_1, mesh_cone_2)

    a, b, c, d = 0.0, 0.0, 1.0, 1.5
    plane_params = [a, b, c, d]

    res = is_intersected(mesh_cones, plane_params)
    
    # should not intersect
    assert not res

    a, b, c, d = 0.0, 0.0, 1.0, 0.5
    plane_params = [a, b, c, d]

    res = is_intersected(mesh_cones, plane_params)

    assert res

    
def test_is_intersected_poly():
    # prepare geometry with possible multiple intersections
    mesh_cone_1 = build_cone(height=1.5, radius=0.5, center=(1.0, 0.0, 0.0))
    mesh_cone_2 = build_cone(height=1.5, radius=0.5, center=(1.0, 0.5, 0.0))

    mesh_cones = merge_polydata(mesh_cone_1, mesh_cone_2)

    a, b, c, d = 0.0, 0.0, 1.0, 0.5
    plane_params = [a, b, c, d]
    
    cut_poly = cut_polydata(mesh_cones, plane_params)
    num_pts = cut_poly.GetNumberOfPoints()
    num_cells = cut_poly.GetNumberOfCells()
    print(f"Cut: {num_pts} points, {num_cells} polylines")

    assert num_cells == 2

    assert num_pts == 64
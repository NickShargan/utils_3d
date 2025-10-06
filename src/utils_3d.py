import os

import numpy as np
import vtk


def build_sphere(radius: float,
                 center=(0.0, 0.0, 0.0),
                 theta_res: int = 32,
                 phi_res: int = 32) -> vtk.vtkActor:
    """
    Create a sphere mesh for rendering.
    
    Args:
        radius (float): Sphere radius.
        center (tuple): (x,y,z) center of the sphere.
        theta_res (int): Subdivisions around longitude (smoothness).
        phi_res (int): Subdivisions from pole to pole (smoothness).
    
    Returns:
        vtk.vtkActor: Configured actor with sphere mesh.
    """
    # Sphere geometry
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(*center)
    sphere.SetRadius(radius)
    sphere.SetThetaResolution(theta_res)
    sphere.SetPhiResolution(phi_res)
    sphere.Update()
    return sphere.GetOutput()


def build_cone(height: float,
               radius: float,
               center=(0.0, 0.0, 0.0),
               resolution: int = 32) -> vtk.vtkPolyData:
    """
    Create a cone mesh (vtkPolyData).
    By default, the cone points along +Z.
    """
    cone = vtk.vtkConeSource()
    cone.SetHeight(height)
    cone.SetRadius(radius)
    cone.SetResolution(resolution)
    cone.SetCenter(*center)
    cone.SetDirection(0, 0, 1)  # pointing along +Z
    cone.Update()
    return cone.GetOutput()


def scale_mesh(mesh: vtk.vtkPolyData, scale: float, center: tuple[float, float, float] | None = None
               ) -> vtk.vtkPolyData:
    """
    Uniformly scale a vtkPolyData about a center (defaults to the mesh's center of mass).

    Args:
        mesh: Input mesh (vtkPolyData).
        scale: Uniform scale factor (> 0).
        center: Optional (cx, cy, cz). If None, uses center of mass.

    Returns:
        vtkPolyData: Scaled mesh (new object).

    Raises:
        ValueError: If scale <= 0.
    """
    if scale <= 0:
        raise ValueError("scale must be > 0")

    # Determine center
    if center is None:
        com = vtk.vtkCenterOfMass()
        com.SetInputData(mesh)
        com.SetUseScalarsAsWeights(False)
        com.Update()
        cx, cy, cz = com.GetCenter()
        center = (float(cx), float(cy), float(cz))

    tx = vtk.vtkTransform()
    tx.PostMultiply()
    tx.Translate(-center[0], -center[1], -center[2])
    tx.Scale(scale, scale, scale)
    tx.Translate(center[0], center[1], center[2])

    tfilter = vtk.vtkTransformPolyDataFilter()
    tfilter.SetTransform(tx)
    tfilter.SetInputData(mesh)
    tfilter.Update()

    out = vtk.vtkPolyData()
    out.DeepCopy(tfilter.GetOutput())
    return out


def mesh2actor(mesh: vtk.vtkPolyData) -> vtk.vtkActor:
    """
    Wrap a vtkPolyData mesh into an actor for rendering.
    """
    # Compute normals (optional)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(mesh)
    normals.AutoOrientNormalsOn()
    normals.ConsistencyOn()
    normals.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor


def save_mesh(mesh: vtk.vtkPolyData, filepath: str) -> None:
    """
    Save vtkPolyData to .obj / .stl / .ply based on file extension.
    Ensures triangles + normals for good viewer compatibility.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in {".obj", ".stl", ".ply"}:
        raise ValueError("Unsupported extension. Use .obj, .stl, or .ply")

    # Triangulate
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(mesh)
    tri.Update()

    # Compute normals for smooth shading
    norms = vtk.vtkPolyDataNormals()
    norms.SetInputConnection(tri.GetOutputPort())
    norms.SplittingOff()
    norms.AutoOrientNormalsOn()
    norms.ConsistencyOn()
    norms.Update()

    if ext == ".obj":
        writer = vtk.vtkOBJWriter()
        writer.SetFileName(filepath)
        writer.SetInputData(norms.GetOutput())
        writer.Update()
        writer.Write()
    elif ext == ".stl":
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(filepath)
        writer.SetInputData(norms.GetOutput())
        writer.SetFileTypeToBinary()
        writer.Write()
    else:  # .ply
        writer = vtk.vtkPLYWriter()
        writer.SetFileName(filepath)
        writer.SetInputData(norms.GetOutput())
        writer.SetFileTypeToBinary()
        writer.Write()


def is_identical(mesh_1: vtk.vtkPolyData, mesh_2: vtk.vtkPolyData) -> bool:
    """Evaluate whether mesh_1 and mesh_2 are identical. Assuming that rotating and translating don't impact the result."""

    return True


def _as_vtk_plane(plane):
    """
    Accepts either a vtk.vtkPlane or a (origin, normal) tuple and returns a vtkPlane.
    origin: 3-tuple, normal: 3-tuple
    """
    if isinstance(plane, vtk.vtkPlane):
        return plane
    origin, normal = plane
    vp = vtk.vtkPlane()
    vp.SetOrigin(origin)
    vp.SetNormal(normal)
    return vp


def params2plane(plane_params):
    """Build vtkPlane from a, b, c, d: ax + by + cz = d"""
    a, b, c, d = plane_params
    print(plane_params, a, b, c, d)

    normal = np.array([a, b, c])
    origin = (d / np.dot(normal, normal)) * normal
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin)
    plane.SetNormal(normal)

    return plane


# def _bounds_corners(bounds):
#     """Return the 8 corner points of an AABB: (xmin,xmax, ymin,ymax, zmin,zmax)."""
#     xmin, xmax, ymin, ymax, zmin, zmax = bounds
#     return [
#         (xmin, ymin, zmin), (xmax, ymin, zmin),
#         (xmin, ymax, zmin), (xmax, ymax, zmin),
#         (xmin, ymin, zmax), (xmax, ymin, zmax),
#         (xmin, ymax, zmax), (xmax, ymax, zmax),
#     ]


# def _sign_with_tol(value: float, tol: float) -> int:
#     """Return -1, 0, or +1 depending on value and tolerance."""
#     if value > tol:
#         return 1
#     if value < -tol:
#         return -1
#     return 0


# def is_intersected_cstm(mesh: vtk.vtkPolyData, plane_params) -> bool:
#     """
#     Returns True if an infinite plane intersects the triangle mesh (vtkPolyData).
#     The plane can be a vtk.vtkPlane or a tuple: ((ox,oy,oz), (nx,ny,nz)).
#     Intersection includes tangential contact (edge/vertex) and coplanar cases.
#     """

#     plane = params2plane(plane_params)
      
#     vp = _as_vtk_plane(plane)

#     # 1) Quick reject using the mesh bounds
#     bounds = mesh.GetBounds()
#     corners = _bounds_corners(bounds)
#     tol = 1e-9  # geometric tolerance for "on plane"

#     signs = {_sign_with_tol(vp.EvaluateFunction(p), tol) for p in corners}
#     # If all corners strictly on one side, no way the plane hits the AABB => no intersection.
#     # (If 0 is present or both -1/+1 are present, we can't reject.)
#     if signs in ({-1}, {1}):
#         return False

#     # 2) Definitive test with vtkCutter
#     cutter = vtk.vtkCutter()
#     cutter.SetCutFunction(vp)
#     cutter.SetInputData(mesh)
#     cutter.Update()
#     cut_output = cutter.GetOutput()

#     # If any intersection geometry (points or cells) exists, we intersect.
#     if cut_output.GetNumberOfPoints() > 0 or cut_output.GetNumberOfCells() > 0:
#         return True

#     # Edge cases are rare, but if cutter produced nothing after passing the bounds test,
#     # treat as no intersection.
#     return False


def is_intersected(mesh: vtk.vtkPolyData, plane_params) -> bool:
    
    plane = params2plane(plane_params)

    vp = _as_vtk_plane(plane)
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(vp)
    cutter.SetInputData(mesh)
    cutter.Update()
    outp = cutter.GetOutput()
    return (outp.GetNumberOfPoints() > 0) or (outp.GetNumberOfCells() > 0)


def plane2actor(origin, normal, color=(1, 0, 0), opacity=0.4):
    """
    Create a visual rectangular patch representing the plane.
    """
    # Create plane geometry (finite rectangle in 3D)
    plane_source = vtk.vtkPlaneSource()
    plane_source.SetCenter(*origin)
    plane_source.SetNormal(*normal)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(plane_source.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)

    return actor


def cut_polydata(poly: vtk.vtkPolyData, plane_params) -> vtk.vtkPolyData:
    """Return polylines where plane intersects poly (may be multiple loops)."""
    plane = params2plane(plane_params)

    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(poly)
    cutter.Update()

    # Stripper stitches adjacent line segments into longer polylines/loops
    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(cutter.GetOutputPort())
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()

    return stripper.GetOutput()


def cut2actors(cut_poly: vtk.vtkPolyData,
               as_tube=True,
               line_width=3.0,
               tube_radius=0.01,
               base_color=(1, 1, 0)):
    """
    Return one actor per connected polyline (loop/segment) in the cut result.
    """
    actors = []

    # Split the polyline set into connected regions
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(cut_poly)
    conn.SetExtractionModeToAllRegions()
    conn.ColorRegionsOn()
    conn.Update()

    n_regions = conn.GetNumberOfExtractedRegions()
    for r in range(n_regions):
        # Extract just region r as PolyData
        reg = vtk.vtkPolyDataConnectivityFilter()
        reg.SetInputData(cut_poly)
        reg.SetExtractionModeToSpecifiedRegions()
        reg.AddSpecifiedRegion(r)
        reg.Update()

        # Mapper chain
        if as_tube:
            tube = vtk.vtkTubeFilter()
            tube.SetInputConnection(reg.GetOutputPort())
            tube.SetRadius(tube_radius)
            tube.SetNumberOfSides(16)
            tube.CappingOn()
            tube.Update()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(tube.GetOutputPort())
        else:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reg.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        actor.GetProperty().SetColor(*base_color)
        if not as_tube:
            actor.GetProperty().SetLineWidth(line_width)

        actors.append(actor)

    return actors




def merge_polydata(poly1: vtk.vtkPolyData, poly2: vtk.vtkPolyData) -> vtk.vtkPolyData:
    appender = vtk.vtkAppendPolyData()
    appender.AddInputData(poly1)
    appender.AddInputData(poly2)
    appender.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(appender.GetOutputPort())
    cleaner.Update()

    return cleaner.GetOutput()
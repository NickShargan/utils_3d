import os

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

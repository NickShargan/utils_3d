import os
import vtk


def read_mesh(path: str) -> vtk.vtkPolyData:
    ext = os.path.splitext(path.lower())[1]
    if ext == ".obj":
        r = vtk.vtkOBJReader()
    elif ext == ".stl":
        r = vtk.vtkSTLReader()
    else:
        raise ValueError(f"Unsupported mesh format: {ext} (use .obj or .stl)")
    r.SetFileName(path)
    r.Update()
    poly = r.GetOutput()
    if not isinstance(poly, vtk.vtkPolyData) or poly.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Failed to read mesh or mesh is empty: {path}")
    return poly


def write_mesh(poly: vtk.vtkPolyData, path: str) -> str:
    ext = os.path.splitext(path.lower())[1]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if ext == ".obj":
        w = vtk.vtkOBJWriter()
        w.SetFileName(path)
        w.SetInputData(poly)
        w.Update()
        w.Write()
    elif ext == ".stl":
        w = vtk.vtkSTLWriter()
        w.SetFileName(path)
        w.SetInputData(poly)
        w.Write()
    else:
        raise ValueError(f"Unsupported mesh format: {ext} (use .obj or .stl)")
    return path

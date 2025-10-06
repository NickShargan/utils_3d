import vtk


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
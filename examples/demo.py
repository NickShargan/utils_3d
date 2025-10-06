import numpy as np
import vtk

from utils_3d import build_cone, build_sphere, mesh2actor, save_mesh, scale_mesh,\
                     is_intersected, plane2actor, cut_polydata,\
                     cut2actors, merge_polydata


def main():
    # --- Renderer setup ---
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.1, 0.1, 0.12)

    # Create meshes and corresponding actors
    mesh_sphere = build_sphere(radius=0.5, center=(-1.0, 0.0, 0.0))
    actor = mesh2actor(mesh=mesh_sphere)
    renderer.AddActor(actor)

    mesh_cone = build_cone(height=1.5, radius=0.5, center=(1.0, 3.0, 0.0))
    mesh_cone_trans = build_cone(height=1.5, radius=0.5, center=(1.0, 0.0, 0.0))
    actor = mesh2actor(mesh=mesh_cone_trans)
    renderer.AddActor(actor)

    mesh_cone_2 = build_cone(height=1.5, radius=0.5, center=(1.0, 0.5, 0.0))
    actor = mesh2actor(mesh=mesh_cone_2)
    renderer.AddActor(actor)

    # Save to OBJ
    save_mesh(mesh_sphere, "sphere.obj")
    save_mesh(mesh_cone, "cone.obj")

    # scale cone x2 times
    mesh_cone_x2 = scale_mesh(mesh_cone, 2.0)
    actor_c = mesh2actor(mesh=mesh_cone_x2)
    renderer.AddActor(actor_c)

    # intersection with plane

    a, b, c, d = 0.0, 0.0, 1.0, 0.5

    # origin = [-1.0, 1.5, 0.0]
    origin = [1.0, 0.3, 0.5]
    normal = [0.0, 0.0, 1.0]
    # normal = [a, b, c]
    # origin = (d / np.dot(normal, normal)) * normal
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin)
    plane.SetNormal(normal)

    mesh_cones = merge_polydata(mesh_cone_trans, mesh_cone_2)

    res = is_intersected(mesh_cones, plane)
    print("is_intersected(mesh, plane) = ", res)
    print(f"plane origin: {origin}; normal: {normal}")

    # origin = [-1.0, 0.5, 0.0]
    plane.SetOrigin(origin)
    plane.SetNormal(normal)

    res = is_intersected(mesh_cones, plane)
    print("is_intersected(mesh, plane) = ", res)
    print(f"plane origin: {origin}; normal: {normal}")
    
    actor_p = plane2actor(origin, normal)
    renderer.AddActor(actor_p)

    # polyline at intersection
    cut_poly = cut_polydata(mesh_cones, origin, normal)
    num_pts = cut_poly.GetNumberOfPoints()
    num_cells = cut_poly.GetNumberOfCells()
    print(f"Cut: {num_pts} points, {num_cells} polylines")

    actors_cut = cut2actors(cut_poly, as_tube=True, tube_radius=0.012)
    print(len(actors_cut))
    for actor_cut in actors_cut:
        renderer.AddActor(actor_cut)

    # Render window & interactor
    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(renderer)
    ren_win.SetSize(800, 600)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)

    # Camera setup
    renderer.ResetCamera()
    renderer.GetActiveCamera().Elevation(20)
    renderer.GetActiveCamera().Azimuth(30)

    # Render loop
    ren_win.Render()
    iren.Initialize()
    iren.Start()


if __name__ == "__main__":
    main()
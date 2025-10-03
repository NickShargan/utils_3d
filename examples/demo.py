import vtk

from utils_3d import build_cone, build_sphere, mesh2actor, save_mesh


def main():
    # --- Renderer setup ---
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.1, 0.1, 0.12)

    # Create meshes and corresponding actors
    mesh_sphere = build_sphere(radius=0.5, center=(-1.0, 0.0, 0.0))
    actor = mesh2actor(mesh=mesh_sphere)
    renderer.AddActor(actor)

    mesh_cone = build_cone(height=1.5, radius=0.5, center=(1.0, 0.0, 0.0))
    actor = mesh2actor(mesh=mesh_cone)
    renderer.AddActor(actor)

    # Save to OBJ
    save_mesh(mesh_sphere, "sphere.obj")
    save_mesh(mesh_cone, "cone.obj")

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
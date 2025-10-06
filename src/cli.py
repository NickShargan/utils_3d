# src/cli.py
import argparse
import sys

from io_mesh import read_mesh, write_mesh

from utils_3d import (
    build_sphere,
    build_cone,
    scale_mesh,
    is_intersected,
    cut_polydata
)


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate meshes and basic mesh ops (OBJ/STL).")
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- sphere ---
    sp = sub.add_parser("sphere", help="Generate a sphere mesh and save to file.")
    sp.add_argument("--radius", type=float, required=True)
    sp.add_argument("--out", required=True)

    # --- cone ---
    co = sub.add_parser("cone", help="Generate a cone mesh and save to file.")
    co.add_argument("--radius", type=float, required=True)
    co.add_argument("--height", type=float, required=True)
    co.add_argument("--out", required=True)

    # --- scale ---
    sc = sub.add_parser("scale", help="Scale a mesh uniformly around its centroid.")
    sc.add_argument("--mesh", required=True, help="Path to input mesh (.obj or .stl)")
    sc.add_argument("--coef", type=float, required=True, help="Uniform scaling coefficient")
    sc.add_argument("--out", required=True, help="Output path")

    # --- is_intersection ---
    ix = sub.add_parser("is_intersect", help="Check if plane ax+by+cz=d intersects the mesh.")
    ix.add_argument("--mesh", required=True, help="Path to input mesh (.obj or .stl)")
    ix.add_argument("--a", type=float, required=True)
    ix.add_argument("--b", type=float, required=True)
    ix.add_argument("--c", type=float, required=True)
    ix.add_argument("--d", type=float, required=True)
    ix.add_argument("--out", required=False, help="Output path")

    args = p.parse_args(argv)

    if args.cmd == "sphere":
        mesh = build_sphere(radius=args.radius)
        write_mesh(mesh, args.out)
        return 0

    if args.cmd == "cone":
        mesh = build_cone(radius=args.radius, height=args.height)
        write_mesh(mesh, args.out)
        return 0

    if args.cmd == "scale":
        mesh = read_mesh(args.mesh)
        mesh_scaled = scale_mesh(mesh, args.coef)
        write_mesh(mesh_scaled, args.out)
        print(f"mesh was written to {args.out}")
        return 0

    if args.cmd == "is_intersect":
        mesh = read_mesh(args.mesh)
        plane_params = [args.a, args.b, args.c, args.d]
        ok = is_intersected(mesh, plane_params)
        print("true" if ok else "false")

        if args.out:
            cut_poly = cut_polydata(mesh, plane_params)
            write_mesh(cut_poly, args.out)

        return 0 if ok else 1

    p.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())

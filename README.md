# Utils for 3D meshes

To run demo that renders generated sphere and cone run: 
```
pip install -e .
python ./examples/demo.py
```

![img](./img/render_exmpl.png)


## Intersection with plane

```
is_intersected(mesh, plane) =  True
plane origin: [-1.0, 0.5, 0.0]; normal: [0.0, 2.0, 1.0]
```

![img](./img/intersect_true.png)

```
is_intersected(mesh, plane) =  False
plane origin: [-1.0, 1.5, 0.0]; normal: [0.0, 2.0, 1.0]
```

![img](./img/intersect_false.png)

```
Cut: 50 points, 1 polylines
```

![img](./img/intersect_polyline.png)

```
Cut: 64 points, 2 polylines
```

![img](./img/intersect_polylines.png)

## Asymptotic complexity analysis

### true / false option

For cases when all bounds (that are cached for vtk polydata) of mesh are located on the same side of plane, complexity is constant - O(1).

Alternatively, each face (triangle) of the mesh should be checked for the same condition: all vertices are located on the same side of the plane. Considering that verifying this condition requires computing dot product (O(1)) for each vertex to define sign of the distance to plane, overall complexity in the worst case scenarion is O(N) where N is number of faces. 

### defining polyline/perimeter of intersection

The complexity is still linear. 

For each triangle which has distances from vertices to plane not with the same sign, it is neccessary to define 2 points of intersection. These 2 points will form segmet of the final polyline. For each pair of poins with different signs, point of intersection can be defined as interpolation weighted by distance to the plane:

```
p_1 = (x_1, y_1, z_1)
p_2 = (x_2, y_2, z_2)

p_intr = p_1 + (p_2 - p_1) * (dist_1 / (dist_1 + dist_2))
```

Considering it takes constant time, the complexity for all vertices will rematin O(N)

In terms of memory the same O(N) applies for both cases to store the input, since no additional data are kept during computations (like for dynamic programing).

## CLI

Use these function as CLI app:

```
pip install -e .

utils-3d sphere --radius 1.0 --out sphere.obj  

utils-3d cone --radius 1.0 --height 2.0 --out cone.obj 

utils-3d scale --mesh sphere.obj --coef 1.5 --out sphere_scaled.obj

# True & writes polylines to *.obj file
utils-3d is_intersect --mesh cone.obj --a 0.0 --b 0.0 --c 1.0 --d 0.5 --out cut.obj

# False
utils-3d is_intersect --mesh cone.obj --a 0.0 --b 0.0 --c 1.0 --d 1.5
```

## Optimisation problem (Task 2)

Let's assume that the given non-convex, non-symmetrical volume is defined by a 3d mesh.

One of solutions (not likely the optimal one) would be applying an algorithm similar to Region Growth. 
The seed (1st sphere) could be initialised randomly. For example, we can define bounds of the defined volume and randomly select coordinates until sphere will be fully inside the volume.
Or seed can be initialised near one of the faces (in opposite direction from the normal) on distance <sphere_radius>.

def is_sphere_inside(center_coords, radius=1) 

Assumption.
Let's assume that the volume doesn't have areas that are more narrow that the small sphere and all spheres will be packed into a single connected blob.

So, basically, at the beginning we are going to initialise origin and direction, and then apply region growth. This process will be repeated with 2 parameters (1 point and 1 vector) with continuous values in some range: center, direction. Ranges of values for coordinates of center could be [-<sphere_radius>, +<sphere_radius>] with step <sphere_radius>/100 and directions in 2d squares with <sphere_radius>/100 size.

Such logic could be well parellalised beyond 32 kernels that are common for CPUs. So, using language like Cuda could be benefitial for speed by utilising GPU computations.

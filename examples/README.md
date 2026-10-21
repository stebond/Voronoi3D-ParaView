# Example data

| File | Content |
|---|---|
| `test_4points_density.vtk` | 4 random points in the unit cube with `Density` = 1, 2, 3, 4. Quick check that the density array is read (all points are fixed anchors). |
| `test_1208points_2planes_density.vtk` | 604 points at z = 0.25 and the same (x, y) at z = 0.75. The four points near the corners of the square, on both planes, have `Density` = 1, 3, 10, 30; all others 0. Used in the manual (weighted Lloyd, Z movement disabled). |

Open them in ParaView (legacy VTK, ASCII) and apply **Filters > Alphabetical > Voronoi3D From Input**.
Suggested settings for the second file: Density array = `Density`, *Move seeds along Z* off,
Lloyd iterations = 50, Density influence radius = 0.5, Density exponent = 3.

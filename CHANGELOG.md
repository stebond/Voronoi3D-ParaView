# Changelog

## 1.1.0 - 2026-11-09
* Lloyd convergence diagnostics: mean/max seed displacement (relative to the reference cell size h = (V/N)^(1/3)) and coefficient of variation of the cell volumes are reported for every iteration in the Output Messages.
* New property *Convergence threshold* (default 0.005, 0 = off): Lloyd iterations stop when the mean seed shift falls below this fraction of h. The maximum number of iterations is kept as upper limit, and a message states whether the threshold or the iteration limit stopped the optimisation.
* Native engine ABI 7 (`v3d_compute_ex5`, with `shift_tol` and `iterations_done`); older engine libraries remain compatible (no convergence report).
* MSVC warnings C4804 fixed in both engines.
* OpenMP multi-core engine (`engine_omp/`, `voronoi3d_engine_omp.dll`): same results as the serial engine, cells computed in parallel; new engine entry and advanced property *Number of threads*.
* OpenMP engine: the assembly of the output (vertex merging, face lists) is now parallel too (open-addressing vertex table, spatial chunks); results identical in geometry, only the vertex numbering differs from the serial engine.
* Benchmark script lists the OpenMP engine and reports the serial/OpenMP speed-up.

## 1.0.0 - 2026-10-19
Initial public version.

* ParaView plugin with a source (`Voronoi3D`) and a filter (`Voronoi3D From Input`).
* Native C++ engine (Voro++), Python engines (VTK Delaunay, SciPy, `tess`) as fallbacks.
* Lloyd iterations, wall clearance, minimum separation, duplicate removal.
* Density weighted (centroidal) Lloyd iterations with fixed anchor points.
* Per-axis movement of the seeds (planar and layered point sets).
* Seed export (second output port, `SeedPosition`/`InitialPosition` arrays).
* Progress bar with cancellation, calculation time and machine information in the output.

OpenMP (multi-core) variant of the native engine
================================================
Same C interface and same results as ../engine, but the loop over the Voronoi cells
(and the density-weighted centroids) runs on several threads with OpenMP.

* Own folder and own DLL: voronoi3d_engine_omp.dll. The serial engine (../engine,
  voronoi3d_engine.dll) is NOT modified and keeps working.
* In ParaView choose  Calculation engine = "Native C++ OpenMP (Voro++ DLL, multi-core)".
  The advanced property "Number of threads" selects the threads (0 = all cores).
* Build (Windows): open "x64 Native Tools Command Prompt for VS", cd to this folder,
  run build_engine_omp.bat. It uses /openmp, so the DLL needs vcomp140.dll at run
  time (Visual C++ Redistributable, already present where Visual Studio is installed).
  Build (Linux/macOS/MinGW): ./build_engine_omp.sh
* Design: Voro++ (0.4.6 sources in voro/, same two patches as ../engine) is used
  read-only by all threads; every thread owns a voro::voro_compute workspace and a
  voro::voronoicell. Cells are stored in slots indexed by the serial order, so the output
  (cell order, vertex ids, volumes, seeds) is identical to the serial engine.
  Vertex merging and output assembly remain serial. Progress and cancellation are
  handled by the calling thread.
* Voro++'s own multi-threaded extension (Lu, Lazar, Rycroft, Comput. Phys. Commun. 291
  (2023) 108832) lives in the "dev" branch of https://github.com/chr1shr/voro with a
  different API; it is not used here.

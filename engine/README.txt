Voronoi 3D native engine
========================

voronoi3d_engine.cpp   C++ engine (plain C interface, loaded by the plugin with ctypes)
voro/                  Voro++ sources (BSD license, see voro/LICENSE)
build_engine.bat       builds voronoi3d_engine.dll with MSVC
build_engine.sh        same with g++ (Linux/macOS/MinGW)

Build on Windows (MSVC)
-----------------------
1. Start menu -> "x64 Native Tools Command Prompt for VS 20xx"
   (comes with Visual Studio or "Build Tools for Visual Studio"; the C++ workload is enough).
2. cd to this engine folder, then run:   build_engine.bat
3. The script creates voronoi3d_engine.dll here and copies it next to voronoi3d_plugin.py.
4. Restart ParaView, load voronoi3d_plugin.py, set "Calculation engine" to Automatic
   (the native engine is tried first) or "Native C++". The Output Messages window shows
   "native engine loaded from ..." when it works, or the reason why it did not.

Notes
-----
* ABI 5 (this version) adds the per-axis movement option (Lloyd) on top of the progress bar
  (ABI 4) and the density weighted Lloyd (ABI 3). An older DLL still works for the plain
  Voronoi/Lloyd, but the newer options need the DLL rebuilt with build_engine.bat.
* The DLL must be 64-bit, like ParaView. It does not need Python or ParaView to build.
* If the DLL is missing or fails to load, the plugin keeps working with the Python engines
  (much slower). They need 'Min distance from walls' > 0 (default 0.01): seeds lying on or
  extremely close to a wall make them produce wrong cells that stick out of the box.
* Voro++ was modified in voro/common.cc and voro/cell.cc only: fatal errors throw a C++
  exception (caught by the engine and reported to ParaView) instead of calling exit(),
  which would close ParaView.
* Voro++ (c) The Regents of the University of California, Lawrence Berkeley National
  Laboratory. See voro/LICENSE.

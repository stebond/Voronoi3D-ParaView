#!/bin/sh
# Builds the native engine with g++ (Linux/macOS/MinGW). On Windows use build_engine.bat.
set -e
cd "$(dirname "$0")"
case "$(uname -s)" in
  Darwin) OUT=libvoronoi3d_engine.dylib ;;
  MINGW*|MSYS*|CYGWIN*) OUT=voronoi3d_engine.dll ;;
  *) OUT=libvoronoi3d_engine.so ;;
esac
g++ -O2 -std=c++14 -fPIC -shared -DVOROPP_VERBOSE=1 -o "$OUT" \
  voronoi3d_engine.cpp voro/c_loops.cc voro/cell.cc voro/common.cc voro/container.cc \
  voro/container_prd.cc voro/pre_container.cc voro/unitcell.cc voro/v_base.cc \
  voro/v_compute.cc voro/wall.cc
cp "$OUT" ../"$OUT"
echo "Built $OUT and copied it next to voronoi3d_plugin.py"

#!/bin/sh
# Builds the OpenMP (multi-core) native engine with g++ (Linux/macOS/MinGW). On Windows use build_engine_omp.bat.
set -e
cd "$(dirname "$0")"
case "$(uname -s)" in
  Darwin) OUT=libvoronoi3d_engine_omp.dylib; OMPFLAG="-Xpreprocessor -fopenmp -lomp" ;;
  MINGW*|MSYS*|CYGWIN*) OUT=voronoi3d_engine_omp.dll; OMPFLAG="-fopenmp" ;;
  *) OUT=libvoronoi3d_engine_omp.so; OMPFLAG="-fopenmp" ;;
esac
g++ -O2 -std=c++14 -fPIC -shared $OMPFLAG -DVOROPP_VERBOSE=1 -o "$OUT" \
  voronoi3d_engine_omp.cpp voro/c_loops.cc voro/cell.cc voro/common.cc voro/container.cc \
  voro/container_prd.cc voro/pre_container.cc voro/unitcell.cc voro/v_base.cc \
  voro/v_compute.cc voro/wall.cc
cp "$OUT" ../"$OUT"
echo "Built $OUT and copied it next to voronoi3d_plugin.py"

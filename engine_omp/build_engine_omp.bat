@echo off
rem ---------------------------------------------------------------------------
rem Builds voronoi3d_engine_omp.dll with MSVC (64-bit), OpenMP multi-core variant.
rem
rem 1. Open "x64 Native Tools Command Prompt for VS 20xx" from the Start menu
rem    (installed with Visual Studio or the "Build Tools for Visual Studio").
rem 2. cd to this folder and run:  build_engine_omp.bat
rem
rem The DLL is built in this folder and copied one folder up, next to
rem voronoi3d_plugin.py (the plugin looks in both places).
rem ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

where cl >nul 2>nul
if errorlevel 1 (
    echo ERROR: cl.exe not found.
    echo Open "x64 Native Tools Command Prompt for VS" and run this script again.
    exit /b 1
)
if /i not "%VSCMD_ARG_TGT_ARCH%"=="x64" (
    echo WARNING: target architecture is "%VSCMD_ARG_TGT_ARCH%". ParaView needs x64.
    echo Use the "x64 Native Tools Command Prompt".
)

if not exist build_omp mkdir build_omp

rem /MT links the C++ runtime statically; /openmp needs vcomp140.dll (VC++ Redistributable, installed with Visual Studio).
cl /nologo /O2 /EHsc /MT /LD /std:c++14 /utf-8 /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS ^
   /DVOROPP_VERBOSE=1 /openmp /Fo:build_omp\ /Fe:voronoi3d_engine_omp.dll ^
   voronoi3d_engine_omp.cpp ^
   /Tpvoro\c_loops.cc /Tpvoro\cell.cc /Tpvoro\common.cc /Tpvoro\container.cc ^
   /Tpvoro\container_prd.cc /Tpvoro\pre_container.cc /Tpvoro\unitcell.cc ^
   /Tpvoro\v_base.cc /Tpvoro\v_compute.cc /Tpvoro\wall.cc
if errorlevel 1 (
    echo.
    echo BUILD FAILED. Copy the messages above and send them to me.
    exit /b 1
)

copy /y voronoi3d_engine_omp.dll ..\voronoi3d_engine_omp.dll >nul
echo.
echo Done: voronoi3d_engine_omp.dll built and copied next to voronoi3d_plugin.py.
echo Now restart ParaView (or reload the plugin) and choose Engine = Native C++ OpenMP (multi-core).
endlocal

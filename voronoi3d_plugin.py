"""
Voronoi 3D for ParaView - Python prototype
==========================================

Load it with: Tools -> Manage Plugins -> Load New... -> voronoi3d_plugin.py
(tick "Auto Load" to have it available at every start).

It adds two items. Press "Apply" after creating them; errors are shown in
View -> Output Messages. Both can also be found with Ctrl+Space.

  * "Voronoi3D" (menu Sources -> Alphabetical) - SOURCE, always enabled, needs
                                    no data loaded: it generates N random points
                                    in a box you type in and builds their Voronoi
                                    partition.
  * "Voronoi3D From Input" (menu Filters -> Alphabetical) - FILTER, greyed out
                                    until an object is selected in the pipeline
                                    (ParaView rule for every filter). The object
                                    defines the volume. Seeds:
                                      - "Points from input": the input points are
                                        the Voronoi seeds;
                                      - "Random points": N random points are
                                        generated in the volume, the input points
                                        are ignored (only its bounds are used).
                                    The volume is the input bounds plus a margin,
                                    or a box you type in.

Output: vtkUnstructuredGrid with one VTK_POLYHEDRON cell per seed, clipped
to the box. Cell data: SeedId, Volume, NumFaces, NumCorners (scalars),
SeedPosition and InitialPosition (3-component vectors).

Calculation engines (the "Automatic" choice tries: native -> tess -> SciPy -> VTK):
  * Native C++: Voro++ compiled into voronoi3d_engine.dll (build it with
    engine\\build_engine.bat, see engine\\README.txt). The plugin looks for the DLL
    next to this file or in the engine\\ subfolder. Fastest by far, and also does
    the Lloyd iterations and the vertex merging in C++.
  * VTK Delaunay: uses only vtkDelaunay3D from ParaView plus numpy, no external
    dependencies (ParaView's Python normally does NOT include SciPy).
    The Voronoi diagram is the dual of the Delaunay triangulation.
  * SciPy/Qhull - if ParaView's Python has SciPy.
  * Voro++ through the Python package `tess` (pip install tess) - fast.
  With VTK and SciPy the clipping to the box is exact thanks to the mirroring
  of the points across the 6 walls (7N points: uses more memory).

"Lloyd iterations" option: moves the seeds to the centroids of their cells
(centroidal Voronoi), useful to get more regular cells.

Author: Stefano Bonduà (DICAM, University of Bologna)
License: MIT (see LICENSE). The native engine bundles Voro++ (BSD-style license,
see THIRD_PARTY_LICENSES.md).
SPDX-License-Identifier: MIT
"""
import ctypes
import os
import platform
import sys
import time
import traceback

import numpy as np
from paraview.util.vtkAlgorithm import (
    VTKPythonAlgorithmBase,
    smdomain,
    smproperty,
    smproxy,
)
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.util.numpy_support import (
    numpy_to_vtk,
    numpy_to_vtkIdTypeArray,
    vtk_to_numpy,
)
from vtkmodules.vtkCommonCore import vtkIdList, vtkOutputWindow, vtkPoints, vtkStringArray
from vtkmodules.vtkCommonDataModel import (
    VTK_POLYHEDRON,
    VTK_VERTEX,
    vtkCellArray,
    vtkDataSet,
    vtkPolyData,
    vtkUnstructuredGrid,
)

# ----------------------------------------------------------------------------
# Calculation engine (independent of ParaView: can be replaced by a C++ module)
# ----------------------------------------------------------------------------
ENGINE_AUTO, ENGINE_TESS, ENGINE_SCIPY, ENGINE_VTK, ENGINE_NATIVE, ENGINE_NATIVE_OMP = 0, 1, 2, 3, 4, 5
MODE_INPUT_POINTS, MODE_RANDOM_POINTS, MODE_INPUT_AND_RANDOM = 0, 1, 2
_ENGINE_NAMES = {ENGINE_TESS: "tess", ENGINE_SCIPY: "SciPy", ENGINE_VTK: "VTK Delaunay",
                 ENGINE_AUTO: "automatic"}


# ----------------------------------------------------------------------------
# Native C++ engine (voronoi3d_engine.dll / .so, built from the engine/ folder)
# ----------------------------------------------------------------------------
class _V3DResult(ctypes.Structure):
    _fields_ = [
        ("n_cells", ctypes.c_int64),
        ("n_points", ctypes.c_int64),
        ("stream_len", ctypes.c_int64),
        ("n_skipped", ctypes.c_int64),
        ("n_faces", ctypes.c_int64),
        ("n_pt_ids", ctypes.c_int64),
        ("points", ctypes.POINTER(ctypes.c_double)),
        ("seed_ids", ctypes.POINTER(ctypes.c_int64)),
        ("volumes", ctypes.POINTER(ctypes.c_double)),
        ("num_faces", ctypes.POINTER(ctypes.c_int32)),
        ("cell_offsets", ctypes.POINTER(ctypes.c_int64)),
        ("stream", ctypes.POINTER(ctypes.c_int64)),
        ("seeds", ctypes.POINTER(ctypes.c_double)),
        ("pt_offsets", ctypes.POINTER(ctypes.c_int64)),
        ("pt_ids", ctypes.POINTER(ctypes.c_int64)),
        ("face_offsets", ctypes.POINTER(ctypes.c_int64)),
        ("face_conn", ctypes.POINTER(ctypes.c_int64)),
        ("cell_face_offsets", ctypes.POINTER(ctypes.c_int64)),
        ("error", ctypes.c_char * 512),
    ]


# 3 adds the density weighted Lloyd (v3d_compute_ex), 4 the progress callback (v3d_compute_ex2),
# 5 the per-axis movement flags (v3d_compute_ex3), 6 the convergence history (v3d_compute_ex4),
# 7 the convergence threshold that stops the iterations (v3d_compute_ex5)
_NATIVE_ABI_VERSIONS = (2, 3, 4, 5, 6, 7)
# Convergence of the last native calculation: array (iterations, 3) with the maximum and
# the mean shift of the seeds (in units of the mean cell size) and the CV of the cell volumes
_last_convergence = None
# How the last Lloyd relaxation ended: dict(done, maximum, tol, shift) or None
_last_stop = None
_PROGRESS_FN = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_double)
_native_cache = {"tried": False, "lib": None, "path": None, "message": ""}
_native_omp_cache = {"tried": False, "lib": None, "path": None, "message": ""}


def _native_candidates(omp=False):
    here = os.path.dirname(os.path.abspath(__file__))
    stem = "voronoi3d_engine_omp" if omp else "voronoi3d_engine"
    if sys.platform.startswith("win"):
        names = [stem + ".dll"]
    elif sys.platform == "darwin":
        names = ["lib" + stem + ".dylib"]
    else:
        names = ["lib" + stem + ".so"]
    folders = (here, os.path.join(here, "engine_omp" if omp else "engine"))
    return [os.path.join(d, n) for d in folders for n in names]


def _load_native(omp=False):
    """Load the native engine once; return the ctypes library or None.
    omp=True loads the OpenMP (multi-core) build voronoi3d_engine_omp."""
    cache = _native_omp_cache if omp else _native_cache
    if cache["tried"]:
        return cache["lib"]
    cache["tried"] = True
    problems = []
    for path in _native_candidates(omp):
        if not os.path.isfile(path):
            continue
        try:
            lib = ctypes.CDLL(path)
            lib.v3d_abi_version.restype = ctypes.c_int32
            abi = lib.v3d_abi_version()
            if abi not in _NATIVE_ABI_VERSIONS:
                raise RuntimeError("engine ABI version mismatch (rebuild the DLL)")
            lib.v3d_compute.restype = ctypes.POINTER(_V3DResult)
            lib.v3d_compute.argtypes = [ctypes.c_int64,
                                        ctypes.POINTER(ctypes.c_double),
                                        ctypes.POINTER(ctypes.c_double),
                                        ctypes.c_int32]
            lib.has_weighted = abi >= 3
            if lib.has_weighted:
                lib.v3d_compute_ex.restype = ctypes.POINTER(_V3DResult)
                lib.v3d_compute_ex.argtypes = [ctypes.c_int64,
                                               ctypes.POINTER(ctypes.c_double),
                                               ctypes.POINTER(ctypes.c_double),
                                               ctypes.c_int32,
                                               ctypes.POINTER(ctypes.c_double),
                                               ctypes.c_double, ctypes.c_double]
            lib.has_progress = abi >= 4
            if lib.has_progress:
                lib.v3d_compute_ex2.restype = ctypes.POINTER(_V3DResult)
                lib.v3d_compute_ex2.argtypes = [ctypes.c_int64,
                                                ctypes.POINTER(ctypes.c_double),
                                                ctypes.POINTER(ctypes.c_double),
                                                ctypes.c_int32,
                                                ctypes.POINTER(ctypes.c_double),
                                                ctypes.c_double, ctypes.c_double,
                                                _PROGRESS_FN]
            lib.has_axes = abi >= 5
            if lib.has_axes:
                lib.v3d_compute_ex3.restype = ctypes.POINTER(_V3DResult)
                lib.v3d_compute_ex3.argtypes = [ctypes.c_int64,
                                                ctypes.POINTER(ctypes.c_double),
                                                ctypes.POINTER(ctypes.c_double),
                                                ctypes.c_int32,
                                                ctypes.POINTER(ctypes.c_double),
                                                ctypes.c_double, ctypes.c_double,
                                                _PROGRESS_FN,
                                                ctypes.POINTER(ctypes.c_int32)]
            lib.has_history = abi >= 6
            if lib.has_history:
                lib.v3d_compute_ex4.restype = ctypes.POINTER(_V3DResult)
                lib.v3d_compute_ex4.argtypes = [ctypes.c_int64,
                                                ctypes.POINTER(ctypes.c_double),
                                                ctypes.POINTER(ctypes.c_double),
                                                ctypes.c_int32,
                                                ctypes.POINTER(ctypes.c_double),
                                                ctypes.c_double, ctypes.c_double,
                                                _PROGRESS_FN,
                                                ctypes.POINTER(ctypes.c_int32),
                                                ctypes.POINTER(ctypes.c_double)]
            lib.has_stop = abi >= 7
            if lib.has_stop:
                lib.v3d_compute_ex5.restype = ctypes.POINTER(_V3DResult)
                lib.v3d_compute_ex5.argtypes = [ctypes.c_int64,
                                                ctypes.POINTER(ctypes.c_double),
                                                ctypes.POINTER(ctypes.c_double),
                                                ctypes.c_int32,
                                                ctypes.POINTER(ctypes.c_double),
                                                ctypes.c_double, ctypes.c_double,
                                                _PROGRESS_FN,
                                                ctypes.POINTER(ctypes.c_int32),
                                                ctypes.POINTER(ctypes.c_double),
                                                ctypes.c_double,
                                                ctypes.POINTER(ctypes.c_int32)]
            lib.v3d_free.restype = None
            lib.v3d_free.argtypes = [ctypes.POINTER(_V3DResult)]
            lib.has_threads = hasattr(lib, "v3d_set_num_threads")
            if lib.has_threads:
                lib.v3d_set_num_threads.restype = None
                lib.v3d_set_num_threads.argtypes = [ctypes.c_int32]
                lib.v3d_max_threads.restype = ctypes.c_int32
            if omp and not lib.has_threads:
                raise RuntimeError("this DLL is not the OpenMP build")
            cache.update(lib=lib, path=path)
            print("Voronoi 3D: native %sengine loaded from %s" % ("OpenMP " if omp else "", path))
            return lib
        except Exception as exc:  # wrong architecture, missing runtime, ...
            problems.append("%s: %s" % (path, exc))
    cache["message"] = ("; ".join(problems) if problems else
                        "%s not found next to the plugin (searched: %s)"
                        % ("voronoi3d_engine_omp" if omp else "voronoi3d_engine",
                           ", ".join(_native_candidates(omp))))
    if problems:
        _report_warning("Voronoi 3D: the native engine exists but could not be loaded "
                        "(%s). Using the slower Python engines." % cache["message"])
    elif omp:
        pass
    else:
        print("Voronoi 3D: native engine not found, using the Python engines "
              "(build it with engine\\build_engine.bat for much higher speed)")
    return None


def _array_from(ptr, shape, dtype):
    n = int(np.prod(shape))
    if n == 0:
        return np.zeros(shape, dtype=dtype)
    return np.ctypeslib.as_array(ptr, shape=(n,)).astype(dtype, copy=True).reshape(shape)


def _polyhedra_ok(out, n_cells, offsets, stream):
    """Spot-check that `out` holds the expected polyhedra (guards the fast path)."""
    if out.GetNumberOfCells() != n_cells:
        return False
    for c in sorted({0, n_cells // 2, n_cells - 1}):
        if out.GetCellType(c) != VTK_POLYHEDRON:
            return False
        ids = vtkIdList()
        out.GetFaceStream(c, ids)
        got = [ids.GetId(i) for i in range(ids.GetNumberOfIds())]
        if got != stream[offsets[c]:offsets[c + 1]].tolist():
            return False
    return True


def _fill_polyhedra_fast(out, n_cells, offsets, stream, pt_offsets, pt_ids,
                         face_offsets, face_conn, cell_face_offsets, n_faces):
    """Build all polyhedral cells at once from flat arrays (no Python loop).

    The VTK API for polyhedra changed between versions (ParaView 5.10 = VTK 9.1
    takes face streams as vtkIdTypeArray, VTK >= 9.4 takes vtkCellArray), so both
    are tried and checked; returns False if neither works.
    """
    def ids(a):
        return numpy_to_vtkIdTypeArray(np.ascontiguousarray(a, dtype=np.int64), deep=True)

    types = numpy_to_vtk(np.full(n_cells, VTK_POLYHEDRON, dtype=np.uint8), deep=True)

    def attempt_legacy():
        cells = vtkCellArray()
        cells.SetData(ids(pt_offsets), ids(pt_ids))
        out.SetCells(types, cells, ids(offsets[:-1]), ids(stream))

    def attempt_cellarray_faces():
        cells = vtkCellArray()
        cells.SetData(ids(pt_offsets), ids(pt_ids))
        faces = vtkCellArray()
        faces.SetData(ids(face_offsets), ids(face_conn))
        locations = vtkCellArray()
        locations.SetData(ids(cell_face_offsets), ids(np.arange(n_faces)))
        out.SetPolyhedralCells(types, cells, locations, faces)

    for attempt in (attempt_legacy, attempt_cellarray_faces):
        try:
            out.Initialize()
            attempt()
            if _polyhedra_ok(out, n_cells, offsets, stream):
                return True
        except Exception:
            pass
    out.Initialize()
    return False


class Cancelled(Exception):
    """The user cancelled the calculation (red button in ParaView)."""


def _generate_native(lib, pts, lo, hi, seed_ids, lloyd_iterations, out,
                     density=None, radius=1.0, exponent=1.0, progress=None,
                     move_axes=(True, True, True), shift_tol=0.0):
    """Run the C++ engine and fill `out` (vtkUnstructuredGrid). shift_tol > 0 stops
    the Lloyd iterations when the mean seed shift (in units of the mean cell size)
    falls below it; lloyd_iterations is then the maximum."""
    flat = np.ascontiguousarray(pts, dtype=np.float64)
    box = np.ascontiguousarray(np.ravel(np.column_stack((lo, hi))), dtype=np.float64)
    c_double_p = ctypes.POINTER(ctypes.c_double)
    callback = None
    if lib.has_progress:
        def callback_py(fraction):
            try:
                return 1 if progress is not None and progress(fraction, "") else 0
            except Exception:
                return 0
        callback = _PROGRESS_FN(callback_py)  # keep a reference during the call
    restricted = not all(move_axes)
    if restricted and not lib.has_axes:
        raise RuntimeError("the native engine is too old for the per-axis movement "
                           "option: rebuild it with engine\\build_engine.bat")
    global _last_convergence, _last_stop
    _last_convergence = None
    _last_stop = None
    history = None
    done_it = ctypes.c_int32(int(lloyd_iterations))
    if lib.has_stop:
        dens = (np.ascontiguousarray(density, dtype=np.float64) if density is not None
                else None)
        axes = np.array([1 if m else 0 for m in move_axes], dtype=np.int32)
        history = np.zeros(max(int(lloyd_iterations), 1) * 3, dtype=np.float64)
        res = lib.v3d_compute_ex5(
            len(flat), flat.ctypes.data_as(c_double_p), box.ctypes.data_as(c_double_p),
            int(lloyd_iterations),
            dens.ctypes.data_as(c_double_p) if dens is not None else None,
            float(radius), float(exponent), callback,
            axes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            history.ctypes.data_as(c_double_p), float(shift_tol), ctypes.byref(done_it))
    elif lib.has_history:
        dens = (np.ascontiguousarray(density, dtype=np.float64) if density is not None
                else None)
        axes = np.array([1 if m else 0 for m in move_axes], dtype=np.int32)
        history = np.zeros(max(int(lloyd_iterations), 1) * 3, dtype=np.float64)
        res = lib.v3d_compute_ex4(
            len(flat), flat.ctypes.data_as(c_double_p), box.ctypes.data_as(c_double_p),
            int(lloyd_iterations),
            dens.ctypes.data_as(c_double_p) if dens is not None else None,
            float(radius), float(exponent), callback,
            axes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            history.ctypes.data_as(c_double_p))
    elif lib.has_axes:
        dens = (np.ascontiguousarray(density, dtype=np.float64) if density is not None
                else None)
        axes = np.array([1 if m else 0 for m in move_axes], dtype=np.int32)
        res = lib.v3d_compute_ex3(
            len(flat), flat.ctypes.data_as(c_double_p), box.ctypes.data_as(c_double_p),
            int(lloyd_iterations),
            dens.ctypes.data_as(c_double_p) if dens is not None else None,
            float(radius), float(exponent), callback,
            axes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
    elif lib.has_progress:
        dens = (np.ascontiguousarray(density, dtype=np.float64) if density is not None
                else None)
        res = lib.v3d_compute_ex2(
            len(flat), flat.ctypes.data_as(c_double_p), box.ctypes.data_as(c_double_p),
            int(lloyd_iterations),
            dens.ctypes.data_as(c_double_p) if dens is not None else None,
            float(radius), float(exponent), callback)
    elif density is not None:
        if not lib.has_weighted:
            raise RuntimeError("the native engine is too old for the density weighted "
                               "Lloyd: rebuild it with engine\\build_engine.bat")
        dens = np.ascontiguousarray(density, dtype=np.float64)
        res = lib.v3d_compute_ex(len(flat), flat.ctypes.data_as(c_double_p),
                                 box.ctypes.data_as(c_double_p), int(lloyd_iterations),
                                 dens.ctypes.data_as(c_double_p),
                                 float(radius), float(exponent))
    else:
        res = lib.v3d_compute(len(flat), flat.ctypes.data_as(c_double_p),
                              box.ctypes.data_as(c_double_p), int(lloyd_iterations))
    if not res:
        raise MemoryError("native engine could not allocate its result")
    try:
        r = res.contents
        if r.error:
            message = r.error.decode("utf-8", "replace")
            if "cancelled" in message:
                raise Cancelled()
            raise RuntimeError("native engine: %s" % message)
        n_cells, n_points = int(r.n_cells), int(r.n_points)
        merged = _array_from(r.points, (n_points, 3), np.float64)
        native_ids = _array_from(r.seed_ids, (n_cells,), np.int64)
        volumes = _array_from(r.volumes, (n_cells,), np.float64)
        face_counts = _array_from(r.num_faces, (n_cells,), np.int32)
        offsets = _array_from(r.cell_offsets, (n_cells + 1,), np.int64)
        stream = _array_from(r.stream, (int(r.stream_len),), np.int64)
        n_faces = int(r.n_faces)
        final_positions = _array_from(r.seeds, (n_cells, 3), np.float64)
        pt_offsets = _array_from(r.pt_offsets, (n_cells + 1,), np.int64)
        pt_ids = _array_from(r.pt_ids, (int(r.n_pt_ids),), np.int64)
        face_offsets = _array_from(r.face_offsets, (n_faces + 1,), np.int64)
        face_conn = _array_from(r.face_conn, (int(face_offsets[-1]),), np.int64)
        cell_face_offsets = _array_from(r.cell_face_offsets, (n_cells + 1,), np.int64)
        skipped = int(r.n_skipped)
    finally:
        lib.v3d_free(res)

    if history is not None and int(lloyd_iterations) > 0:
        n_done = max(1, min(int(done_it.value), int(lloyd_iterations)))
        _last_convergence = history.reshape(-1, 3)[:n_done].copy()
        _last_stop = dict(done=n_done, maximum=int(lloyd_iterations), tol=float(shift_tol),
                          shift=float(_last_convergence[-1, 1]))

    if skipped:
        print("Voronoi 3D: %d seeds produced no valid cell and were skipped" % skipped)

    if not _fill_polyhedra_fast(out, n_cells, offsets, stream, pt_offsets, pt_ids,
                                face_offsets, face_conn, cell_face_offsets, n_faces):
        # slow but always available: one InsertNextCell call per cell
        out.Initialize()
        out.Allocate(n_cells)
        for i in range(n_cells):
            seg = stream[offsets[i]:offsets[i + 1]].tolist()
            id_list = vtkIdList()
            id_list.SetNumberOfIds(len(seg))
            for j, v in enumerate(seg):
                id_list.SetId(j, v)
            out.InsertNextCell(VTK_POLYHEDRON, id_list)
    vtk_pts = vtkPoints()
    vtk_pts.SetData(numpy_to_vtk(np.ascontiguousarray(merged), deep=True))
    out.SetPoints(vtk_pts)

    for name, values, dtype in (
        ("SeedId", np.asarray(seed_ids)[native_ids], np.int64),
        ("Volume", volumes, np.float64),
        ("NumFaces", face_counts, np.int32),
    ):
        vtk_array = numpy_to_vtk(np.ascontiguousarray(values.astype(dtype)), deep=True)
        vtk_array.SetName(name)
        out.GetCellData().AddArray(vtk_array)
    return final_positions


def _report_error(prefix="Voronoi 3D"):
    """Show the current exception in ParaView -> View -> Output Messages."""
    msg = "%s: %s" % (prefix, traceback.format_exc())
    print(msg)
    try:
        vtkOutputWindow.GetInstance().DisplayErrorText(msg + "\n")
    except Exception:
        pass


def _report_info(msg):
    """Show an information message (ParaView -> View -> Output Messages)."""
    try:
        vtkOutputWindow.GetInstance().DisplayText(msg + "\n")
    except Exception:
        print(msg)


def _progress_for(algorithm):
    """Progress callback for generate(): drives the ParaView progress bar and
    returns True when the user asked to abort."""
    def report(fraction, text=""):
        try:
            if text:
                algorithm.SetProgressText(text)
            algorithm.UpdateProgress(min(max(float(fraction), 0.0), 1.0))
            return bool(algorithm.GetAbortExecute())
        except Exception:
            return False
    return report


def _report_warning(msg):
    """Show a warning in ParaView -> View -> Output Messages."""
    print(msg)
    try:
        vtkOutputWindow.GetInstance().DisplayWarningText(msg + "\n")
    except Exception:
        pass


def _check_volumes(out, lo, hi, engine_name):
    """The cells must tile the box: warn if they do not fill it or stick out of it."""
    array = out.GetCellData().GetArray("Volume")
    if array is None:
        return
    hint = ("The result is inaccurate: increase 'Min distance from walls' (seeds "
            "on or very close to a wall break the Python engines) or use the Native "
            "C++ engine (build voronoi3d_engine.dll, see engine\\README.txt).")
    box_volume = float(np.prod(hi - lo))
    total = float(vtk_to_numpy(array).sum())
    relative_error = abs(total - box_volume) / box_volume
    if relative_error > 1e-5:
        _report_warning(
            "Voronoi 3D WARNING (%s engine): the cell volumes add up to %.8g but the "
            "box volume is %.8g (relative error %.2g). %s" %
            (engine_name, total, box_volume, relative_error, hint))
    vertices = vtk_to_numpy(out.GetPoints().GetData())
    overshoot = np.maximum(lo - vertices, vertices - hi).max(axis=1)
    outside = int((overshoot > 1e-6 * float(np.max(hi - lo))).sum())
    if outside:
        _report_warning(
            "Voronoi 3D WARNING (%s engine): %d cell vertices lie outside the box "
            "(by up to %.3g). %s" % (engine_name, outside, float(overshoot.max()), hint))


def _import_tess():
    try:
        import tess  # noqa: WPS433

        return tess
    except ImportError:
        return None


def _mirror(points, lo, hi):
    """Original points plus their mirror images across the 6 box walls.

    The bisector between a point and its mirror image is the wall itself, so
    the cell of every original point is bounded and equal to
    (Voronoi cell) intersected with (box).
    """
    blocks = [points]
    for ax in range(3):
        for wall in (lo[ax], hi[ax]):
            m = points.copy()
            m[:, ax] = 2.0 * wall - m[:, ax]
            blocks.append(m)
    return np.vstack(blocks)


def _pack_cells(faces_of, coords):
    """From lists of faces (global Voronoi vertex ids) to per-cell local data."""
    cells = []
    for face_list in faces_of:
        if not face_list:
            cells.append((np.zeros((0, 3)), []))
            continue
        flat = np.concatenate([np.asarray(f, dtype=np.int64) for f in face_list])
        uniq, inv = np.unique(flat, return_inverse=True)
        inv = inv.reshape(-1)
        faces, k = [], 0
        for f in face_list:
            faces.append(inv[k:k + len(f)])
            k += len(f)
        cells.append((coords[uniq], faces))
    return cells


def _tessellate_scipy(points, lo, hi):
    """Voronoi cells clipped to the box [lo, hi] with SciPy/Qhull."""
    from scipy.spatial import Voronoi

    n = len(points)
    vor = Voronoi(_mirror(points, lo, hi))
    faces_of = [[] for _ in range(n)]
    for (a, b), ridge in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 in ridge:
            continue
        if a < n:
            faces_of[a].append(ridge)
        if b < n:
            faces_of[b].append(ridge)
    return _pack_cells(faces_of, vor.vertices)


_TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _tessellate_vtk(points, lo, hi):
    """Voronoi as the dual of the vtkDelaunay3D Delaunay triangulation.

    * Voronoi vertices = circumcenters of the Delaunay tetrahedra;
    * face between seeds a-b = polygon of the circumcenters of the tetrahedra
      around the edge a-b, sorted by angle around that edge.
    """
    from vtkmodules.vtkFiltersCore import vtkDelaunay3D

    # Jitter of ~1e-8 of the box size: breaks degeneracies (regular grids,
    # cospherical points) on which vtkDelaunay3D fails; negligible geometric error.
    extent = float(np.max(hi - lo))
    margin = np.minimum(points - lo, hi - points).min(axis=1)
    amp = np.minimum(1e-8 * extent, 0.25 * margin)[:, None]
    points = points + np.random.default_rng(12345).uniform(-1.0, 1.0, points.shape) * amp

    n = len(points)
    all_pts = np.ascontiguousarray(_mirror(points, lo, hi), dtype=np.float64)
    total = len(all_pts)
    # The mirror images are exactly symmetric to the seeds, i.e. many groups of
    # coplanar/cospherical points. Older VTK (ParaView 5.10 = VTK 9.1) builds
    # inconsistent triangulations from them, so perturb the mirrors as well.
    mirror_rng = np.random.default_rng(54321)
    all_pts[n:] += mirror_rng.uniform(-1.0, 1.0, all_pts[n:].shape) * (1e-8 * extent)

    vtk_pts = vtkPoints()
    vtk_pts.SetData(numpy_to_vtk(all_pts, deep=True))
    poly = vtkPolyData()
    poly.SetPoints(vtk_pts)

    delaunay = vtkDelaunay3D()
    delaunay.SetInputData(poly)
    delaunay.SetTolerance(1e-12)
    delaunay.SetAlpha(0.0)
    delaunay.Update()
    grid = delaunay.GetOutput()

    if grid.GetNumberOfPoints() != total:
        raise RuntimeError(
            "vtkDelaunay3D discarded points (%d -> %d): coincident or "
            "too close points" % (total, grid.GetNumberOfPoints()))
    P = vtk_to_numpy(grid.GetPoints().GetData()).astype(np.float64)
    conn = vtk_to_numpy(grid.GetCells().GetConnectivityArray()).astype(np.int64)
    if grid.GetNumberOfCells() == 0 or conn.size != 4 * grid.GetNumberOfCells():
        raise RuntimeError("vtkDelaunay3D did not produce valid tetrahedra")
    tets = conn.reshape(-1, 4)
    n_tets = len(tets)

    # circumcenters of the tetrahedra
    A = P[tets[:, 0]]
    B = P[tets[:, 1]] - A
    C = P[tets[:, 2]] - A
    D = P[tets[:, 3]] - A
    cross_cd, cross_db, cross_bc = np.cross(C, D), np.cross(D, B), np.cross(B, C)
    det = np.einsum("ij,ij->i", B, cross_cd)
    tiny = 1e-300
    det = np.where(np.abs(det) < tiny, tiny, det)
    num = ((B * B).sum(1)[:, None] * cross_cd
           + (C * C).sum(1)[:, None] * cross_db
           + (D * D).sum(1)[:, None] * cross_bc)
    circumcenters = A + num / (2.0 * det)[:, None]

    # edges (a<b) with at least one seed endpoint, grouped
    edge_a = np.concatenate([tets[:, i] for i, _ in _TET_EDGES])
    edge_b = np.concatenate([tets[:, j] for _, j in _TET_EDGES])
    tet_id = np.tile(np.arange(n_tets, dtype=np.int64), len(_TET_EDGES))
    a = np.minimum(edge_a, edge_b)
    b = np.maximum(edge_a, edge_b)
    keep = a < n
    a, b, tet_id = a[keep], b[keep], tet_id[keep]
    key = a * total + b
    order = np.argsort(key, kind="stable")
    key, tet_id = key[order], tet_id[order]
    cuts = np.flatnonzero(np.diff(key)) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [len(key)]))

    faces_of = [[] for _ in range(n)]
    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    for s, e in zip(starts, ends):
        ring = tet_id[s:e]
        if len(ring) < 3:
            continue
        ia, ib = divmod(int(key[s]), total)
        axis = P[ib] - P[ia]
        norm = np.linalg.norm(axis)
        if norm == 0.0:
            continue
        axis = axis / norm
        helper = ex if abs(axis[0]) < 0.9 else ey
        u = np.cross(axis, helper)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        ring_pts = circumcenters[ring]
        r = ring_pts - ring_pts.mean(axis=0)
        face = ring[np.argsort(np.arctan2(r @ v, r @ u))]
        faces_of[ia].append(face)
        if ib < n:
            faces_of[ib].append(face)
    return _pack_cells(faces_of, circumcenters)


def _tessellate_tess(tess, points, lo, hi):
    """Voronoi cells with Voro++ (package `tess`). Box shifted to [0, L]."""
    size = tuple(float(x) for x in (hi - lo))
    container = tess.Container((points - lo).tolist(), limits=size,
                               periodic=(False, False, False))
    raw = list(container)
    if all(isinstance(c, dict) and "id" in c for c in raw):
        raw.sort(key=lambda c: c["id"])
    cells = []
    for c in raw:
        verts = np.asarray(c["vertices"], dtype=float).reshape(-1, 3) + lo
        faces = [np.asarray(f["vertices"], dtype=np.int64) for f in c["faces"]]
        cells.append((verts, faces))
    return cells


def tessellate(points, lo, hi, engine):
    if engine == ENGINE_VTK:
        return _tessellate_vtk(points, lo, hi)
    if engine == ENGINE_SCIPY:
        return _tessellate_scipy(points, lo, hi)
    if engine == ENGINE_TESS:
        tess = _import_tess()
        if tess is None:
            raise RuntimeError("Package 'tess' is not installed (pip install tess)")
        return _tessellate_tess(tess, points, lo, hi)

    # Automatic: the first available engine that works
    tess = _import_tess()
    if tess is not None:
        try:
            return _tessellate_tess(tess, points, lo, hi)
        except Exception:
            _report_error("Voronoi 3D (tess failed, trying the next engine)")
    try:
        return _tessellate_scipy(points, lo, hi)
    except ImportError:
        pass  # SciPy missing: the normal case in ParaView's Python
    except Exception:
        _report_error("Voronoi 3D (SciPy failed, trying the next engine)")
    return _tessellate_vtk(points, lo, hi)


def cell_geometry(seed, verts, faces):
    """Orient the faces outward and compute the cell volume and centroid."""
    total_volume = 0.0
    centroid_acc = np.zeros(3)
    out_faces = []
    for f in faces:
        f = np.asarray(f)
        if len(f) < 3:
            continue
        p = verts[f]
        normal = np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)  # Newell
        if np.dot(normal, p.mean(axis=0) - seed) < 0.0:
            f = f[::-1]
            p = p[::-1]
        out_faces.append(f)
        a = p[0] - seed
        b = p[1:-1] - seed
        c = p[2:] - seed
        tet_volume = np.einsum("i,ji->j", a, np.cross(b, c)) / 6.0
        tet_centroid = (seed + p[0] + p[1:-1] + p[2:]) / 4.0
        total_volume += tet_volume.sum()
        centroid_acc += (tet_volume[:, None] * tet_centroid).sum(axis=0)
    centroid = (centroid_acc / total_volume if total_volume > 0
                else np.asarray(seed, dtype=float))
    return out_faces, total_volume, centroid


class DensityModel:
    """Density field of the weighted Lloyd iterations.

    Input points with density d_j > 0 are the anchors p_j. The field decays
    linearly with the distance r from the anchors:

        rho(x) = 1 + max_j  d_j * max(0, 1 - |x - p_j| / radius)

    so rho = 1 far from every anchor. The centroid integrals are weighted by
    rho ** exponent (same definition as in the C++ engine).
    """

    def __init__(self, points, density, radius, exponent=1.0):
        density = np.asarray(density, dtype=float)
        keep = density > 0.0
        self.anchors = np.asarray(points, dtype=float)[keep]
        self.anchor_density = density[keep]
        self.radius = float(radius)
        self.exponent = float(exponent)

    @property
    def has_anchors(self):
        return len(self.anchors) > 0

    def rho(self, x):
        x = np.asarray(x, dtype=float).reshape(-1, 3)
        extra = np.zeros(len(x))
        if self.has_anchors:
            step = max(1, int(4e6 // len(self.anchors)))
            for s in range(0, len(x), step):
                dist = np.linalg.norm(x[s:s + step, None, :] - self.anchors[None], axis=2)
                extra[s:s + step] = (self.anchor_density
                                     * np.clip(1.0 - dist / self.radius, 0.0, None)).max(axis=1)
        return 1.0 + extra

    def weight(self, x):
        return self.rho(x) ** self.exponent


_WEIGHT_SUBDIVISIONS = 1  # same as kSubdivisions in the C++ engine


def _subdivide_triangles(tri, levels):
    """Split triangles (T, 3, 3) into 4**levels triangles each (edge midpoints)."""
    for _ in range(levels):
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        ab, bc, ca = (a + b) / 2.0, (b + c) / 2.0, (c + a) / 2.0
        tri = np.concatenate((np.stack((a, ab, ca), axis=1), np.stack((ab, b, bc), axis=1),
                              np.stack((ca, bc, c), axis=1), np.stack((ab, bc, ca), axis=1)))
    return tri


def _weighted_centroids(points, cells, model, fixed):
    """Density weighted centroid of every free cell.

    The cell is split into tetrahedra: the seed plus a fan of triangles around the
    centre of every face (independent of the vertex a face starts with, so mirror
    symmetric cells give mirror symmetric centroids), subdivided
    _WEIGHT_SUBDIVISIONS times; the weight is sampled at each tetrahedron centroid.
    """
    owners, volumes, centres = [], [], []
    for i, (verts, faces) in enumerate(cells):
        if fixed[i] or len(verts) == 0:
            continue
        seed = points[i]
        verts = np.asarray(verts)
        tris = []
        for f in faces:
            if len(f) < 3:
                continue
            p = verts[np.asarray(f)] - seed
            centre = p.mean(axis=0)
            tris.append(np.stack((np.broadcast_to(centre, p.shape), p, np.roll(p, -1, axis=0)),
                                 axis=1))
        if not tris:
            continue
        tri = _subdivide_triangles(np.concatenate(tris), _WEIGHT_SUBDIVISIONS)
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        vol = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0
        owners.append(np.full(len(vol), i))
        volumes.append(vol)
        centres.append(seed + (a + b + c) / 4.0)
    result = points.copy()
    if not owners:
        return result
    owners, volumes, centres = np.concatenate(owners), np.concatenate(volumes), np.vstack(centres)
    w = volumes * model.weight(centres)
    n = len(points)
    wsum = np.bincount(owners, weights=w, minlength=n)
    for a in range(3):
        acc = np.bincount(owners, weights=w * centres[:, a], minlength=n)
        ok = wsum > 0
        result[ok, a] = acc[ok] / wsum[ok]
    return result


def lloyd(points, lo, hi, engine, iterations, model=None, fixed=None, progress=None,
          move_axes=(True, True, True), shift_tol=0.0):
    """Lloyd relaxation. With `model` (DensityModel) the centroids are density
    weighted and the seeds flagged in `fixed` (bool array) do not move. With
    shift_tol > 0 the iterations stop when the mean seed shift (in units of the mean
    cell size h = (V / N)^(1/3)) falls below it; `iterations` is then the maximum."""
    global _last_stop
    _last_stop = None
    h_ref = (float(np.prod(np.asarray(hi) - np.asarray(lo))) / max(len(points), 1)) ** (1 / 3)
    for it in range(int(iterations)):
        if progress is not None and progress(0.9 * it / max(int(iterations), 1),
                                             "Lloyd iteration %d of %d" % (it + 1, iterations)):
            raise Cancelled()
        cells = tessellate(points, lo, hi, engine)
        if model is not None and model.has_anchors:
            new_points = _weighted_centroids(points, cells, model, fixed)
        else:
            new_points = points.copy()
            for i, (verts, faces) in enumerate(cells):
                _, volume, centroid = cell_geometry(points[i], verts, faces)
                if volume > 0:
                    new_points[i] = centroid
        for a in range(3):
            if not move_axes[a]:  # frozen axis: the seeds keep their coordinate
                new_points[:, a] = points[:, a]
        shifts = np.linalg.norm(new_points - points, axis=1) / h_ref
        if fixed is not None:
            shifts = shifts[~np.asarray(fixed, dtype=bool)]
        mean_shift = float(shifts.mean()) if len(shifts) else 0.0
        points = new_points
        _last_stop = dict(done=it + 1, maximum=int(iterations), tol=float(shift_tol),
                          shift=mean_shift)
        if shift_tol > 0.0 and len(shifts) and mean_shift < shift_tol:
            break
    return points


def build_grid(points, seed_ids, cells, out):
    """Fill `out` (vtkUnstructuredGrid) with polyhedral cells."""
    geometry = [cell_geometry(points[i], v, f) for i, (v, f) in enumerate(cells)]

    # merge coincident vertices shared by adjacent cells
    all_verts = [v for v, _ in cells if len(v)]
    if not all_verts:
        raise ValueError("No Voronoi cell was computed")
    extent = float(np.max(np.ptp(points, axis=0))) or 1.0
    tol = 1e-7 * extent
    stacked = np.vstack(all_verts)
    key = np.round(stacked / tol).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    merged = stacked[first]

    vtk_pts = vtkPoints()
    vtk_pts.SetData(numpy_to_vtk(np.ascontiguousarray(merged), deep=True))
    out.SetPoints(vtk_pts)
    out.Allocate(len(cells))

    offset = 0
    volumes, face_counts, seeds, final_positions = [], [], [], []
    for i, (verts, _faces) in enumerate(cells):
        oriented_faces, volume, _ = geometry[i]
        global_ids = inverse[offset:offset + len(verts)]
        offset += len(verts)

        clean = []
        for f in oriented_faces:  # after merging a vertex may appear twice
            face_ids = global_ids[f]
            face_ids = face_ids[np.concatenate(([True], face_ids[1:] != face_ids[:-1]))]
            if len(face_ids) > 1 and face_ids[0] == face_ids[-1]:
                face_ids = face_ids[:-1]
            if len(face_ids) >= 3:
                clean.append(face_ids)
        if len(clean) < 4:
            continue

        stream = vtkIdList()
        stream.InsertNextId(len(clean))
        for face_ids in clean:
            stream.InsertNextId(len(face_ids))
            for v in face_ids:
                stream.InsertNextId(int(v))
        out.InsertNextCell(VTK_POLYHEDRON, stream)
        volumes.append(volume)
        face_counts.append(len(clean))
        seeds.append(seed_ids[i])
        final_positions.append(points[i])

    for name, values, dtype in (
        ("SeedId", np.asarray(seeds), np.int64),
        ("Volume", np.asarray(volumes), np.float64),
        ("NumFaces", np.asarray(face_counts), np.int32),
    ):
        vtk_array = numpy_to_vtk(values.astype(dtype), deep=True)
        vtk_array.SetName(name)
        out.GetCellData().AddArray(vtk_array)
    return np.asarray(final_positions, dtype=np.float64).reshape(-1, 3)


def _named_array(name, values, dtype):
    array = numpy_to_vtk(np.ascontiguousarray(np.asarray(values).astype(dtype)), deep=True)
    array.SetName(name)
    return array


def _add_seed_data(out, input_points, final_positions, seeds_out,
                   input_density=None, model=None):
    """Seed coordinates for export.

    * cell data of `out`: SeedPosition (seed after the Lloyd iterations) and
      InitialPosition (the seed as given / generated, before Lloyd);
    * `seeds_out` (optional second output): one vertex per seed at its final
      position, with the same information as point data.
    """
    seed_id = vtk_to_numpy(out.GetCellData().GetArray("SeedId")).astype(np.int64)
    initial = np.asarray(input_points, dtype=np.float64)[seed_id]
    final = np.asarray(final_positions, dtype=np.float64)
    out.GetCellData().AddArray(_named_array("SeedPosition", final, np.float64))
    out.GetCellData().AddArray(_named_array("InitialPosition", initial, np.float64))
    extra = []  # (name, values, dtype): density information
    if input_density is not None and model is not None:
        extra = [("InputDensity", np.asarray(input_density, dtype=float)[seed_id], np.float64),
                 ("LocalDensity", model.rho(final), np.float64)]
        for name, values, dtype in extra:
            out.GetCellData().AddArray(_named_array(name, values, dtype))
    # number of corners (distinct vertices) of every cell = connectivity size
    offsets = vtk_to_numpy(out.GetCells().GetOffsetsArray()).astype(np.int64)
    out.GetCellData().AddArray(_named_array("NumCorners", np.diff(offsets), np.int32))
    if seeds_out is None:
        return
    n = len(final)
    seeds_out.Initialize()
    vtk_pts = vtkPoints()
    vtk_pts.SetData(numpy_to_vtk(np.ascontiguousarray(final), deep=True))
    seeds_out.SetPoints(vtk_pts)
    vertices = vtkCellArray()
    vertices.SetData(numpy_to_vtkIdTypeArray(np.arange(n + 1, dtype=np.int64), deep=True),
                     numpy_to_vtkIdTypeArray(np.arange(n, dtype=np.int64), deep=True))
    seeds_out.SetCells(VTK_VERTEX, vertices)
    point_data = seeds_out.GetPointData()
    point_data.AddArray(_named_array("SeedId", seed_id, np.int64))
    point_data.AddArray(_named_array("InitialPosition", initial, np.float64))
    for name, values, dtype in extra:
        point_data.AddArray(_named_array(name, values, dtype))
    for name, dtype in (("Volume", np.float64), ("NumFaces", np.int32),
                        ("NumCorners", np.int32)):
        point_data.AddArray(_named_array(
            name, vtk_to_numpy(out.GetCellData().GetArray(name)), dtype))


_machine_cache = {}


def _machine_info():
    """Description of the computer (no host or user name): OS, CPU, cores, RAM,
    Python and VTK versions. Cached; every item falls back to 'unknown'."""
    if _machine_cache:
        return _machine_cache
    cpu, ram_gb = "", 0.0
    try:
        if sys.platform.startswith("win"):
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0]

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            ("total_phys", ctypes.c_ulonglong),
                            ("avail_phys", ctypes.c_ulonglong),
                            ("total_page", ctypes.c_ulonglong),
                            ("avail_page", ctypes.c_ulonglong),
                            ("total_virtual", ctypes.c_ulonglong),
                            ("avail_virtual", ctypes.c_ulonglong),
                            ("avail_ext", ctypes.c_ulonglong)]
            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            ram_gb = status.total_phys / 1024.0 ** 3
        elif sys.platform == "darwin":
            import subprocess
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            ram_gb = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode()) / 1024.0 ** 3
        else:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        cpu = line.split(":", 1)[1].strip()
                        break
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        ram_gb = int(line.split()[1]) / 1024.0 ** 2
                        break
    except Exception:
        pass
    try:
        from vtkmodules.vtkCommonCore import vtkVersion
        vtk_version = vtkVersion.GetVTKVersion()
    except Exception:
        vtk_version = "unknown"
    _machine_cache.update(
        OS="%s %s (%s)" % (platform.system(), platform.release(), platform.machine()),
        CPU=cpu or platform.processor() or "unknown",
        LogicalCores=int(os.cpu_count() or 0),
        RAM_GB=round(ram_gb, 1),
        Python=platform.python_version(),
        VTK=vtk_version)
    return _machine_cache


def _report_convergence(conv, field_data, threshold=0.01):
    """Report the convergence of the Lloyd iterations (native engine). `conv` has one row
    per iteration: maximum and mean shift of the free seeds, in units of the mean cell size
    h = (V / N)^(1/3), and CV of the cell volumes at the start of the iteration. The
    relaxation is considered stable when the mean shift falls below `threshold` (1 % of h)."""
    n_it = len(conv)
    below = np.nonzero(conv[:, 1] < threshold)[0]
    stable = ("mean shift below %.0f %% of h from iteration %d" % (100 * threshold, below[0] + 1)
              if len(below) else "mean shift still above %.0f %% of h" % (100 * threshold))
    _report_info("Voronoi 3D: convergence after %d iterations: mean seed shift %.4f h, "
                 "maximum %.4f h, volume CV %.3f (%s)"
                 % (n_it, conv[-1, 1], conv[-1, 0], conv[-1, 2], stable))
    field_data.AddArray(_named_array("Lloyd_MaxShift_h", conv[:, 0], np.float64))
    field_data.AddArray(_named_array("Lloyd_MeanShift_h", conv[:, 1], np.float64))
    field_data.AddArray(_named_array("Lloyd_VolumeCV", conv[:, 2], np.float64))


def _report_lloyd_stop(stop, field_data):
    """Say why the Lloyd iterations ended: convergence threshold reached or maximum number
    of iterations performed."""
    done, maximum, tol, shift = stop["done"], stop["maximum"], stop["tol"], stop["shift"]
    if tol > 0.0 and shift < tol:
        _report_info("Voronoi 3D: Lloyd iterations stopped at iteration %d of at most %d: the "
                     "mean seed shift (%.4f h) is below the convergence threshold (%.4f h)"
                     % (done, maximum, shift, tol))
    elif tol > 0.0:
        _report_info("Voronoi 3D: Lloyd iterations stopped at the maximum number of "
                     "iterations (%d): the mean seed shift (%.4f h) is still above the "
                     "convergence threshold (%.4f h)" % (done, shift, tol))
    else:
        _report_info("Voronoi 3D: %d Lloyd iterations performed, mean seed shift %.4f h "
                     "(convergence threshold disabled)" % (done, shift))
    field_data.AddArray(_named_array("Lloyd_IterationsDone", [done], np.int32))
    field_data.AddArray(_named_array("Lloyd_ConvergenceThreshold_h", [tol], np.float64))
    field_data.AddArray(_named_array("Lloyd_MeanShiftFinal_h", [shift], np.float64))


def _report_timing(out, started, engine_name, lloyd_iterations, model):
    """Print the total calculation time and store it in the field data of `out`
    (ComputationTime_s), so it can be quoted in a publication."""
    elapsed = time.perf_counter() - started
    n_cells = out.GetNumberOfCells()
    kind = "weighted " if model is not None and model.has_anchors else ""
    _report_info(
        "Voronoi 3D: calculation time %.3f s  (%d cells, %d %sLloyd iterations, "
        "engine: %s)" % (elapsed, n_cells, int(lloyd_iterations), kind, engine_name))
    machine = _machine_info()
    _report_info("Voronoi 3D: machine: %s | CPU: %s | %d logical cores | RAM %.1f GB | "
                 "Python %s | VTK %s"
                 % (machine["OS"], machine["CPU"], machine["LogicalCores"], machine["RAM_GB"],
                    machine["Python"], machine["VTK"]))
    field_data = out.GetFieldData()
    global _last_convergence
    global _last_stop
    conv, _last_convergence = _last_convergence, None
    stop, _last_stop = _last_stop, None
    if conv is not None and len(conv):
        _report_convergence(conv, field_data)
    if stop is not None:
        _report_lloyd_stop(stop, field_data)
    field_data.AddArray(_named_array("ComputationTime_s", [elapsed], np.float64))
    field_data.AddArray(_named_array("NumLloydIterations", [int(lloyd_iterations)], np.int32))
    engine_array = vtkStringArray()
    engine_array.SetName("Engine")
    engine_array.InsertNextValue(str(engine_name))
    field_data.AddArray(engine_array)
    for key in ("OS", "CPU", "Python", "VTK"):
        array = vtkStringArray()
        array.SetName("Machine_" + key)
        array.InsertNextValue(str(machine[key]))
        field_data.AddArray(array)
    field_data.AddArray(_named_array("Machine_LogicalCores", [machine["LogicalCores"]], np.int32))
    field_data.AddArray(_named_array("Machine_RAM_GB", [machine["RAM_GB"]], np.float64))


def remove_close_points(points, tolerance, priority=None):
    """Indices of the points to keep after dropping every point that lies within
    `tolerance` of a point that is kept. Of two close points the one with the
    higher `priority` (e.g. density > 0) survives, otherwise the first one.

    Very close or coincident seeds make the Voronoi engines unstable (Voro++
    can abort), so they are removed before the tessellation.
    """
    n = len(points)
    keep = np.ones(n, dtype=bool)
    if n < 2 or tolerance <= 0.0:
        return np.arange(n)
    order = np.argsort(points[:, 0], kind="stable")
    xs = points[order, 0]
    pairs = []
    k = 1
    while k < n:
        close = np.nonzero(xs[k:] - xs[:-k] <= tolerance)[0]
        if len(close) == 0:
            break
        a, b = order[close], order[close + k]
        near = np.linalg.norm(points[a] - points[b], axis=1) <= tolerance
        pairs.extend(zip(a[near].tolist(), b[near].tolist()))
        k += 1
    pairs.sort()
    for a, b in pairs:
        if not (keep[a] and keep[b]):
            continue
        if a > b:
            a, b = b, a
        if priority is not None and priority[b] and not priority[a]:
            keep[a] = False
        else:
            keep[b] = False
    return np.nonzero(keep)[0]


def generate(points, bounds, engine, lloyd_iterations, out, min_distance=0.0,
             seeds_out=None, density=None, density_radius=0.2, density_exponent=1.0,
             min_separation=1e-6, progress=None, move_axes=(True, True, True), threads=0,
             convergence_threshold=0.0):
    """Full pipeline: points -> (Lloyd) -> Voronoi -> vtkUnstructuredGrid.

    min_distance: seeds closer than this fraction of the box size (per axis) to
    a wall are ignored, because seeds on or next to a wall give degenerate cells.
    seeds_out: optional vtkUnstructuredGrid filled with the final seed points.
    density: optional array (one value per point). Points with density > 0 are
    fixed during the Lloyd iterations and make the cells around them smaller
    (weighted centroidal Voronoi tessellation); points with density 0 move.
    density_radius: reach of the refinement, fraction of the largest box size.
    density_exponent: the centroid weight is (1 + local density) ** exponent.
    move_axes: (x, y, z) booleans: the seeds only move along the checked axes during
    the Lloyd iterations (points on a plane stay on it if that axis is unchecked).
    convergence_threshold: the Lloyd iterations stop when the mean shift of the free seeds,
    in units of the mean cell size, falls below this value (0 = always run all
    `lloyd_iterations`, which is then a maximum when the threshold is set).
    progress: optional callable progress(fraction, text) -> bool (True = abort),
    see _progress_for(). The total calculation time is reported at the end and
    stored in the field data of `out` (ComputationTime_s).
    min_separation: points closer than this (fraction of the largest box side)
    to another point are ignored: coincident or almost coincident seeds would
    make Voro++ and the other engines fail.
    """
    started = time.perf_counter()
    move_axes = tuple(bool(m) for m in move_axes)
    if lloyd_iterations > 0 and not any(move_axes):
        _report_warning("Voronoi 3D: movement is disabled along all axes: the Lloyd "
                        "iterations do not move any point")
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    b = np.asarray(bounds, dtype=float)
    lo, hi = b[0::2], b[1::2]
    if np.any(hi <= lo):
        raise ValueError("Invalid box: min must be smaller than max on every axis")

    clearance = _clearance(lo, hi, min_distance)
    inside = np.all((points > lo + clearance) & (points < hi - clearance), axis=1)
    seed_ids = np.nonzero(inside)[0]
    pts = points[inside]
    if density is not None:
        density = np.asarray(density, dtype=float).reshape(-1)
        if len(density) != len(points):
            raise ValueError("The density array must have one value per point")
        if not np.all(np.isfinite(density)) or np.any(density < 0.0):
            raise ValueError("Density values must be finite and >= 0")
    if len(pts) < len(points):
        _report_warning(
            "Voronoi 3D: %d of %d points were ignored because they are outside the box "
            "or closer than %.3g (fraction of the box size) to a wall"
            % (len(points) - len(pts), len(points), min_distance))
    if len(pts) < 2:
        raise ValueError("At least 2 points inside the box are needed")

    # identical points would give degenerate (empty) cells: keep one of each
    # (a point with density > 0 wins over a free one, otherwise the first)
    perm = np.arange(len(pts))
    if density is not None:
        perm = np.argsort(~(density[seed_ids] > 0.0), kind="stable")
    _, first = np.unique(pts[perm], axis=0, return_index=True)
    first = perm[first]
    if len(first) < len(pts):
        first.sort()
        print("Voronoi 3D: %d duplicate points were ignored" % (len(pts) - len(first)))
        pts, seed_ids = pts[first], seed_ids[first]
        if len(pts) < 2:
            raise ValueError("At least 2 distinct points inside the box are needed")

    separation = float(max(min_separation, 0.0)) * float(np.max(hi - lo))
    if separation > 0.0:
        priority = (density[seed_ids] > 0.0) if density is not None else None
        kept = remove_close_points(pts, separation, priority)
        if len(kept) < len(pts):
            _report_warning(
                "Voronoi 3D: %d points were ignored because they are closer than %.3g "
                "(%.3g of the box size) to another point"
                % (len(pts) - len(kept), separation, min_separation))
            pts, seed_ids = pts[kept], seed_ids[kept]
            if len(pts) < 2:
                raise ValueError("At least 2 distinct points inside the box are needed")

    model, pts_density = None, None
    if density is not None:
        pts_density = density[seed_ids]
        radius = float(density_radius) * float(np.max(hi - lo))
        if not radius > 0.0 or not density_exponent > 0.0:
            raise ValueError("Density radius and exponent must be > 0")
        model = DensityModel(pts, pts_density, radius, density_exponent)
        if not model.has_anchors:
            _report_warning("Voronoi 3D: no point has density > 0: plain Lloyd iterations")
        else:
            print("Voronoi 3D: weighted Lloyd with %d fixed points (density > 0), "
                  "radius %.4g, exponent %.3g" % (len(model.anchors), radius, density_exponent))

    # an engine whose Python package is missing is replaced by the Automatic choice
    if engine == ENGINE_TESS and _import_tess() is None:
        _report_warning("Voronoi 3D: the Python package 'tess' is not installed in "
                        "ParaView's Python: using the Automatic engine instead "
                        "(native C++ if the DLL is built, otherwise SciPy/VTK)")
        engine = ENGINE_AUTO
    elif engine == ENGINE_SCIPY:
        try:
            import scipy.spatial  # noqa: F401
        except ImportError:
            _report_warning("Voronoi 3D: SciPy is not installed in ParaView's Python: "
                            "using the Automatic engine instead")
            engine = ENGINE_AUTO

    if engine in (ENGINE_AUTO, ENGINE_NATIVE, ENGINE_NATIVE_OMP):
        omp = engine == ENGINE_NATIVE_OMP
        lib = _load_native(omp)
        if lib is not None:
            engine_label = "Native C++"
            if getattr(lib, "has_threads", False):
                lib.v3d_set_num_threads(int(threads))
                engine_label = "Native C++ OpenMP (%d threads)" % (
                    int(threads) if int(threads) > 0 else lib.v3d_max_threads())
            try:
                final_positions = _generate_native(
                    lib, pts, lo, hi, seed_ids, lloyd_iterations, out,
                    pts_density, model.radius if model else 1.0,
                    model.exponent if model else 1.0, progress, move_axes,
                    float(convergence_threshold))
                _check_volumes(out, lo, hi, "native")
                _add_seed_data(out, points, final_positions, seeds_out, density, model)
                _report_timing(out, started, engine_label, lloyd_iterations, model)
                return
            except Cancelled:
                raise
            except Exception:
                if engine in (ENGINE_NATIVE, ENGINE_NATIVE_OMP):
                    raise
                _report_error("Voronoi 3D (native engine failed, trying the next engine)")
                out.Initialize()
        elif engine in (ENGINE_NATIVE, ENGINE_NATIVE_OMP):
            raise RuntimeError("Native engine not available: %s"
                               % (_native_omp_cache if omp else _native_cache)["message"])

    if lloyd_iterations > 0:
        fixed = (pts_density > 0.0) if pts_density is not None else None
        pts = lloyd(pts, lo, hi, engine, lloyd_iterations, model, fixed, progress,
                    move_axes, float(convergence_threshold))
    if progress is not None and progress(0.92, "Building the cells"):
        raise Cancelled()
    cells = tessellate(pts, lo, hi, engine)
    final_positions = build_grid(pts, seed_ids, cells, out)
    _check_volumes(out, lo, hi, _ENGINE_NAMES.get(engine, "selected"))
    _add_seed_data(out, points, final_positions, seeds_out, density, model)
    _report_timing(out, started, _ENGINE_NAMES.get(engine, "selected"), lloyd_iterations,
                   model)


def _clearance(lo, hi, min_distance):
    """Minimum distance from the walls, per axis, as a length."""
    return (hi - lo) * float(min(max(min_distance, 0.0), 0.49))


def random_points(num_points, rng_seed, bounds, min_distance=0.0):
    """Uniform random points in the box `bounds` (xmin xmax ymin ymax zmin zmax),
    kept at least `min_distance` (fraction of the box size) away from every wall."""
    lo = np.array(bounds[0::2], dtype=float)
    hi = np.array(bounds[1::2], dtype=float)
    clearance = _clearance(lo, hi, min_distance)
    lo_safe, hi_safe = lo + clearance, hi - clearance
    rng = np.random.default_rng(int(rng_seed))
    pts = lo_safe + rng.random((int(num_points), 3)) * (hi_safe - lo_safe)
    # rng.random() is in [0, 1): make sure no coordinate can touch the upper limit
    return np.clip(pts, lo_safe, np.nextafter(hi_safe, lo_safe))


def box_from_points(points, margin_percent, min_distance=0.0):
    """Bounding box of `points` enlarged by margin_percent on every side, and at
    least enough for every point to be `min_distance` (fraction of the final box
    size) away from the walls."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    size = hi - lo
    d = float(min(max(min_distance, 0.0), 0.49))
    # (1 + 1e-6): the extreme points must be strictly farther than the clearance
    pad = np.maximum(size * margin_percent / 100.0,
                     size * d / (1.0 - 2.0 * d) * (1.0 + 1e-6))
    pad[hi == lo] = 1e-3  # input degenerate (flat) along an axis
    return np.ravel(np.column_stack((lo - pad, hi + pad)))


def _doc(text):
    return "<Documentation>%s</Documentation>" % text


_ENGINE_XML = """
<EnumerationDomain name="enum">
  <Entry value="0" text="Automatic (native, tess, SciPy, VTK)"/>
  <Entry value="1" text="Voro++ via 'tess' package (needs pip install)"/>
  <Entry value="2" text="SciPy / Qhull (needs SciPy)"/>
  <Entry value="3" text="VTK Delaunay (no dependencies)"/>
  <Entry value="4" text="Native C++ (Voro++ DLL, serial)"/>
  <Entry value="5" text="Native C++ OpenMP (Voro++ DLL, multi-core)"/>
</EnumerationDomain>
"""

_MODE_XML = """
<EnumerationDomain name="enum">
  <Entry value="0" text="Points from input"/>
  <Entry value="1" text="Random points"/>
  <Entry value="2" text="Input points + random points"/>
</EnumerationDomain>
"""


# ----------------------------------------------------------------------------
# Source: random points (no input object needed)
# ----------------------------------------------------------------------------
@smproxy.source(name="Voronoi3D", label="Voronoi3D")
@smproperty.xml('<OutputPort name="Voronoi cells" index="0"/>')
@smproperty.xml('<OutputPort name="Seed points" index="1"/>')
class Voronoi3D(VTKPythonAlgorithmBase):
    def __init__(self):
        super().__init__(nInputPorts=0, nOutputPorts=2,
                         outputType="vtkUnstructuredGrid")
        self._num_points = 1000
        self._rng_seed = 0
        self._box = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        self._min_distance = 0.01
        self._engine = ENGINE_AUTO
        self._threads = 0
        self._lloyd = 0
        self._conv_tol = 0.005
        self._min_separation = 1e-6
        self._move_axes = [True, True, True]

    @smproperty.intvector(name="NumPoints", label="Number of points",
                          default_values=1000)
    @smdomain.xml(_doc("Number of random points (= number of Voronoi cells)."))
    def SetNumPoints(self, v):
        self._num_points = max(2, int(v))
        self.Modified()

    @smproperty.intvector(name="RandomSeed", label="Random seed",
                          default_values=0)
    @smdomain.xml(_doc("Same seed = same points. Change it for a different set."))
    def SetRandomSeed(self, v):
        self._rng_seed = int(v)
        self.Modified()

    @smproperty.doublevector(name="Box",
                             label="Box (xmin xmax ymin ymax zmin zmax)",
                             default_values=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                             number_of_elements=6)
    @smdomain.xml(_doc("Volume to discretize: the random points are generated "
                       "in it and the cells are clipped to it."))
    def SetBox(self, xmin, xmax, ymin, ymax, zmin, zmax):
        self._box = [xmin, xmax, ymin, ymax, zmin, zmax]
        self.Modified()

    @smproperty.doublevector(name="MinDistance",
                             label="Min distance from walls (fraction of box size)",
                             default_values=0.01)
    @smdomain.xml(_doc("Seeds are kept at least this far from every wall of the box, "
                       "as a fraction of the box size (0.01 = 1%). Random points are "
                       "generated outside this margin; input points inside it are "
                       "ignored. Seeds on or next to a wall give degenerate cells."))
    def SetMinDistance(self, v):
        self._min_distance = min(max(float(v), 0.0), 0.49)
        self.Modified()

    @smproperty.intvector(name="LloydIterations", label="Lloyd iterations",
                          default_values=0)
    @smdomain.xml(_doc("0 = plain Voronoi. Each iteration moves the seeds to the "
                       "cell centroids (more regular cells). With a convergence "
                       "threshold this is the maximum number of iterations."))
    def SetLloydIterations(self, v):
        self._lloyd = max(0, int(v))
        self.Modified()

    @smproperty.doublevector(name="ConvergenceThreshold",
                             label="Convergence threshold (mean seed shift / cell size)",
                             default_values=0.005)
    @smdomain.xml(_doc("The Lloyd iterations stop earlier when the mean displacement of "
                       "the free seeds, divided by the mean cell size, falls below this "
                       "value (0.005 = 0.5 %). The number of Lloyd iterations is then "
                       "the maximum. 0 = always perform all iterations. The Output "
                       "Messages report which condition ended the calculation."))
    def SetConvergenceThreshold(self, v):
        self._conv_tol = max(0.0, float(v))
        self.Modified()

    @smproperty.doublevector(name="MinSeparation",
                             label="Min distance between points (fraction of box size)",
                             default_values=1e-6)
    @smdomain.xml(_doc("Points closer than this to another point are ignored (the one "
                       "with density > 0 is kept). Coincident or almost coincident "
                       "points make Voro++ unstable. 0 disables the check."))
    def SetMinSeparation(self, v):
        self._min_separation = max(0.0, float(v))
        self.Modified()

    @smproperty.intvector(name="MoveX", label="Move seeds along X", default_values=1)
    @smdomain.xml('<BooleanDomain name="bool"/>'
                  + _doc("Lloyd iterations: the seeds may move along X. Uncheck the axes "
                         "perpendicular to planar (2D) or layered (2.5D) point sets so "
                         "that rounding errors do not move the points out of their planes."))
    def SetMoveX(self, v):
        self._move_axes[0] = bool(int(v))
        self.Modified()

    @smproperty.intvector(name="MoveY", label="Move seeds along Y", default_values=1)
    @smdomain.xml('<BooleanDomain name="bool"/>')
    def SetMoveY(self, v):
        self._move_axes[1] = bool(int(v))
        self.Modified()

    @smproperty.intvector(name="MoveZ", label="Move seeds along Z", default_values=1)
    @smdomain.xml('<BooleanDomain name="bool"/>')
    def SetMoveZ(self, v):
        self._move_axes[2] = bool(int(v))
        self.Modified()

    @smproperty.intvector(name="Engine", label="Calculation engine",
                          default_values=0)
    @smdomain.xml(_ENGINE_XML)
    def SetEngine(self, v):
        self._engine = int(v)
        self.Modified()

    @smproperty.intvector(name="NumThreads", label="Number of threads (OpenMP engine, 0 = all)",
                          default_values=0, panel_visibility="advanced")
    @smdomain.xml('<IntRangeDomain name="range" min="0" max="1024"/>')
    def SetNumThreads(self, v):
        self._threads = int(v)
        self.Modified()

    def RequestData(self, request, inInfo, outInfo):
        try:
            out = vtkUnstructuredGrid.GetData(outInfo, 0)
            seeds_out = vtkUnstructuredGrid.GetData(outInfo, 1)
            pts = random_points(self._num_points, self._rng_seed, self._box,
                                self._min_distance)
            generate(pts, self._box, self._engine, self._lloyd, out,
                     self._min_distance, seeds_out,
                     min_separation=self._min_separation,
                     progress=_progress_for(self), move_axes=self._move_axes,
                     threads=self._threads, convergence_threshold=self._conv_tol)
        except Cancelled:
            _report_warning("Voronoi 3D: calculation cancelled")
            out.Initialize()
            return 1
        except Exception:
            _report_error()
            return 0
        return 1


# ----------------------------------------------------------------------------
# Filter: seeds from an input dataset, or random points in its volume
# ----------------------------------------------------------------------------
@smproxy.filter(name="Voronoi3DFromInput", label="Voronoi3D From Input")
@smproperty.input(name="Input", port_index=0)
@smdomain.datatype(dataTypes=["vtkDataSet"], composite_data_supported=False)
@smdomain.xml('<InputArrayDomain name="inputs_array" attribute_type="point" '
              'number_of_components="1"/>')
@smproperty.xml('<OutputPort name="Voronoi cells" index="0"/>')
@smproperty.xml('<OutputPort name="Seed points" index="1"/>')
class Voronoi3DFromInput(VTKPythonAlgorithmBase):
    def __init__(self):
        super().__init__(nInputPorts=1, nOutputPorts=2,
                         outputType="vtkUnstructuredGrid")
        self._mode = MODE_INPUT_POINTS
        self._num_points = 1000
        self._rng_seed = 0
        self._use_input_bounds = 1
        self._margin = 5.0
        self._box = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        self._min_distance = 0.01
        self._engine = ENGINE_AUTO
        self._threads = 0
        self._lloyd = 0
        self._conv_tol = 0.005
        self._min_separation = 1e-6
        self._move_axes = [True, True, True]
        self._density_array = ""
        self._density_field = 0
        self._density_radius = 0.2
        self._density_exponent = 1.0

    @smproperty.intvector(name="Mode", label="Seeds", default_values=0)
    @smdomain.xml(_MODE_XML)
    def SetMode(self, v):
        self._mode = int(v)
        self.Modified()

    @smproperty.intvector(name="NumPoints",
                          label="Number of random points",
                          default_values=1000)
    @smdomain.xml(_doc("Used with Seeds = Random points, or Input points + random "
                       "points (the random points have density 0, so they are free to "
                       "move)."))
    def SetNumPoints(self, v):
        self._num_points = max(2, int(v))
        self.Modified()

    @smproperty.intvector(name="RandomSeed", label="Random seed",
                          default_values=0)
    @smdomain.xml(_doc("Only used when random points are generated."))
    def SetRandomSeed(self, v):
        self._rng_seed = int(v)
        self.Modified()

    @smproperty.intvector(name="UseInputBounds",
                          label="Box from input bounds",
                          default_values=1)
    @smdomain.xml('<BooleanDomain name="bool"/>'
                  + _doc("On: the volume is the bounding box of the input plus "
                         "the margin. Off: the volume is the manual box below."))
    def SetUseInputBounds(self, v):
        self._use_input_bounds = int(v)
        self.Modified()

    @smproperty.doublevector(name="Margin", label="Margin on bounds (%)",
                             default_values=5.0)
    def SetMargin(self, v):
        self._margin = max(0.0, float(v))
        self.Modified()

    @smproperty.doublevector(name="Box",
                             label="Manual box (xmin xmax ymin ymax zmin zmax)",
                             default_values=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                             number_of_elements=6)
    def SetBox(self, xmin, xmax, ymin, ymax, zmin, zmax):
        self._box = [xmin, xmax, ymin, ymax, zmin, zmax]
        self.Modified()

    @smproperty.doublevector(name="MinDistance",
                             label="Min distance from walls (fraction of box size)",
                             default_values=0.01)
    @smdomain.xml(_doc("Seeds are kept at least this far from every wall of the box, "
                       "as a fraction of the box size (0.01 = 1%). Random points are "
                       "generated outside this margin; input points inside it are "
                       "ignored. Seeds on or next to a wall give degenerate cells."))
    def SetMinDistance(self, v):
        self._min_distance = min(max(float(v), 0.0), 0.49)
        self.Modified()

    @smproperty.intvector(name="LloydIterations", label="Lloyd iterations",
                          default_values=0)
    @smdomain.xml(_doc("0 = plain Voronoi. Each iteration moves the seeds to the "
                       "cell centroids (more regular cells). With a convergence "
                       "threshold this is the maximum number of iterations."))
    def SetLloydIterations(self, v):
        self._lloyd = max(0, int(v))
        self.Modified()

    @smproperty.doublevector(name="ConvergenceThreshold",
                             label="Convergence threshold (mean seed shift / cell size)",
                             default_values=0.005)
    @smdomain.xml(_doc("The Lloyd iterations stop earlier when the mean displacement of "
                       "the free seeds, divided by the mean cell size, falls below this "
                       "value (0.005 = 0.5 %). The number of Lloyd iterations is then "
                       "the maximum. 0 = always perform all iterations. The Output "
                       "Messages report which condition ended the calculation."))
    def SetConvergenceThreshold(self, v):
        self._conv_tol = max(0.0, float(v))
        self.Modified()

    @smproperty.stringvector(name="DensityArray", label="Density array",
                             number_of_elements=5, element_types="0 0 0 0 2")
    @smdomain.xml("""
        <ArrayListDomain name="array_list" attribute_type="Scalars"
                         input_domain_name="inputs_array" none_string="None">
          <RequiredProperties>
            <Property name="Input" function="Input"/>
          </RequiredProperties>
        </ArrayListDomain>
        """ + _doc("Point-data array of the input with the density of every point "
                   "(drop-down list). Points with density > 0 never move during the "
                   "Lloyd iterations and the free points (density 0) are attracted to "
                   "them: more, smaller cells around high densities. Choose None for "
                   "the ordinary (unweighted) Lloyd iterations."))
    def SetDensityArray(self, idx, port, connection, field, name):
        # called by ParaView with (index, port, connection, field association, name)
        self._density_array = "" if name in (None, "None") else str(name)
        self._density_field = int(field) if str(field).lstrip("-").isdigit() else 0
        self.Modified()

    @smproperty.doublevector(name="DensityRadius",
                             label="Density influence radius (fraction of box size)",
                             default_values=0.2)
    @smdomain.xml(_doc("Distance over which a point with density > 0 refines the mesh, "
                       "as a fraction of the largest box side. The density around it "
                       "decreases linearly from its value to 0 at this distance."))
    def SetDensityRadius(self, v):
        self._density_radius = max(1e-6, float(v))
        self.Modified()

    @smproperty.doublevector(name="DensityExponent", label="Density exponent",
                             default_values=1.0)
    @smdomain.xml(_doc("The centroid weight is (1 + local density) ^ exponent. Larger "
                       "values give a stronger refinement."))
    def SetDensityExponent(self, v):
        self._density_exponent = max(1e-3, float(v))
        self.Modified()

    @smproperty.doublevector(name="MinSeparation",
                             label="Min distance between points (fraction of box size)",
                             default_values=1e-6)
    @smdomain.xml(_doc("Points closer than this to another point are ignored (the one "
                       "with density > 0 is kept). Coincident or almost coincident "
                       "points make Voro++ unstable. 0 disables the check."))
    def SetMinSeparation(self, v):
        self._min_separation = max(0.0, float(v))
        self.Modified()

    @smproperty.intvector(name="MoveX", label="Move seeds along X", default_values=1)
    @smdomain.xml('<BooleanDomain name="bool"/>'
                  + _doc("Lloyd iterations: the seeds may move along X. Uncheck the axes "
                         "perpendicular to planar (2D) or layered (2.5D) point sets so "
                         "that rounding errors do not move the points out of their planes."))
    def SetMoveX(self, v):
        self._move_axes[0] = bool(int(v))
        self.Modified()

    @smproperty.intvector(name="MoveY", label="Move seeds along Y", default_values=1)
    @smdomain.xml('<BooleanDomain name="bool"/>')
    def SetMoveY(self, v):
        self._move_axes[1] = bool(int(v))
        self.Modified()

    @smproperty.intvector(name="MoveZ", label="Move seeds along Z", default_values=1)
    @smdomain.xml('<BooleanDomain name="bool"/>')
    def SetMoveZ(self, v):
        self._move_axes[2] = bool(int(v))
        self.Modified()

    @smproperty.intvector(name="Engine", label="Calculation engine",
                          default_values=0)
    @smdomain.xml(_ENGINE_XML)
    def SetEngine(self, v):
        self._engine = int(v)
        self.Modified()

    @smproperty.intvector(name="NumThreads", label="Number of threads (OpenMP engine, 0 = all)",
                          default_values=0, panel_visibility="advanced")
    @smdomain.xml('<IntRangeDomain name="range" min="0" max="1024"/>')
    def SetNumThreads(self, v):
        self._threads = int(v)
        self.Modified()

    def RequestData(self, request, inInfo, outInfo):
        try:
            inp = dsa.WrapDataObject(vtkDataSet.GetData(inInfo[0], 0))
            out = vtkUnstructuredGrid.GetData(outInfo, 0)
            seeds_out = vtkUnstructuredGrid.GetData(outInfo, 1)
            if inp.Points is None or len(inp.Points) == 0:
                raise ValueError("The input contains no points")
            input_points = np.asarray(inp.Points, dtype=float)

            if self._use_input_bounds:
                box = box_from_points(input_points, self._margin, self._min_distance)
            else:
                box = self._box

            density = None
            if self._density_array:
                if self._density_field != 0:
                    _report_warning("Voronoi 3D: the density array must be point data: "
                                    "'%s' is ignored" % self._density_array)
                else:
                    array = inp.PointData[self._density_array]
                    if array is None or isinstance(array, dsa.VTKNoneArray):
                        _report_warning(
                            "Voronoi 3D: point-data array '%s' not found in the input "
                            "(available: %s): unweighted Lloyd iterations"
                            % (self._density_array, list(inp.PointData.keys())))
                    else:
                        array = np.asarray(array, dtype=float)
                        if array.ndim > 1:
                            array = array[:, 0]
                        density = array

            print("Voronoi 3D: %d input points, density array: %s"
                  % (len(input_points), self._density_array or "none"))
            if self._mode == MODE_RANDOM_POINTS:
                seeds = random_points(self._num_points, self._rng_seed, box,
                                      self._min_distance)
                density = None  # the input points are not seeds
            elif self._mode == MODE_INPUT_AND_RANDOM:
                extra = random_points(self._num_points, self._rng_seed, box,
                                      self._min_distance)
                seeds = np.vstack((input_points, extra))
                if density is not None:
                    density = np.concatenate((density, np.zeros(len(extra))))
            else:
                seeds = input_points
            generate(seeds, box, self._engine, self._lloyd, out, self._min_distance,
                     seeds_out, density, self._density_radius, self._density_exponent,
                     self._min_separation, _progress_for(self), self._move_axes,
                     threads=self._threads, convergence_threshold=self._conv_tol)
        except Cancelled:
            _report_warning("Voronoi 3D: calculation cancelled")
            out.Initialize()
            return 1
        except Exception:
            _report_error()
            return 0
        return 1

# -*- coding: utf-8 -*-
"""
Benchmark of the Voronoi3D ParaView plugin (source "Voronoi3D").

Run it from the ParaView Python Shell (View > Python Shell > Run Script) or with
    pvpython benchmark_voronoi3d.py

For every number of seeds and for every engine (Native C++ / Voro++ and VTK
Delaunay) the tessellation is computed several times and the times are collected.
At the end the script writes, in the output folder:

    benchmark_voronoi3d.csv    all individual runs
    benchmark_voronoi3d.html   summary table (open it in Word or copy-paste it into Word)
    benchmark_voronoi3d.docx   same table as a Word file (only if python-docx is installed)
    benchmark_voronoi3d.md     summary table in Markdown
    benchmark_voronoi3d.txt    machine description + summary table (tab separated)

Two times are recorded for each run:
    wall time   : time.perf_counter() around the whole pipeline update
                  (random points + tessellation + output grid, as seen by the user);
    plugin time : "ComputationTime_s" stored by the plugin in the field data of the output.

Author: Stefano Bonduà (DICAM, University of Bologna)
License: MIT
"""
import csv
import gc
import html
import os
import platform
import statistics
import time

# ----------------------------------------------------------------------------
# USER SETTINGS
# ----------------------------------------------------------------------------
PLUGIN_PATH = r""          # full path of voronoi3d_plugin.py; leave "" if already loaded
OUTPUT_DIR = r""           # output folder; "" = folder of this script / current directory

POINT_COUNTS = [10, 100, 1000, 10000, 100000, 1000000]   # add 10 .. 1e6 as you like
REPEATS = 3                # runs per case (the median and the minimum are reported)
RANDOM_SEED = 0
BOX = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
LLOYD_ITERATIONS = 0       # 0 = plain Voronoi tessellation (as requested)

# Engines to test: label -> value of the "Engine" property of the plugin
ENGINES = [
    ("Voro++ (native C++)", 4),
    ("Voro++ OpenMP", 5),      # needs voronoi3d_engine_omp.dll; remove this line to skip
    ("VTK Delaunay", 3),
]
OMP_THREADS = 0                # threads of the OpenMP engine (0 = all logical cores)
# The VTK engine is very slow and memory hungry for large sets. Cases above these
# limits are skipped (marked "skipped" in the table). None = no limit.
MAX_POINTS = {
    "Voro++ (native C++)": None,
    "Voro++ OpenMP": None,
    "VTK Delaunay": 100000,   # raise (e.g. to 1000000) only if you accept hours and tens of GB
}
# If one run of a case takes longer than this (seconds), its remaining repeats and
# larger cases of the same engine are skipped.
SLOW_RUN_LIMIT_S = 1800.0
# ----------------------------------------------------------------------------

from paraview import servermanager as sm
from paraview import simple


def _script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:               # Python Shell "Run Script" may not define __file__
        return os.getcwd()


def _find_plugin_file():
    """PLUGIN_PATH if set, otherwise search voronoi3d_plugin.py in the usual places."""
    if PLUGIN_PATH:
        return PLUGIN_PATH
    folders = [_script_dir(), os.getcwd(), os.path.join(_script_dir(), ".."),
               os.path.join(os.getcwd(), "..")]
    for folder in folders:
        candidate = os.path.join(folder, "voronoi3d_plugin.py")
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return ""


def _factory():
    """Return a callable that creates the Voronoi3D source, or None if not available."""
    f = getattr(simple, "Voronoi3D", None)
    if f is not None:
        return f
    f = globals().get("Voronoi3D")
    if f is not None:
        return f
    try:
        f = getattr(sm.sources, "Voronoi3D", None)       # works after a GUI plugin load
        if f is not None:
            return f
    except Exception:
        pass
    return None


def _load_plugin():
    if _factory() is not None:
        return
    path = _find_plugin_file()
    if path:
        print("Loading plugin: %s" % path)
        simple.LoadPlugin(path, remote=False, ns=globals())
    if _factory() is None:
        raise RuntimeError(
            "Voronoi3D is not available. Set PLUGIN_PATH (top of this script) to the full path "
            "of voronoi3d_plugin.py, e.g. PLUGIN_PATH = r\"C:\\...\\voronoi3d_plugin.py\"; "
            "searched: %s" % (path or "no voronoi3d_plugin.py found next to the script"))


def _norm(text):
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def _set_prop(src, label, value):
    """Set a property. ParaView's Python names are derived from the panel labels
    ('Number of points' -> 'Numberofpoints'), so the label is used for matching."""
    props = list(src.ListProperties())
    wanted = _norm(label)
    for p in props:
        if _norm(p) == wanted:
            setattr(src, p, value)
            return
    raise RuntimeError("property '%s' not found. Properties of the proxy: %s" % (label, props))


def _make_source(n, engine):
    try:
        src = _factory()(registrationName="bench_voronoi")
    except Exception:                      # factory without registrationName
        src = _factory()()
    _set_prop(src, "Number of points", int(n))
    _set_prop(src, "Random seed", RANDOM_SEED)
    _set_prop(src, "Box (xmin xmax ymin ymax zmin zmax)", list(BOX))
    _set_prop(src, "Lloyd iterations", LLOYD_ITERATIONS)
    _set_prop(src, "Calculation engine", engine)
    if engine == 5:
        _set_prop(src, "Number of threads (OpenMP engine, 0 = all)", int(OMP_THREADS))
    return src


def _field(dataset, name):
    """Return the first value of a field-data array (number or string) or None."""
    fd = dataset.GetFieldData()
    arr = fd.GetAbstractArray(name)
    if arr is None:
        return None
    try:
        return arr.GetValue(0)
    except Exception:
        return None


def _run_once(n, engine):
    src = _make_source(n, engine)
    try:
        t0 = time.perf_counter()
        src.UpdatePipeline()
        wall = time.perf_counter() - t0
        out = src.GetClientSideObject().GetOutputDataObject(0)
        cells = out.GetNumberOfCells()
        plugin_time = _field(out, "ComputationTime_s")
        engine_used = _field(out, "Engine")
        machine = {k: _field(out, "Machine_" + k)
                   for k in ("OS", "CPU", "LogicalCores", "RAM_GB", "Python", "VTK")}
        return dict(wall=wall, plugin=plugin_time, cells=cells,
                    engine_used=engine_used, machine=machine)
    finally:
        try:
            simple.Delete(src)
        except Exception:
            pass
        del src
        gc.collect()


def _fmt_time(x):
    if x is None:
        return "-"
    if x < 0.01:
        return "%.4f" % x
    if x < 10:
        return "%.3f" % x
    if x < 100:
        return "%.2f" % x
    return "%.1f" % x


def main():
    _load_plugin()
    out_dir = OUTPUT_DIR or _script_dir()
    os.makedirs(out_dir, exist_ok=True)

    runs = []          # individual runs
    summary = {}       # (label, n) -> dict
    machine = {}
    stopped = set()    # engines that must skip the remaining cases

    print("=" * 70)
    print("Voronoi3D benchmark  -  %d repeats, Lloyd iterations: %d" % (REPEATS, LLOYD_ITERATIONS))
    print("=" * 70)

    for n in POINT_COUNTS:
        for label, eng in ENGINES:
            limit = MAX_POINTS.get(label)
            if label in stopped or (limit is not None and n > limit):
                summary[(label, n)] = dict(status="skipped")
                print("%-22s N=%-8d skipped" % (label, n))
                continue
            times_wall, times_plugin, cells, used = [], [], None, None
            status = "ok"
            for r in range(REPEATS):
                try:
                    res = _run_once(n, eng)
                except Exception as exc:              # memory error, plugin error, ...
                    status = "failed: %s" % exc
                    print("%-22s N=%-8d FAILED (%s)" % (label, n, exc))
                    stopped.add(label)
                    break
                times_wall.append(res["wall"])
                if res["plugin"] is not None:
                    times_plugin.append(res["plugin"])
                cells, used = res["cells"], res["engine_used"]
                if res["machine"].get("CPU"):
                    machine = res["machine"]
                runs.append(dict(engine=label, n=n, repeat=r + 1, wall_s=res["wall"],
                                 plugin_s=res["plugin"], cells=res["cells"],
                                 engine_used=res["engine_used"]))
                print("%-22s N=%-8d run %d/%d  wall %s s  plugin %s s  cells %d  engine=%s"
                      % (label, n, r + 1, REPEATS, _fmt_time(res["wall"]),
                         _fmt_time(res["plugin"]), res["cells"], res["engine_used"]))
                if res["wall"] > SLOW_RUN_LIMIT_S:
                    print("   run longer than %.0f s: skipping larger cases for %s"
                          % (SLOW_RUN_LIMIT_S, label))
                    stopped.add(label)
                    break
            if times_wall:
                summary[(label, n)] = dict(
                    status=status, cells=cells, engine_used=used,
                    wall_med=statistics.median(times_wall), wall_min=min(times_wall),
                    plugin_med=statistics.median(times_plugin) if times_plugin else None,
                    runs=len(times_wall))
            else:
                summary[(label, n)] = dict(status=status)

    # warn when an engine silently fell back to another one
    expected = {4: "native", 5: "openmp", 3: "vtk"}
    for r in runs:
        eng = dict(ENGINES)[r["engine"]]
        if r["engine_used"] and expected.get(eng, "") not in str(r["engine_used"]).lower():
            print("WARNING: requested '%s' but the plugin reports engine '%s' (N=%d)"
                  % (r["engine"], r["engine_used"], r["n"]))
            break

    # ------------------------------------------------------------------ files
    base = os.path.join(out_dir, "benchmark_voronoi3d")
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["engine", "n_seeds", "repeat", "wall_time_s", "plugin_time_s", "n_cells",
                    "engine_reported"])
        for r in runs:
            w.writerow([r["engine"], r["n"], r["repeat"], "%.6f" % r["wall_s"],
                        "" if r["plugin_s"] is None else "%.6f" % r["plugin_s"],
                        r["cells"], r["engine_used"]])

    header = ["Number of seeds"]
    for label, _ in ENGINES:
        header += ["%s - time (s)" % label]
    labels = [l for l, _ in ENGINES]
    i_ser = labels.index("Voro++ (native C++)") if "Voro++ (native C++)" in labels else None
    i_omp = labels.index("Voro++ OpenMP") if "Voro++ OpenMP" in labels else None
    i_vtk = labels.index("VTK Delaunay") if "VTK Delaunay" in labels else None
    if i_ser is not None and i_omp is not None:
        header += ["Speed-up (serial / OpenMP)"]
    if i_ser is not None and i_vtk is not None:
        header += ["Ratio (VTK / Voro++ serial)"]
    rows = []
    for n in POINT_COUNTS:
        row = ["{:,}".format(n).replace(",", " ")]
        vals = []
        for label, _ in ENGINES:
            s = summary.get((label, n), dict(status="skipped"))
            if "wall_med" in s:
                row.append(_fmt_time(s["wall_med"]))
                vals.append(s["wall_med"])
            else:
                row.append("n/a" if s["status"] == "skipped" else "failed")
                vals.append(None)
        if i_ser is not None and i_omp is not None:
            row.append("%.1f" % (vals[i_ser] / vals[i_omp]) if vals[i_ser] and vals[i_omp] else "-")
        if i_ser is not None and i_vtk is not None:
            row.append("%.0f" % (vals[i_vtk] / vals[i_ser]) if vals[i_ser] and vals[i_vtk] else "-")
        rows.append(row)

    ml = ("%s | CPU: %s | %s logical cores | RAM %s GB | Python %s | VTK %s"
          % (machine.get("OS") or platform.platform(), machine.get("CPU") or platform.processor(),
             machine.get("LogicalCores"), machine.get("RAM_GB"),
             machine.get("Python") or platform.python_version(), machine.get("VTK")))
    caption = ("Table. Calculation time of the 3D Voronoi tessellation of random seeds in the unit "
               "cube (median of %d runs, %d Lloyd iterations, wall-clock time of the whole pipeline). "
               "Machine: %s. n/a = not run (size limit)." % (REPEATS, LLOYD_ITERATIONS, ml))

    # text / markdown
    with open(base + ".txt", "w", encoding="utf-8") as f:
        f.write(caption + "\n\n")
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(caption + "\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(r) + " |\n")
    # html (Word opens it and keeps the table when pasted)
    with open(base + ".html", "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'><style>table{border-collapse:collapse;"
                "font-family:Calibri;font-size:10pt}td,th{border:1px solid #444;padding:3px 8px;"
                "text-align:right}th{background:#dce6f1}td:first-child{text-align:left}</style></head><body>\n")
        f.write("<p>%s</p>\n<table>\n<tr>" % html.escape(caption))
        f.write("".join("<th>%s</th>" % html.escape(h) for h in header) + "</tr>\n")
        for r in rows:
            f.write("<tr>" + "".join("<td>%s</td>" % html.escape(c) for c in r) + "</tr>\n")
        f.write("</table></body></html>\n")
    # docx (optional)
    docx_written = False
    try:
        import docx
        d = docx.Document()
        d.add_paragraph(caption)
        t = d.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        for i, h in enumerate(header):
            t.rows[0].cells[i].text = h
        for r in rows:
            cells = t.add_row().cells
            for i, c in enumerate(r):
                cells[i].text = c
        d.save(base + ".docx")
        docx_written = True
    except Exception:
        pass

    print()
    print(caption)
    print("\t".join(header))
    for r in rows:
        print("\t".join(r))
    print()
    print("Files written in: %s" % out_dir)
    print("  benchmark_voronoi3d.csv / .html / .md / .txt" + (" / .docx" if docx_written else
          "   (python-docx not available: open the .html file in Word instead)"))


main()

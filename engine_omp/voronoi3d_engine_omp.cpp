// voronoi3d_engine_omp.cpp  (OpenMP-parallel variant of voronoi3d_engine.cpp)
// ---------------------------------------------------------------------------
// Native Voronoi engine for the ParaView plugin voronoi3d_plugin.py.
//
// Computes the 3D Voronoi cells of N points clipped to an axis-aligned box
// using Voro++ (https://math.lbl.gov/voro++/), optionally relaxing the points
// with Lloyd iterations, and returns everything the plugin needs to build a
// vtkUnstructuredGrid of VTK_POLYHEDRON cells:
//   - merged vertices shared by neighbouring cells,
//   - for every cell the polyhedron face stream [nFaces, n0, id.., n1, id.., ...]
//     with faces oriented outward,
//   - cell volumes and the (final) seed positions.
//
// Plain C interface, loaded from Python with ctypes (no Python/ParaView
// dependency). Build with build_engine.bat (MSVC) or build_engine.sh (g++).
//
// Author: Stefano Bonduà (DICAM, University of Bologna)
// License: MIT (see LICENSE); links against Voro++ (see THIRD_PARTY_LICENSES.md).
// SPDX-License-Identifier: MIT
// ---------------------------------------------------------------------------
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "voro/cell.hh"
#include "voro/c_loops.hh"
#include "voro/container.hh"
#include "voro/pre_container.hh"
#include "voro/v_compute.hh"
#include <atomic>
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_WIN32)
#define V3D_API extern "C" __declspec(dllexport)
#else
#define V3D_API extern "C" __attribute__((visibility("default")))
#endif

extern "C" {

// Progress callback: receives the completed fraction (0..1); returning non-zero
// cancels the computation.
typedef int (*V3DProgressFn)(double fraction);

struct V3DResult {
    int64_t n_cells;       // number of output cells
    int64_t n_points;      // number of merged vertices
    int64_t stream_len;    // length of `stream`
    int64_t n_skipped;     // seeds for which Voro++ produced no cell
    int64_t n_faces;       // total number of faces (all cells)
    int64_t n_pt_ids;      // length of `pt_ids`
    double *points;        // n_points * 3
    int64_t *seed_ids;     // n_cells: index of the seed in the input array
    double *volumes;       // n_cells
    int32_t *num_faces;    // n_cells
    int64_t *cell_offsets; // n_cells + 1: start of each cell in `stream`
    int64_t *stream;       // concatenated polyhedron face streams
    double *seeds;         // n_cells * 3: final seed positions (after Lloyd)
    int64_t *pt_offsets;   // n_cells + 1: unique vertex ids of every cell ...
    int64_t *pt_ids;       // ... (needed by VTK's polyhedron cell array)
    int64_t *face_offsets; // n_faces + 1: faces as a flat cell array ...
    int64_t *face_conn;    // ... (vertex ids of all faces, cell by cell)
    int64_t *cell_face_offsets;  // n_cells + 1: faces (index range) of each cell
    char error[512];       // empty when everything went fine
};

}  // extern "C"

namespace {

// Reports progress of the current stage: the fraction `base + span * inner`.
struct Progress {
    V3DProgressFn fn = nullptr;
    double base = 0.0, span = 1.0;
    void report(double inner) const {
        if (fn && fn(base + span * inner))
            throw std::runtime_error("calculation cancelled");
    }
};

struct RawCell {
    int64_t seed_id;
    double volume;
    double centroid[3];                // absolute coordinates
    std::vector<double> verts;         // 3 * nv, absolute coordinates
    std::vector<int> faces;            // Voro++ format: n, v0, .., v(n-1), n, ...
};


// Density field used by the weighted Lloyd iterations.
//   rho(x) = 1 + max_j  d_j * max(0, 1 - |x - p_j| / R)
// over the anchor points p_j (input points with density d_j > 0). The weight of
// a location in the centroid integral is rho(x)^exponent.
class DensityField {
public:
    DensityField(const std::vector<double> &pts, const double *density, const double *lo,
                 const double *hi, double radius, double exponent)
        : radius_(radius), exponent_(exponent) {
        const int64_t n = static_cast<int64_t>(pts.size() / 3);
        for (int64_t i = 0; i < n; ++i) {
            if (density[i] > 0.0) {
                ap_.push_back(pts[3 * i]);
                ap_.push_back(pts[3 * i + 1]);
                ap_.push_back(pts[3 * i + 2]);
                ad_.push_back(density[i]);
            }
        }
        for (int a = 0; a < 3; ++a) {
            lo_[a] = lo[a];
            const double ext = hi[a] - lo[a];
            nb_[a] = std::max(1, std::min(128, static_cast<int>(std::floor(ext / radius_))));
            h_[a] = ext / nb_[a];  // >= radius_ (so 27 buckets always suffice)
        }
        const size_t nbk = static_cast<size_t>(nb_[0]) * nb_[1] * nb_[2];
        start_.assign(nbk + 1, 0);
        const size_t na = ad_.size();
        std::vector<size_t> bucket(na);
        for (size_t j = 0; j < na; ++j) {
            bucket[j] = index(cell(ap_[3 * j], 0), cell(ap_[3 * j + 1], 1), cell(ap_[3 * j + 2], 2));
            ++start_[bucket[j] + 1];
        }
        for (size_t b = 0; b < nbk; ++b) start_[b + 1] += start_[b];
        order_.resize(na);
        std::vector<size_t> fill(start_.begin(), start_.end() - 1);
        for (size_t j = 0; j < na; ++j) order_[fill[bucket[j]]++] = j;
    }

    bool has_anchors() const { return !ad_.empty(); }

    // rho(x): 1 far from every anchor
    double rho(double x, double y, double z) const {
        double best = 0.0;
        const int cx = cell(x, 0), cy = cell(y, 1), cz = cell(z, 2);
        for (int ix = std::max(cx - 1, 0); ix <= std::min(cx + 1, nb_[0] - 1); ++ix)
            for (int iy = std::max(cy - 1, 0); iy <= std::min(cy + 1, nb_[1] - 1); ++iy)
                for (int iz = std::max(cz - 1, 0); iz <= std::min(cz + 1, nb_[2] - 1); ++iz) {
                    const size_t b = index(ix, iy, iz);
                    for (size_t k = start_[b]; k < start_[b + 1]; ++k) {
                        const size_t j = order_[k];
                        const double dx = ap_[3 * j] - x, dy = ap_[3 * j + 1] - y,
                                     dz = ap_[3 * j + 2] - z;
                        const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                        if (r < radius_) best = std::max(best, ad_[j] * (1.0 - r / radius_));
                    }
                }
        return 1.0 + best;
    }

    double weight(double x, double y, double z) const {
        const double r = rho(x, y, z);
        return exponent_ == 1.0 ? r : std::pow(r, exponent_);
    }

private:
    int cell(double v, int a) const {
        int c = static_cast<int>(std::floor((v - lo_[a]) / h_[a]));
        return std::max(0, std::min(nb_[a] - 1, c));
    }
    size_t index(int x, int y, int z) const {
        return (static_cast<size_t>(x) * nb_[1] + y) * nb_[2] + z;
    }
    double radius_, exponent_;
    double lo_[3], h_[3];
    int nb_[3];
    std::vector<double> ap_, ad_;
    std::vector<size_t> start_, order_;
};

// Centroid of a Voronoi cell weighted by the density field. The cell is split
// into tetrahedra: the seed plus a fan of triangles around the centre of every
// face (so the result does not depend on which vertex of a face comes first, and
// mirror-symmetric cells give mirror-symmetric centroids), each triangle being
// subdivided kSubdivisions times (x4 per level). The weight is sampled at the
// centroid of every tetrahedron. Returns false if the cell has no volume.
constexpr int kSubdivisions = 1;

struct WeightAccumulator {
    const DensityField *field;
    const double *seed;
    double wsum = 0.0, acc[3] = {0.0, 0.0, 0.0};

    // tetrahedron (origin = seed, a, b, c) given relative to the seed
    void tetra(const double *a, const double *b, const double *c, int level) {
        if (level > 0) {
            double ab[3], bc[3], ca[3];
            for (int k = 0; k < 3; ++k) {
                ab[k] = 0.5 * (a[k] + b[k]);
                bc[k] = 0.5 * (b[k] + c[k]);
                ca[k] = 0.5 * (c[k] + a[k]);
            }
            tetra(a, ab, ca, level - 1);
            tetra(ab, b, bc, level - 1);
            tetra(ca, bc, c, level - 1);
            tetra(ab, bc, ca, level - 1);
            return;
        }
        const double det = a[0] * (b[1] * c[2] - b[2] * c[1]) -
                           a[1] * (b[0] * c[2] - b[2] * c[0]) +
                           a[2] * (b[0] * c[1] - b[1] * c[0]);
        const double vol = std::fabs(det) / 6.0;
        const double tx = 0.25 * (a[0] + b[0] + c[0]);
        const double ty = 0.25 * (a[1] + b[1] + c[1]);
        const double tz = 0.25 * (a[2] + b[2] + c[2]);
        const double w = vol * field->weight(seed[0] + tx, seed[1] + ty, seed[2] + tz);
        wsum += w;
        acc[0] += w * tx;
        acc[1] += w * ty;
        acc[2] += w * tz;
    }
};

bool weighted_centroid(voro::voronoicell &c, const double seed[3], const DensityField &field,
                       double out[3]) {
    static thread_local std::vector<double> v;
    static thread_local std::vector<int> f;
    c.vertices(v);  // relative to the seed
    c.face_vertices(f);
    WeightAccumulator W;
    W.field = &field;
    W.seed = seed;
    size_t pos = 0;
    while (pos < f.size()) {
        const int m = f[pos++];
        double fc[3] = {0.0, 0.0, 0.0};
        for (int k = 0; k < m; ++k)
            for (int d = 0; d < 3; ++d) fc[d] += v[3 * f[pos + k] + d];
        for (int d = 0; d < 3; ++d) fc[d] /= m;
        for (int k = 0; k < m; ++k) {
            const double *p0 = &v[3 * f[pos + k]];
            const double *p1 = &v[3 * f[pos + (k + 1) % m]];
            W.tetra(fc, p0, p1, kSubdivisions);
        }
        pos += m;
    }
    if (!(W.wsum > 0.0)) return false;
    for (int a = 0; a < 3; ++a) out[a] = seed[a] + W.acc[a] / W.wsum;
    return true;
}

// Number of OpenMP threads (0 = all available); set from the plugin.
int g_num_threads = 0;

// Computes all cells for `pts` (n*3) inside box lo/hi, in parallel (OpenMP).
// With `field` the centroid of every cell is weighted by the density field.
//
// Voro++ is thread-safe for reading a container as long as every thread owns its
// own voro_compute workspace and its own voronoicell. The (cheap) list of
// particles is built serially; each thread then computes a share of the cells and
// stores each result in its own slot, so the output order is identical to the
// serial engine.
void compute_cells(const std::vector<double> &pts, const double *lo, const double *hi,
                   std::vector<RawCell> &cells, int64_t &skipped, bool need_geometry,
                   const DensityField *field = nullptr, const Progress *progress = nullptr) {
    const int n = static_cast<int>(pts.size() / 3);
    voro::pre_container pcon(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], false, false, false);
    for (int i = 0; i < n; ++i) pcon.put(i, pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]);
    int nx, ny, nz;
    pcon.guess_optimal(nx, ny, nz);
    voro::container con(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], nx, ny, nz,
                        false, false, false, 8);
    pcon.setup(con);

    struct Task {
        int ijk, q, i, j, k, pid;
        double x, y, z;
    };
    std::vector<Task> tasks;
    tasks.reserve(n);
    {
        voro::c_loop_all cl(con);
        if (cl.start()) {
            do {
                Task t;
                t.ijk = cl.ijk; t.q = cl.q; t.i = cl.i; t.j = cl.j; t.k = cl.k;
                t.pid = cl.pid();
                cl.pos(t.x, t.y, t.z);
                tasks.push_back(t);
            } while (cl.inc());
        }
    }
    const int64_t ntask = static_cast<int64_t>(tasks.size());
    std::vector<RawCell> slots(static_cast<size_t>(ntask));
    std::vector<char> ok(static_cast<size_t>(ntask), 0);
    std::atomic<bool> stop(false), cancelled(false);
    std::atomic<int64_t> done(0);
    std::string failure;  // first error message (written under `critical`)

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = g_num_threads > 0 ? g_num_threads : omp_get_max_threads();
#endif
#pragma omp parallel num_threads(nthreads)
    {
        voro::voro_compute<voro::container> vc(con, nx, ny, nz);
        voro::voronoicell c;
        int64_t mine = 0;
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
#pragma omp for schedule(dynamic, 64)
        for (int64_t t = 0; t < ntask; ++t) {
            if (stop.load(std::memory_order_relaxed)) continue;
            try {
                const Task &T = tasks[static_cast<size_t>(t)];
                if (vc.compute_cell(c, T.ijk, T.q, T.i, T.j, T.k)) {
                    RawCell &rc = slots[static_cast<size_t>(t)];
                    rc.seed_id = T.pid;
                    rc.volume = c.volume();
                    double cx, cy, cz;
                    c.centroid(cx, cy, cz);
                    rc.centroid[0] = T.x + cx;
                    rc.centroid[1] = T.y + cy;
                    rc.centroid[2] = T.z + cz;
                    if (field) {
                        const double seed[3] = {T.x, T.y, T.z};
                        weighted_centroid(c, seed, *field, rc.centroid);
                    }
                    if (need_geometry) {
                        c.vertices(T.x, T.y, T.z, rc.verts);
                        c.face_vertices(rc.faces);
                    }
                    ok[static_cast<size_t>(t)] = 1;
                }
            } catch (const std::exception &e) {
#pragma omp critical(v3d_fail)
                {
                    if (failure.empty()) failure = e.what();
                }
                stop.store(true);
            } catch (...) {
#pragma omp critical(v3d_fail)
                {
                    if (failure.empty()) failure = "unknown error in a worker thread";
                }
                stop.store(true);
            }
            const int64_t d = ++done;
            // progress / cancel only from the thread that called the engine
            if (tid == 0 && progress && (++mine & 255) == 0) {
                try {
                    progress->report(double(d) / double(ntask));
                } catch (...) {
                    cancelled.store(true);
                    stop.store(true);
                }
            }
        }
    }
    if (cancelled.load()) throw std::runtime_error("calculation cancelled");
    if (!failure.empty()) throw std::runtime_error(failure);

    cells.clear();
    cells.reserve(static_cast<size_t>(ntask));
    for (int64_t t = 0; t < ntask; ++t)
        if (ok[static_cast<size_t>(t)]) cells.push_back(std::move(slots[static_cast<size_t>(t)]));
    skipped = static_cast<int64_t>(n) - static_cast<int64_t>(cells.size());
    if (progress) progress->report(1.0);
}

// Merges vertices closer than `tol` using a spatial hash. Buckets are much
// larger than `tol`, so a vertex only needs to look into a neighbouring bucket
// when it lies within `tol` of the bucket border (checked per axis).
class VertexMerger {
public:
    explicit VertexMerger(double tol, size_t expected)
        : tol_(tol), h_(64.0 * tol), inv_(1.0 / (64.0 * tol)) {
        size_t cap = 16;
        while (cap < 2 * expected + 2) cap <<= 1;
        tkey_.assign(cap, kEmpty);
        thead_.assign(cap, -1);
        mask_ = cap - 1;
        pts_.reserve(3 * expected);
        next_.reserve(expected);
    }

    int64_t add(double x, double y, double z) {
        const double fx = x * inv_, fy = y * inv_, fz = z * inv_;
        const int64_t kx = static_cast<int64_t>(std::floor(fx));
        const int64_t ky = static_cast<int64_t>(std::floor(fy));
        const int64_t kz = static_cast<int64_t>(std::floor(fz));
        // neighbouring bucket to inspect on each axis (0 = none)
        const double margin = tol_ * inv_;
        const int sx = side(fx - kx, margin), sy = side(fy - ky, margin), sz = side(fz - kz, margin);
        for (int ix = 0; ix <= (sx != 0 ? 1 : 0); ++ix)
            for (int iy = 0; iy <= (sy != 0 ? 1 : 0); ++iy)
                for (int iz = 0; iz <= (sz != 0 ? 1 : 0); ++iz) {
                    const size_t slot = find(key(kx + ix * sx, ky + iy * sy, kz + iz * sz));
                    if (tkey_[slot] == kEmpty) continue;
                    for (int64_t id = thead_[slot]; id >= 0; id = next_[id]) {
                        const double ex = pts_[3 * id] - x, ey = pts_[3 * id + 1] - y,
                                     ez = pts_[3 * id + 2] - z;
                        if (ex * ex + ey * ey + ez * ez <= tol_ * tol_) return id;
                    }
                }
        const int64_t id = static_cast<int64_t>(next_.size());
        pts_.push_back(x);
        pts_.push_back(y);
        pts_.push_back(z);
        if (2 * (used_ + 1) > tkey_.size()) grow();
        const uint64_t k = key(kx, ky, kz);
        const size_t slot = find(k);
        if (tkey_[slot] == kEmpty) {
            tkey_[slot] = k;
            ++used_;
            next_.push_back(-1);
        } else {
            next_.push_back(thead_[slot]);
        }
        thead_[slot] = id;
        return id;
    }

    std::vector<double> &points() { return pts_; }

private:
    static constexpr uint64_t kEmpty = ~0ULL;
    // slot of `k` in the open-addressing table (or the empty slot where it belongs)
    size_t find(uint64_t k) const {
        size_t i = static_cast<size_t>((k * 0x9E3779B97F4A7C15ULL) >> 20) & mask_;
        while (tkey_[i] != kEmpty && tkey_[i] != k) i = (i + 1) & mask_;
        return i;
    }
    void grow() {
        std::vector<uint64_t> ok;
        std::vector<int64_t> oh;
        ok.swap(tkey_);
        oh.swap(thead_);
        const size_t cap = ok.size() * 2;
        tkey_.assign(cap, kEmpty);
        thead_.assign(cap, -1);
        mask_ = cap - 1;
        for (size_t i = 0; i < ok.size(); ++i)
            if (ok[i] != kEmpty) {
                const size_t s = find(ok[i]);
                tkey_[s] = ok[i];
                thead_[s] = oh[i];
            }
    }
    static int side(double frac, double margin) {
        if (frac < margin) return -1;
        if (frac > 1.0 - margin) return 1;
        return 0;
    }
    static uint64_t key(int64_t a, int64_t b, int64_t c) {
        // 21 bits per axis, wrapped (collisions only cost a few extra comparisons)
        const uint64_t m = (1ULL << 21) - 1;
        return (static_cast<uint64_t>(a) & m) | ((static_cast<uint64_t>(b) & m) << 21) |
               ((static_cast<uint64_t>(c) & m) << 42);
    }
    double tol_, h_, inv_;
    std::vector<uint64_t> tkey_;                  // open-addressing table: bucket key ...
    std::vector<int64_t> thead_;                  // ... -> newest vertex id of the bucket
    size_t mask_ = 0, used_ = 0;
    std::vector<int64_t> next_;                   // vertex id -> previous in bucket (-1 = end)
    std::vector<double> pts_;
};

}  // namespace

V3D_API void v3d_free(V3DResult *r) {
    if (!r) return;
    delete[] r->points;
    delete[] r->seed_ids;
    delete[] r->volumes;
    delete[] r->num_faces;
    delete[] r->cell_offsets;
    delete[] r->stream;
    delete[] r->seeds;
    delete[] r->pt_offsets;
    delete[] r->pt_ids;
    delete[] r->face_offsets;
    delete[] r->face_conn;
    delete[] r->cell_face_offsets;
    delete r;
}

// pts: n*3 doubles, all strictly inside the box. box: xmin xmax ymin ymax zmin zmax.
// Returns a newly allocated result (release it with v3d_free), never NULL
// except when the allocation of the result itself fails; check `error`.
//
// density (optional, may be NULL): one value per point. Points with density > 0
// are fixed during the Lloyd iterations and attract the free points (density 0)
// within `radius`; the centroids are weighted by (1 + local density)^exponent.
//
// move_axes (optional, may be NULL): 3 flags (x, y, z); a seed only moves along
// the axes whose flag is non-zero during the Lloyd iterations (e.g. points on a
// plane z = const stay on that plane).
//
// history (optional, may be NULL): caller-allocated array of 3 * lloyd_iterations
// doubles. For every Lloyd iteration it receives: the maximum and the mean
// displacement of the free seeds, both divided by the mean cell size
// h = (box volume / n)^(1/3), and the coefficient of variation of the cell volumes
// of the tessellation the iteration started from. They measure the convergence.
//
// shift_tol (0 = off): the iterations stop as soon as the mean displacement of the free
// seeds (in units of h) falls below this value; lloyd_iterations is then a maximum.
// iterations_done (optional, may be NULL) receives the number of iterations performed.
V3D_API V3DResult *v3d_compute_ex5(int64_t n, const double *pts_in, const double *box,
                                   int32_t lloyd_iterations, const double *density,
                                   double radius, double exponent, V3DProgressFn progress_fn,
                                   const int32_t *move_axes, double *history, double shift_tol,
                                   int32_t *iterations_done) {
    V3DResult *r = new (std::nothrow) V3DResult();
    if (!r) return nullptr;
    std::memset(r, 0, sizeof(V3DResult));
    try {
        if (n < 2 || n > 2000000000LL) throw std::runtime_error("invalid number of points");
        const double lo[3] = {box[0], box[2], box[4]};
        const double hi[3] = {box[1], box[3], box[5]};
        for (int a = 0; a < 3; ++a)
            if (!(hi[a] > lo[a])) throw std::runtime_error("invalid box");

        std::vector<double> pts(pts_in, pts_in + 3 * n);
        std::vector<RawCell> cells;
        int64_t skipped = 0;

        std::unique_ptr<DensityField> field;
        if (density) {
            if (!(radius > 0.0) || !(exponent > 0.0))
                throw std::runtime_error("invalid density radius or exponent");
            field.reset(new DensityField(pts, density, lo, hi, radius, exponent));
            if (!field->has_anchors()) field.reset();  // all densities 0: plain Lloyd
        }

        // progress: every Lloyd iteration counts 1 unit, the final tessellation 1
        // and the assembly of the output 0.6
        const double units = lloyd_iterations + 1.6;
        Progress prog;
        prog.fn = progress_fn;
        prog.span = 1.0 / units;

        // Lloyd relaxation: move every free seed to the (weighted) centroid of its cell
        const double h_ref = std::cbrt((hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]) /
                                       static_cast<double>(n));
        int done_iterations = 0;
        for (int it = 0; it < lloyd_iterations; ++it) {
            prog.base = it / units;
            compute_cells(pts, lo, hi, cells, skipped, false, field.get(), &prog);
            double d_max = 0.0, d_sum = 0.0, v_sum = 0.0, v_sum2 = 0.0;
            int64_t n_moved = 0, n_vol = 0;
            for (const RawCell &rc : cells) {
                if (rc.volume <= 0.0) continue;
                v_sum += rc.volume;
                v_sum2 += rc.volume * rc.volume;
                ++n_vol;
                if (density && density[rc.seed_id] > 0.0) continue;  // fixed anchor
                double d2 = 0.0;
                for (int a = 0; a < 3; ++a) {
                    if (move_axes && !move_axes[a]) continue;  // frozen axis
                    // keep the seed strictly inside the box
                    const double eps = 1e-9 * (hi[a] - lo[a]);
                    double v = rc.centroid[a];
                    if (v < lo[a] + eps) v = lo[a] + eps;
                    if (v > hi[a] - eps) v = hi[a] - eps;
                    const double dv = v - pts[3 * rc.seed_id + a];
                    d2 += dv * dv;
                    pts[3 * rc.seed_id + a] = v;
                }
                const double d = std::sqrt(d2) / h_ref;
                d_max = std::fmax(d_max, d);
                d_sum += d;
                ++n_moved;
            }
            if (history) {
                double cv = 0.0;
                if (n_vol > 0 && v_sum > 0.0) {
                    const double mean = v_sum / n_vol;
                    cv = std::sqrt(std::fmax(v_sum2 / n_vol - mean * mean, 0.0)) / mean;
                }
                history[3 * it] = d_max;
                history[3 * it + 1] = n_moved ? d_sum / n_moved : 0.0;
                history[3 * it + 2] = cv;
            }
            done_iterations = it + 1;
            if (shift_tol > 0.0 && n_moved > 0 && d_sum / n_moved < shift_tol) break;
        }
        if (iterations_done) *iterations_done = done_iterations;

        prog.base = lloyd_iterations / units;
        compute_cells(pts, lo, hi, cells, skipped, true, nullptr, &prog);
        r->n_skipped = skipped;
        const int64_t nc = static_cast<int64_t>(cells.size());
        if (nc == 0) throw std::runtime_error("Voro++ computed no cell");

        // Voro++ visits the cells block by block, i.e. in spatial order. The cells
        // are assembled in that order (neighbouring cells share most vertices), in
        // chunks processed in parallel; the result is then written in the order of
        // the input seeds, as the serial engine does.
        double extent = 0.0;
        for (int a = 0; a < 3; ++a) extent = std::fmax(extent, hi[a] - lo[a]);
        const double tol = 1e-9 * extent;

        struct Chunk {
            std::vector<double> pts;        // vertices merged inside the chunk
            std::vector<int64_t> stream;    // polyhedron streams, chunk-local vertex ids
            std::vector<int64_t> pt_ids;    // unique chunk-local vertex ids of every cell
            std::vector<int64_t> gmap;      // chunk-local vertex -> global vertex
        };
        struct CellRec {
            int64_t seed_id = -1, s0 = 0, s1 = 0, p0 = 0, p1 = 0;
            double volume = 0.0, seed[3] = {0, 0, 0};
            int32_t nf = 0, chunk = 0;
            bool kept = false;
        };
        const int64_t CH = 1024;  // cells per chunk (fixed: the result does not depend on the threads)
        const int64_t nchunks = (nc + CH - 1) / CH;
        std::vector<Chunk> chunks(static_cast<size_t>(nchunks));
        std::vector<CellRec> rec(static_cast<size_t>(nc));
        int nthreads_a = 1;
#ifdef _OPENMP
        nthreads_a = g_num_threads > 0 ? g_num_threads : omp_get_max_threads();
#endif
        Progress build;
        build.fn = progress_fn;
        build.base = (lloyd_iterations + 1.0) / units;
        build.span = 0.6 / units;
        std::atomic<int64_t> chunks_done(0), n_degenerate(0);
        std::atomic<bool> a_stop(false), a_cancel(false);
        std::string a_fail;
#pragma omp parallel num_threads(nthreads_a)
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            std::vector<int64_t> gid, face, cell_pts;
#pragma omp for schedule(dynamic, 1)
            for (int64_t ci = 0; ci < nchunks; ++ci) {
                if (a_stop.load(std::memory_order_relaxed)) continue;
                try {
                    Chunk &ch = chunks[static_cast<size_t>(ci)];
                    const int64_t c0 = ci * CH, c1 = std::min<int64_t>(nc, c0 + CH);
                    VertexMerger merger(tol, static_cast<size_t>(c1 - c0) * 7);
                    for (int64_t cidx = c0; cidx < c1; ++cidx) {
                        RawCell &rc = cells[static_cast<size_t>(cidx)];
                        CellRec &cr = rec[static_cast<size_t>(cidx)];
                        const int nv = static_cast<int>(rc.verts.size() / 3);
                        gid.assign(nv, 0);
                        for (int v = 0; v < nv; ++v)
                            gid[v] = merger.add(rc.verts[3 * v], rc.verts[3 * v + 1], rc.verts[3 * v + 2]);
                        const double *sp = &pts[3 * rc.seed_id];
                        const size_t cell_start = ch.stream.size();
                        cell_pts.clear();
                        ch.stream.push_back(0);  // number of faces, filled below
                        int32_t kept_faces = 0;
                        size_t pos = 0;
                        const std::vector<double> &P = merger.points();
                        while (pos < rc.faces.size()) {
                            const int m = rc.faces[pos++];
                            face.clear();
                            for (int k = 0; k < m; ++k) {
                                const int64_t id = gid[rc.faces[pos + k]];
                                if (face.empty() || face.back() != id) face.push_back(id);
                            }
                            pos += m;
                            if (face.size() > 1 && face.front() == face.back()) face.pop_back();
                            if (face.size() < 3) continue;
                            // orient outward: Newell normal against (face centre - seed)
                            double nx = 0, ny = 0, nz = 0, cx = 0, cy = 0, cz = 0;
                            const size_t fs = face.size();
                            for (size_t k = 0; k < fs; ++k) {
                                const double *a = &P[3 * face[k]];
                                const double *b = &P[3 * face[(k + 1) % fs]];
                                nx += (a[1] - b[1]) * (a[2] + b[2]);
                                ny += (a[2] - b[2]) * (a[0] + b[0]);
                                nz += (a[0] - b[0]) * (a[1] + b[1]);
                                cx += a[0];
                                cy += a[1];
                                cz += a[2];
                            }
                            cx = cx / fs - sp[0];
                            cy = cy / fs - sp[1];
                            cz = cz / fs - sp[2];
                            if (nx * cx + ny * cy + nz * cz < 0.0)
                                for (size_t k = 0; k < fs / 2; ++k) std::swap(face[k], face[fs - 1 - k]);
                            ch.stream.push_back(static_cast<int64_t>(fs));
                            for (int64_t id : face) {
                                ch.stream.push_back(id);
                                cell_pts.push_back(id);
                            }
                            ++kept_faces;
                        }
                        cr.kept = false;
                        if (kept_faces < 4) {  // degenerate cell: drop it
                            ch.stream.resize(cell_start);
                            ++n_degenerate;
                        } else {
                            std::sort(cell_pts.begin(), cell_pts.end());
                            cell_pts.erase(std::unique(cell_pts.begin(), cell_pts.end()), cell_pts.end());
                            cr.p0 = static_cast<int64_t>(ch.pt_ids.size());
                            ch.pt_ids.insert(ch.pt_ids.end(), cell_pts.begin(), cell_pts.end());
                            cr.p1 = static_cast<int64_t>(ch.pt_ids.size());
                            ch.stream[cell_start] = kept_faces;
                            cr.s0 = static_cast<int64_t>(cell_start);
                            cr.s1 = static_cast<int64_t>(ch.stream.size());
                            cr.nf = kept_faces;
                            cr.chunk = static_cast<int32_t>(ci);
                            cr.seed_id = rc.seed_id;
                            cr.volume = rc.volume;
                            cr.seed[0] = sp[0]; cr.seed[1] = sp[1]; cr.seed[2] = sp[2];
                            cr.kept = true;
                        }
                        std::vector<double>().swap(rc.verts);   // release the raw cell
                        std::vector<int>().swap(rc.faces);
                    }
                    ch.pts.swap(merger.points());
                } catch (const std::exception &e) {
#pragma omp critical(v3d_fail_asm)
                    {
                        if (a_fail.empty()) a_fail = e.what();
                    }
                    a_stop.store(true);
                } catch (...) {
#pragma omp critical(v3d_fail_asm)
                    {
                        if (a_fail.empty()) a_fail = "unknown error in a worker thread";
                    }
                    a_stop.store(true);
                }
                const int64_t d = ++chunks_done;
                if (tid == 0 && progress_fn) {
                    try {
                        build.report(double(d) / double(nchunks));
                    } catch (...) {
                        a_cancel.store(true);
                        a_stop.store(true);
                    }
                }
            }
        }
        if (a_cancel.load()) throw std::runtime_error("calculation cancelled");
        if (!a_fail.empty()) throw std::runtime_error(a_fail);
        r->n_skipped += n_degenerate.load();

        // Merge the vertices of the chunks (only vertices on chunk borders are shared).
        VertexMerger gmerge(tol, static_cast<size_t>(nc) * 7);
        for (Chunk &ch : chunks) {
            const size_t np = ch.pts.size() / 3;
            ch.gmap.resize(np);
            for (size_t v = 0; v < np; ++v)
                ch.gmap[v] = gmerge.add(ch.pts[3 * v], ch.pts[3 * v + 1], ch.pts[3 * v + 2]);
            std::vector<double>().swap(ch.pts);
        }

        // Output order = order of the input seeds (as in the serial engine)
        std::vector<int64_t> slot(static_cast<size_t>(n), -1);
        for (int64_t i = 0; i < nc; ++i)
            if (rec[static_cast<size_t>(i)].kept) slot[static_cast<size_t>(rec[static_cast<size_t>(i)].seed_id)] = i;
        std::vector<int64_t> order;
        order.reserve(static_cast<size_t>(nc));
        for (int64_t s = 0; s < n; ++s)
            if (slot[static_cast<size_t>(s)] >= 0) order.push_back(slot[static_cast<size_t>(s)]);
        std::vector<int64_t>().swap(slot);
        const int64_t kept = static_cast<int64_t>(order.size());
        if (kept == 0) throw std::runtime_error("no valid cell produced");

        // exclusive scans of the sizes
        r->cell_offsets = new int64_t[kept + 1];
        r->pt_offsets = new int64_t[kept + 1];
        r->cell_face_offsets = new int64_t[kept + 1];
        std::vector<int64_t> fconn_off(static_cast<size_t>(kept) + 1, 0);
        r->cell_offsets[0] = r->pt_offsets[0] = r->cell_face_offsets[0] = 0;
        for (int64_t k = 0; k < kept; ++k) {
            const CellRec &cr = rec[static_cast<size_t>(order[static_cast<size_t>(k)])];
            const int64_t slen = cr.s1 - cr.s0;
            r->cell_offsets[k + 1] = r->cell_offsets[k] + slen;
            r->pt_offsets[k + 1] = r->pt_offsets[k] + (cr.p1 - cr.p0);
            r->cell_face_offsets[k + 1] = r->cell_face_offsets[k] + cr.nf;
            fconn_off[static_cast<size_t>(k) + 1] = fconn_off[static_cast<size_t>(k)] + (slen - 1 - cr.nf);
        }
        const std::vector<double> &P = gmerge.points();
        r->n_cells = kept;
        r->n_points = static_cast<int64_t>(P.size() / 3);
        r->stream_len = r->cell_offsets[kept];
        r->n_faces = r->cell_face_offsets[kept];
        r->n_pt_ids = r->pt_offsets[kept];
        r->points = new double[P.size()];
        std::memcpy(r->points, P.data(), P.size() * sizeof(double));
        r->seed_ids = new int64_t[kept];
        r->volumes = new double[kept];
        r->num_faces = new int32_t[kept];
        r->seeds = new double[3 * kept];
        r->stream = new int64_t[r->stream_len];
        r->pt_ids = new int64_t[r->n_pt_ids];
        r->face_offsets = new int64_t[r->n_faces + 1];
        r->face_conn = new int64_t[fconn_off[static_cast<size_t>(kept)]];
        r->face_offsets[0] = 0;

#pragma omp parallel for num_threads(nthreads_a) schedule(dynamic, 256)
        for (int64_t k = 0; k < kept; ++k) {
            const CellRec &cr = rec[static_cast<size_t>(order[static_cast<size_t>(k)])];
            const Chunk &ch = chunks[static_cast<size_t>(cr.chunk)];
            const int64_t *gm = ch.gmap.data();
            r->seed_ids[k] = cr.seed_id;
            r->volumes[k] = cr.volume;
            r->num_faces[k] = cr.nf;
            r->seeds[3 * k] = cr.seed[0];
            r->seeds[3 * k + 1] = cr.seed[1];
            r->seeds[3 * k + 2] = cr.seed[2];
            int64_t *so = r->stream + r->cell_offsets[k];
            int64_t *fc = r->face_conn + fconn_off[static_cast<size_t>(k)];
            int64_t fi = r->cell_face_offsets[k];
            int64_t fpos = fconn_off[static_cast<size_t>(k)];
            const int64_t *src = ch.stream.data() + cr.s0;
            *so++ = *src++;  // number of faces
            for (int32_t f = 0; f < cr.nf; ++f) {
                const int64_t m = *src++;
                *so++ = m;
                for (int64_t j = 0; j < m; ++j) {
                    const int64_t g = gm[*src++];
                    *so++ = g;
                    *fc++ = g;
                }
                fpos += m;
                r->face_offsets[++fi] = fpos;
            }
            int64_t *po = r->pt_ids + r->pt_offsets[k];
            for (int64_t q = cr.p0; q < cr.p1; ++q) *po++ = gm[ch.pt_ids[static_cast<size_t>(q)]];
            std::sort(r->pt_ids + r->pt_offsets[k], po);
        }
    } catch (const std::exception &e) {
        std::strncpy(r->error, e.what(), sizeof(r->error) - 1);
        delete[] r->points;       r->points = nullptr;
        delete[] r->seed_ids;     r->seed_ids = nullptr;
        delete[] r->volumes;      r->volumes = nullptr;
        delete[] r->num_faces;    r->num_faces = nullptr;
        delete[] r->cell_offsets; r->cell_offsets = nullptr;
        delete[] r->stream;       r->stream = nullptr;
        delete[] r->seeds;        r->seeds = nullptr;
        delete[] r->pt_offsets;   r->pt_offsets = nullptr;
        delete[] r->pt_ids;       r->pt_ids = nullptr;
        delete[] r->face_offsets; r->face_offsets = nullptr;
        delete[] r->face_conn;    r->face_conn = nullptr;
        delete[] r->cell_face_offsets; r->cell_face_offsets = nullptr;
        r->n_cells = r->n_points = r->stream_len = 0;
    } catch (...) {
        std::strncpy(r->error, "unknown error in Voronoi engine", sizeof(r->error) - 1);
    }
    return r;
}

V3D_API V3DResult *v3d_compute_ex4(int64_t n, const double *pts_in, const double *box,
                                   int32_t lloyd_iterations, const double *density,
                                   double radius, double exponent, V3DProgressFn progress_fn,
                                   const int32_t *move_axes, double *history) {
    return v3d_compute_ex5(n, pts_in, box, lloyd_iterations, density, radius, exponent,
                           progress_fn, move_axes, history, 0.0, nullptr);
}

V3D_API V3DResult *v3d_compute_ex3(int64_t n, const double *pts_in, const double *box,
                                   int32_t lloyd_iterations, const double *density,
                                   double radius, double exponent, V3DProgressFn progress_fn,
                                   const int32_t *move_axes) {
    return v3d_compute_ex5(n, pts_in, box, lloyd_iterations, density, radius, exponent,
                           progress_fn, move_axes, nullptr, 0.0, nullptr);
}

V3D_API V3DResult *v3d_compute_ex2(int64_t n, const double *pts_in, const double *box,
                                   int32_t lloyd_iterations, const double *density,
                                   double radius, double exponent, V3DProgressFn progress_fn) {
    return v3d_compute_ex3(n, pts_in, box, lloyd_iterations, density, radius, exponent,
                           progress_fn, nullptr);
}

V3D_API V3DResult *v3d_compute_ex(int64_t n, const double *pts_in, const double *box,
                                  int32_t lloyd_iterations, const double *density,
                                  double radius, double exponent) {
    return v3d_compute_ex2(n, pts_in, box, lloyd_iterations, density, radius, exponent, nullptr);
}

V3D_API V3DResult *v3d_compute(int64_t n, const double *pts_in, const double *box,
                               int32_t lloyd_iterations) {
    return v3d_compute_ex2(n, pts_in, box, lloyd_iterations, nullptr, 1.0, 1.0, nullptr);
}

// 2 = plain Voronoi / Lloyd, 3 = adds density weighted Lloyd (v3d_compute_ex),
// 4 = adds the progress / cancel callback (v3d_compute_ex2),
// 5 = adds the per-axis movement flags (v3d_compute_ex3),
// 6 = adds the convergence history (v3d_compute_ex4),
// 7 = adds the convergence threshold that stops the iterations (v3d_compute_ex5)
V3D_API int32_t v3d_abi_version() { return 7; }

// OpenMP build only: number of threads (0 = all available) and thread count / flag.
V3D_API void v3d_set_num_threads(int32_t n) { g_num_threads = n < 0 ? 0 : n; }
V3D_API int32_t v3d_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
V3D_API int32_t v3d_is_parallel() {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

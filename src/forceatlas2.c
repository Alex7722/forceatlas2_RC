/*
 * ForceAtlas2 graph layout for R.
 *
 * A C implementation of the ForceAtlas2 algorithm:
 *
 *   Jacomy M, Venturini T, Heymann S, Bastian M (2014). ForceAtlas2, a
 *   Continuous Graph Layout Algorithm for Handy Network Visualization Designed
 *   for the Gephi Software. PLoS ONE 9(6): e98679.
 *
 * The force model, the adaptive speed heuristics and the default behaviour
 * follow the reference implementation distributed with Gephi
 * (org.gephi.layout.plugin.forceAtlas2, (c) Gephi Consortium, GPL-3 / CDDL).
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 */

#include <math.h>
#include <string.h>

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Utils.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "fa2.h"

/* ------------------------------------------------------------------------ */
/* Barnes-Hut tree (the data structures are described in fa2.h)             */
/* ------------------------------------------------------------------------ */

/* Regions with at most this many nodes are not split any further: when such
 * a region is too close to be approximated, its nodes are visited directly. */
#define FA2_LEAF_SIZE 8

/* Quadrant of a node around the centre (cx, cy) of its region. */
#define FA2_QUADRANT(node) (((node).x < cx ? 0 : 2) + ((node).y < cy ? 0 : 1))

static inline int all_in_one_quadrant(const int *cnt, int count)
{
    return cnt[0] == count || cnt[1] == count || cnt[2] == count ||
           cnt[3] == count;
}

/*
 * The tree is built breadth first, one level at a time: there is no
 * recursion, hence no risk of exhausting the C stack on degenerate (very
 * deep) trees.
 *
 * The nodes (position, mass and index) are kept grouped by region, so that
 * every pass over a region reads memory sequentially. When a region is split,
 * its nodes are copied to the other of two buffers, quadrant by quadrant and
 * without changing their relative order.
 *
 * The regions of a level cover separate sets of nodes, so they are processed
 * in parallel; the tree does not depend on the number of threads.
 */
static void tree_build(fa2_tree *t, int n, const double *x, const double *y,
                       const double *mass, const double *size, int threads)
{
    fa2_region *reg = t->reg;
    int *cnts = t->cnt;
#ifndef _OPENMP
    (void) threads;
#endif

    /* The root contains all the nodes. */
    {
        fa2_node *nodes = t->buf[0];
        double m = 0.0, sx = 0.0, sy = 0.0;
        for (int i = 0; i < n; i++) {
            nodes[i].x = x[i];
            nodes[i].y = y[i];
            nodes[i].m = mass[i];
            nodes[i].idx = i;
            m += mass[i];
            sx += x[i] * mass[i];
            sy += y[i] * mass[i];
        }
        reg[0].start = 0;
        reg[0].count = n;
        reg[0].skip = -1;
        reg[0].buf = 0;
        reg[0].child = -1;
        reg[0].nchild = 0;
        reg[0].mass = n > 1 ? m : mass[0];
        reg[0].cx = n > 1 ? sx / m : x[0];
        reg[0].cy = n > 1 ? sy / m : y[0];
        reg[0].size = 0.0;
        t->nreg = 1;
    }

    int lo = 0, hi = 1; /* the regions of the current level */
    while (lo < hi) {
#ifdef _OPENMP
        int chunk = (hi - lo) / (8 * threads);
        if (chunk < 1) chunk = 1;
#endif

        /* 1. Size of each region, and number of nodes in its quadrants. */
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic, chunk) if (threads > 1 && hi - lo > 1)
#endif
        for (int r = lo; r < hi; r++) {
            const int count = reg[r].count;
            int *cnt = cnts + 4 * (size_t) (r - lo);
            cnt[0] = cnt[1] = cnt[2] = cnt[3] = 0;
            if (count < 2) continue;

            const fa2_node *nodes = t->buf[reg[r].buf] + reg[r].start;
            const double cx = reg[r].cx, cy = reg[r].cy;
            double d2max = 0.0;
            for (int k = 0; k < count; k++) {
                const double dx = nodes[k].x - cx, dy = nodes[k].y - cy;
                const double d2 = dx * dx + dy * dy;
                if (d2 > d2max) d2max = d2;
                cnt[FA2_QUADRANT(nodes[k])]++;
            }
            reg[r].size = 2.0 * sqrt(d2max);
        }

        /* 2. Create the regions of the next level, in order. */
        for (int r = lo; r < hi; r++) {
            const int start = reg[r].start, count = reg[r].count;
            const int *cnt = cnts + 4 * (size_t) (r - lo);
            if (count <= FA2_LEAF_SIZE) continue;

            reg[r].child = t->nreg;
            if (all_in_one_quadrant(cnt, count)) {
                /* Coincident nodes: each of them becomes a leaf, in place. */
                reg[r].nchild = count;
                for (int k = start; k < start + count; k++) {
                    fa2_region *c = &reg[t->nreg];
                    c->start = k;
                    c->count = 1;
                    c->skip = k == start ? reg[r].skip : t->nreg - 1;
                    c->buf = reg[r].buf;
                    c->child = -1;
                    c->nchild = 0;
                    t->nreg++;
                }
                continue;
            }
            int off = start;
            for (int q = 0; q < 4; q++) {
                if (cnt[q] == 0) continue;
                fa2_region *c = &reg[t->nreg];
                c->start = off;
                c->count = cnt[q];
                c->skip = reg[r].nchild == 0 ? reg[r].skip : t->nreg - 1;
                c->buf = 1 - reg[r].buf;
                c->child = -1;
                c->nchild = 0;
                reg[r].nchild++;
                t->nreg++;
                off += cnt[q];
            }
        }

        /* 3. Move the nodes to their quadrant, and compute the mass and the
         *    centre of the new regions. */
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic, chunk) if (threads > 1 && hi - lo > 1)
#endif
        for (int r = lo; r < hi; r++) {
            const int count = reg[r].count;
            const int *cnt = cnts + 4 * (size_t) (r - lo);
            if (reg[r].nchild == 0) continue;

            const fa2_node *nodes = t->buf[reg[r].buf] + reg[r].start;
            if (!all_in_one_quadrant(cnt, count)) {
                fa2_node *out = t->buf[1 - reg[r].buf];
                const double cx = reg[r].cx, cy = reg[r].cy;
                double m[4] = {0.0, 0.0, 0.0, 0.0};
                double sx[4] = {0.0, 0.0, 0.0, 0.0};
                double sy[4] = {0.0, 0.0, 0.0, 0.0};
                int pos[4];
                pos[0] = reg[r].start;
                for (int q = 1; q < 4; q++) pos[q] = pos[q - 1] + cnt[q - 1];

                for (int k = 0; k < count; k++) {
                    const int q = FA2_QUADRANT(nodes[k]);
                    out[pos[q]++] = nodes[k];
                    m[q] += nodes[k].m;
                    sx[q] += nodes[k].x * nodes[k].m;
                    sy[q] += nodes[k].y * nodes[k].m;
                }

                fa2_region *c = &reg[reg[r].child];
                for (int q = 0; q < 4; q++) {
                    if (cnt[q] == 0) continue;
                    c->mass = m[q];
                    c->cx = sx[q] / m[q];
                    c->cy = sy[q] / m[q];
                    c->size = 0.0;
                    c++;
                }
                nodes = out + reg[r].start;
            }
            /* A region of a single node is that node, exactly. */
            for (int j = 0; j < reg[r].nchild; j++) {
                fa2_region *c = &reg[reg[r].child + j];
                if (c->count > 1) continue;
                const fa2_node *node = nodes + (c->start - reg[r].start);
                c->mass = node->m;
                c->cx = node->x;
                c->cy = node->y;
                c->size = 0.0;
            }
        }

        lo = hi;
        hi = t->nreg;
    }

    /* The nodes in the order of the tree: those of each region that was not
     * split are in the buffer where it left them. */
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static) if (threads > 1)
#endif
    for (int r = 0; r < t->nreg; r++) {
        if (reg[r].nchild > 0) continue;
        const fa2_node *nodes = t->buf[reg[r].buf];
        for (int k = reg[r].start; k < reg[r].start + reg[r].count; k++) {
            const int i = nodes[k].idx;
            t->perm[k] = i;
            t->px[k] = nodes[k].x;
            t->py[k] = nodes[k].y;
            t->pm[k] = nodes[k].m;
            if (size) t->ps[k] = size[i];
        }
    }
}

/* ------------------------------------------------------------------------ */
/* Forces                                                                   */
/* ------------------------------------------------------------------------ */

typedef struct {
    int n, m;
    double *x, *y;     /* positions */
    double *dx, *dy;   /* forces */
    const double *mass;
    const double *size; /* node radii, used only if adjust */
    const int *from, *to;
    const double *w;
    double scaling, gravity, theta;
    int strong, linlog, outbound, adjust;
    /* The edges of each node, for the parallel computation of the attraction:
     * node i has the entries inc[inc_off[i] .. inc_off[i + 1] - 1], each
     * being 2 * edge + (1 if the node is the target of the edge). */
    const int *inc_off, *inc;
} fa2_state;

/* Repulsion factor between two nodes whose centres are sqrt(d2) apart. The
 * force on the first node is (xdist, ydist) * factor. `mm` is the product of
 * the two masses and `ss` the sum of the two radii (unused unless adjust).
 * The result does not depend on the order of the two nodes. */
static inline double repulsion_factor(const fa2_state *s, double d2,
                                      double mm, double ss)
{
    if (s->adjust) {
        const double d = sqrt(d2) - ss;
        if (d > 0) return s->scaling * mm / (d * d);
        if (d < 0) return 100.0 * s->scaling * mm;
        return 0.0;
    }
    return d2 > 0 ? s->scaling * mm / d2 : 0.0;
}

/* Repulsion exerted on node i by every other node (exact, O(n)). */
static void repulse_node_exact(const fa2_state *s, int i)
{
    const double xi = s->x[i], yi = s->y[i], mi = s->mass[i];
    const double si = s->adjust ? s->size[i] : 0.0;
    double fx = 0.0, fy = 0.0;
    for (int j = 0; j < s->n; j++) {
        if (j == i) continue;
        const double xd = xi - s->x[j], yd = yi - s->y[j];
        const double f = repulsion_factor(
            s, xd * xd + yd * yd, mi * s->mass[j],
            s->adjust ? si + s->size[j] : 0.0);
        fx += xd * f;
        fy += yd * f;
    }
    s->dx[i] += fx;
    s->dy[i] += fy;
}

/* Exact repulsion using the symmetry of the force. Each node receives its
 * contributions in increasing order of the other node's index, i.e. in the
 * same order as repulse_node_exact(), so both give identical results. */
static void repulse_all_exact_symmetric(const fa2_state *s)
{
    for (int i = 0; i < s->n; i++) {
        const double xi = s->x[i], yi = s->y[i], mi = s->mass[i];
        const double si = s->adjust ? s->size[i] : 0.0;
        double fx = 0.0, fy = 0.0;
        for (int j = 0; j < i; j++) {
            const double xd = xi - s->x[j], yd = yi - s->y[j];
            const double f = repulsion_factor(
                s, xd * xd + yd * yd, mi * s->mass[j],
                s->adjust ? si + s->size[j] : 0.0);
            fx += xd * f;
            fy += yd * f;
            s->dx[j] -= xd * f;
            s->dy[j] -= yd * f;
        }
        s->dx[i] += fx;
        s->dy[i] += fy;
    }
}

/* Repulsion exerted on node i, approximated with the Barnes-Hut tree. The
 * GPU kernel fa2_bh (gpu.c) visits the same regions in the same order, but
 * follows the `skip` links of the regions instead of using a stack. (On a
 * CPU the stack is much faster: with the links, the next region to visit is
 * only known once the distance test is computed, which prevents the processor
 * from working on several regions at once.) */
static void repulse_node_bh(const fa2_state *s, const fa2_tree *t, int i,
                            int *stack)
{
    const double xi = s->x[i], yi = s->y[i], mi = s->mass[i];
    const double si = s->adjust ? s->size[i] : 0.0;
    const double theta2 = s->theta * s->theta;
    double fx = 0.0, fy = 0.0;
    int sp = 0;

    stack[sp++] = 0;
    while (sp > 0) {
        const fa2_region *r = &t->reg[stack[--sp]];
        if (r->count > 1) {
            const double xd = xi - r->cx, yd = yi - r->cy;
            const double d2 = xd * xd + yd * yd;
            if (d2 * theta2 > r->size * r->size) {
                /* Far enough: the region acts as a single node. */
                const double f = s->scaling * (mi * r->mass) / d2;
                fx += xd * f;
                fy += yd * f;
                continue;
            }
            if (r->nchild > 0) {
                /* The last child is on top of the stack: visited first. */
                for (int c = 0; c < r->nchild; c++) stack[sp++] = r->child + c;
                continue;
            }
        }
        /* A single node, or a small region that is too close. */
        const int end = r->start + r->count;
        for (int k = r->start; k < end; k++) {
            if (t->perm[k] == i) continue;
            const double xd = xi - t->px[k], yd = yi - t->py[k];
            const double f = repulsion_factor(
                s, xd * xd + yd * yd, mi * t->pm[k],
                s->adjust ? si + t->ps[k] : 0.0);
            fx += xd * f;
            fy += yd * f;
        }
    }
    s->dx[i] += fx;
    s->dy[i] += fy;
}

/* Gravity pulls every node towards the origin. */
static inline void gravity_node(const fa2_state *s, int i)
{
    const double xi = s->x[i], yi = s->y[i];
    const double dist = sqrt(xi * xi + yi * yi);
    if (dist > 0) {
        const double f = s->strong ? s->gravity * s->mass[i]
                                   : s->gravity * s->mass[i] / dist;
        s->dx[i] -= xi * f;
        s->dy[i] -= yi * f;
    }
}

/* Attraction factor of edge e: the force on its source is (xd, yd) * factor
 * and the force on its target is the opposite. */
static inline double attraction_factor(const fa2_state *s, int e, double coef,
                                       double xd, double yd)
{
    const int a = s->from[e], b = s->to[e];
    double f = -coef * s->w[e];

    if (s->adjust || s->linlog) {
        double dist = sqrt(xd * xd + yd * yd);
        if (s->adjust) dist -= s->size[a] + s->size[b];
        if (!(dist > 0)) return 0.0;
        if (s->linlog) f *= log1p(dist) / dist;
    }
    if (s->outbound) f /= s->mass[a];
    return f;
}

/* Attraction along the edges. */
static void attract_all(const fa2_state *s, double coef)
{
    for (int e = 0; e < s->m; e++) {
        const int a = s->from[e], b = s->to[e];
        if (a == b) continue;
        const double xd = s->x[a] - s->x[b], yd = s->y[a] - s->y[b];
        const double f = attraction_factor(s, e, coef, xd, yd);
        if (f == 0.0) continue;

        s->dx[a] += xd * f;
        s->dy[a] += yd * f;
        s->dx[b] -= xd * f;
        s->dy[b] -= yd * f;
    }
}

/* Attraction exerted on node i by its neighbours. Each edge is computed
 * twice, once from each end, but the nodes can be processed in parallel. A
 * node receives the same contributions in the same order as in attract_all(),
 * so both give identical results. */
static void attract_node(const fa2_state *s, double coef, int i)
{
    double fx = s->dx[i], fy = s->dy[i];
    for (int k = s->inc_off[i]; k < s->inc_off[i + 1]; k++) {
        const int e = s->inc[k] >> 1;
        const int a = s->from[e], b = s->to[e];
        const double xd = s->x[a] - s->x[b], yd = s->y[a] - s->y[b];
        const double f = attraction_factor(s, e, coef, xd, yd);
        if (f == 0.0) continue;
        if (s->inc[k] & 1) {
            fx -= xd * f;
            fy -= yd * f;
        } else {
            fx += xd * f;
            fy += yd * f;
        }
    }
    s->dx[i] = fx;
    s->dy[i] = fy;
}

/* Adaptive global speed: the nodes go faster as long as they do not swing
 * (change direction between two iterations) more than they travel. */
void fa2_update_speed(int n, double total_swinging, double total_traction,
                      double jitter_tolerance, double *speed_,
                      double *speed_efficiency_)
{
    double speed = *speed_, speed_efficiency = *speed_efficiency_;

    /* Nothing moves at all: keep the current speed. */
    if (!(total_swinging > 0)) return;

    const double est_jt = 0.05 * sqrt((double) n);
    const double min_jt = sqrt(est_jt), max_jt = 10.0;
    double jt = est_jt * total_traction / ((double) n * (double) n);
    if (jt > max_jt) jt = max_jt;
    if (jt < min_jt) jt = min_jt;
    jt *= jitter_tolerance;

    const double min_speed_efficiency = 0.05;

    /* Protection against erratic behaviour */
    if (total_swinging > 2.0 * total_traction) {
        if (speed_efficiency > min_speed_efficiency) speed_efficiency *= 0.5;
        if (jt < jitter_tolerance) jt = jitter_tolerance;
    }

    const double target_speed =
        jt * speed_efficiency * total_traction / total_swinging;

    if (total_swinging > jt * total_traction) {
        if (speed_efficiency > min_speed_efficiency) speed_efficiency *= 0.7;
    } else if (speed < 1000) {
        speed_efficiency *= 1.3;
    }

    /* The speed may not rise by more than 50% per iteration. */
    const double max_rise = 0.5;
    const double rise = target_speed - speed;
    speed += rise < max_rise * speed ? rise : max_rise * speed;

    *speed_ = speed;
    *speed_efficiency_ = speed_efficiency;
}

/* ------------------------------------------------------------------------ */
/* Entry point                                                              */
/* ------------------------------------------------------------------------ */

/*
 * n_          number of nodes
 * from_, to_  0-based integer vectors of edge endpoints
 * w_          edge weights (already raised to the edge weight influence)
 * pos_        n x 2 matrix of initial positions
 * size_       node radii (length n)
 * fixed_      logical vector (length n), TRUE for nodes that must not move
 * iter_       number of iterations
 * dpar_       c(scaling, gravity, jitter tolerance, theta)
 * flags_      c(strong gravity, linlog, outbound distribution, adjust sizes,
 *               Barnes-Hut)
 * threads_    number of OpenMP threads
 * gpu_        c(device, double precision): the 0-based index of the OpenCL
 *             device that computes the repulsion, or -1 to use the CPU, and
 *             whether that device computes in double precision
 */

/* R_CheckUserInterrupt() does not return when the user interrupts, which
 * would leak the resources held on the GPU. This reports the interruption
 * instead, so that they can be released first. */
static void check_interrupt_fn(void *dummy)
{
    (void) dummy;
    R_CheckUserInterrupt();
}
static int user_interrupted(void)
{
    return R_ToplevelExec(check_interrupt_fn, NULL) == FALSE;
}

SEXP C_forceatlas2(SEXP n_, SEXP from_, SEXP to_, SEXP w_, SEXP pos_,
                   SEXP size_, SEXP fixed_, SEXP iter_, SEXP dpar_,
                   SEXP flags_, SEXP threads_, SEXP gpu_)
{
    const int n = asInteger(n_);
    const int m = LENGTH(from_);
    const int iterations = asInteger(iter_);
    int threads = asInteger(threads_);

    if (n == NA_INTEGER || n < 0) error("invalid number of nodes");
    if (!isInteger(from_) || !isInteger(to_) || LENGTH(to_) != m)
        error("invalid edge list");
    if (!isReal(w_) || LENGTH(w_) != m) error("invalid edge weights");
    if (!isReal(pos_) || XLENGTH(pos_) != 2 * (R_xlen_t) n)
        error("invalid initial positions");
    if (!isReal(size_) || LENGTH(size_) != n) error("invalid node sizes");
    if (!isLogical(fixed_) || LENGTH(fixed_) != n)
        error("invalid 'fixed' vector");
    if (!isReal(dpar_) || LENGTH(dpar_) != 4) error("invalid parameters");
    if (!isLogical(flags_) || LENGTH(flags_) != 5) error("invalid flags");
    if (iterations == NA_INTEGER || iterations < 0)
        error("invalid number of iterations");
    if (m > (1 << 30) - 1) error("too many edges");
    if (!isInteger(gpu_) || LENGTH(gpu_) != 2) error("invalid GPU settings");
    const int gpu_device = INTEGER(gpu_)[0];
    const int gpu_double = INTEGER(gpu_)[1] != 0;
    if (threads == NA_INTEGER || threads < 1) threads = 1;
#ifndef _OPENMP
    threads = 1;
#endif

    const int *from = INTEGER(from_), *to = INTEGER(to_);
    for (int e = 0; e < m; e++)
        if (from[e] < 0 || from[e] >= n || to[e] < 0 || to[e] >= n)
            error("edge %d refers to a node that does not exist", e + 1);

    SEXP ans = PROTECT(duplicate(pos_));
    if (n == 0 || iterations == 0) {
        UNPROTECT(1);
        return ans;
    }

    fa2_state s;
    s.n = n;
    s.m = m;
    s.x = REAL(ans);
    s.y = REAL(ans) + n;
    s.from = from;
    s.to = to;
    s.w = REAL(w_);
    s.size = REAL(size_);
    s.scaling = REAL(dpar_)[0];
    s.gravity = REAL(dpar_)[1];
    const double jitter_tolerance = REAL(dpar_)[2];
    s.theta = REAL(dpar_)[3];
    s.strong = LOGICAL(flags_)[0];
    s.linlog = LOGICAL(flags_)[1];
    s.outbound = LOGICAL(flags_)[2];
    s.adjust = LOGICAL(flags_)[3];
    const int barnes_hut = LOGICAL(flags_)[4];
    const int *fixed = LOGICAL(fixed_);

    /* R_alloc'ed memory is reclaimed by R at the end of .Call(), including
     * when the user interrupts the computation. */
    double *mass = (double *) R_alloc(n, sizeof(double));
    double *old_dx = (double *) R_alloc(n, sizeof(double));
    double *old_dy = (double *) R_alloc(n, sizeof(double));
    s.dx = (double *) R_alloc(n, sizeof(double));
    s.dy = (double *) R_alloc(n, sizeof(double));
    s.mass = mass;

    /* As in Gephi, the mass of a node is its degree plus one. */
    double mean_mass = 0.0;
    for (int i = 0; i < n; i++) mass[i] = 1.0;
    for (int e = 0; e < m; e++) {
        mass[from[e]] += 1.0;
        mass[to[e]] += 1.0;
    }
    for (int i = 0; i < n; i++) {
        mean_mass += mass[i];
        s.dx[i] = s.dy[i] = 0.0;
    }
    mean_mass /= n;
    double *swinging = (double *) R_alloc(n, sizeof(double));
    double *traction = (double *) R_alloc(n, sizeof(double));

    /* With several threads, the attraction is computed node by node, which
     * needs the list of the edges of each node (self-loops have no effect). */
    s.inc_off = NULL;
    s.inc = NULL;
    if (threads > 1 && m > 0) {
        int *off = (int *) R_alloc((size_t) n + 1, sizeof(int));
        int *inc = (int *) R_alloc(2 * (size_t) m, sizeof(int));
        memset(off, 0, ((size_t) n + 1) * sizeof(int));
        for (int e = 0; e < m; e++) {
            if (from[e] == to[e]) continue;
            off[from[e] + 1]++;
            off[to[e] + 1]++;
        }
        for (int i = 0; i < n; i++) off[i + 1] += off[i];
        /* off[i] is used as the insertion point of node i, then restored. */
        for (int e = 0; e < m; e++) {
            if (from[e] == to[e]) continue;
            inc[off[from[e]]++] = 2 * e;
            inc[off[to[e]]++] = 2 * e + 1;
        }
        for (int i = n; i > 0; i--) off[i] = off[i - 1];
        off[0] = 0;
        s.inc_off = off;
        s.inc = inc;
    }

    fa2_tree tree;
    memset(&tree, 0, sizeof(tree));
    int *stacks = NULL;
    size_t stack_len = 0;
    if (barnes_hut) {
        /* Every internal region has at least two children, hence there are
         * at most 2n - 1 regions. */
        stack_len = 2 * (size_t) n;
        tree.reg = (fa2_region *) R_alloc(stack_len, sizeof(fa2_region));
        tree.perm = (int *) R_alloc(n, sizeof(int));
        tree.buf[0] = (fa2_node *) R_alloc(n, sizeof(fa2_node));
        tree.buf[1] = (fa2_node *) R_alloc(n, sizeof(fa2_node));
        tree.cnt = (int *) R_alloc(4 * (size_t) n, sizeof(int));
        tree.px = (double *) R_alloc(n, sizeof(double));
        tree.py = (double *) R_alloc(n, sizeof(double));
        tree.pm = (double *) R_alloc(n, sizeof(double));
        tree.ps = s.adjust ? (double *) R_alloc(n, sizeof(double)) : NULL;
        if (gpu_device < 0)
            stacks = (int *) R_alloc(stack_len * (size_t) threads, sizeof(int));
    }

    /* From here on, no R error may be raised while `gpu` is open. */
    fa2_gpu *gpu = NULL;
    if (gpu_device >= 0) {
        gpu = fa2_gpu_open(gpu_device, gpu_double, n, s.adjust, barnes_hut,
                           threads);
        if (!gpu) error("%s", fa2_gpu_error());
    }

    const double attraction_coef = s.outbound ? mean_mass : 1.0;
    double speed = 1.0, speed_efficiency = 1.0;
    double work = 0.0;
    double time_tree = 0.0, time_repulsion = 0.0;
    const double time_start = fa2_now();

    for (int it = 0; it < iterations; it++) {
        /* Let the user interrupt long computations. */
        work += (double) n + (double) m;
        if (work >= 1e6) {
            if (gpu) {
                if (user_interrupted()) {
                    fa2_gpu_close(gpu);
                    error("the layout was interrupted");
                }
            } else {
                R_CheckUserInterrupt();
            }
            work = 0.0;
        }

        /* The forces of the previous iteration are kept. */
        {
            double *tmp = old_dx;
            old_dx = s.dx;
            s.dx = tmp;
            tmp = old_dy;
            old_dy = s.dy;
            s.dy = tmp;
        }
        memset(s.dx, 0, (size_t) n * sizeof(double));
        memset(s.dy, 0, (size_t) n * sizeof(double));

        /* Repulsion */
        double t0 = fa2_now(), t1;
        if (barnes_hut) {
            tree_build(&tree, n, s.x, s.y, mass, s.adjust ? s.size : NULL,
                       threads);
            t1 = fa2_now();
            time_tree += t1 - t0;
            t0 = t1;
        }
        if (gpu) {
            const int failed =
                barnes_hut ? fa2_gpu_repulse_bh(gpu, &tree, s.scaling, s.theta,
                                                s.dx, s.dy)
                           : fa2_gpu_repulse_exact(gpu, s.x, s.y, mass,
                                                   s.adjust ? s.size : NULL,
                                                   s.scaling, s.dx, s.dy);
            if (failed) {
                fa2_gpu_close(gpu);
                error("%s", fa2_gpu_error());
            }
        } else if (barnes_hut) {
            /* Nodes are visited in the order of the tree: consecutive nodes
             * are neighbours in space and traverse the same regions, which
             * makes a much better use of the CPU caches. */
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic, 64) if (threads > 1)
#endif
            for (int k = 0; k < n; k++) {
#ifdef _OPENMP
                int *stack = stacks + stack_len * (size_t) omp_get_thread_num();
#else
                int *stack = stacks;
#endif
                repulse_node_bh(&s, &tree, tree.perm[k], stack);
            }
        } else if (threads > 1) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
            for (int i = 0; i < n; i++) repulse_node_exact(&s, i);
        } else {
            repulse_all_exact_symmetric(&s);
        }
        time_repulsion += fa2_now() - t0;

        /* Gravity */
        if (s.gravity != 0) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static) if (threads > 1)
#endif
            for (int i = 0; i < n; i++) gravity_node(&s, i);
        }

        /* Attraction */
        if (s.inc) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic, 256)
#endif
            for (int i = 0; i < n; i++) attract_node(&s, attraction_coef, i);
        } else {
            attract_all(&s, attraction_coef);
        }

        /* Adaptive global speed: how much the nodes swing (change direction
         * between two iterations) compared to how much they travel. The sums
         * are always made in the same order, whatever the number of threads. */
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static) if (threads > 1)
#endif
        for (int i = 0; i < n; i++) {
            const double sx = old_dx[i] - s.dx[i], sy = old_dy[i] - s.dy[i];
            const double tx = old_dx[i] + s.dx[i], ty = old_dy[i] + s.dy[i];
            swinging[i] = mass[i] * sqrt(sx * sx + sy * sy);
            traction[i] = 0.5 * mass[i] * sqrt(tx * tx + ty * ty);
        }
        double total_swinging = 0.0, total_traction = 0.0;
        for (int i = 0; i < n; i++) {
            if (fixed[i]) continue;
            total_swinging += swinging[i];
            total_traction += traction[i];
        }

        fa2_update_speed(n, total_swinging, total_traction, jitter_tolerance,
                         &speed, &speed_efficiency);

        /* Apply the forces */
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static) if (threads > 1)
#endif
        for (int i = 0; i < n; i++) {
            if (fixed[i]) continue;
            double factor;
            if (s.adjust) {
                /* With overlap prevention the swinging measure is less
                 * reliable: move slower and bound the displacement. */
                const double df = sqrt(s.dx[i] * s.dx[i] + s.dy[i] * s.dy[i]);
                factor = 0.1 * speed / (1.0 + sqrt(speed * swinging[i]));
                if (df > 0) {
                    if (factor * df > 10.0) factor = 10.0 / df;
                } else {
                    factor = 0.0;
                }
            } else {
                factor = speed / (1.0 + sqrt(speed * swinging[i]));
            }
            s.x[i] += s.dx[i] * factor;
            s.y[i] += s.dy[i] * factor;
        }
    }

    /* Where the time went, in seconds, for diagnostic purposes. */
    double gpu_times[3] = {0.0, 0.0, 0.0};
    if (gpu) fa2_gpu_times(gpu, gpu_times);
    fa2_gpu_close(gpu);

    const char *time_names[] = {"total", "tree", "repulsion", "gpu_send",
                                "gpu_compute", "gpu_fetch", "other"};
    SEXP times = PROTECT(allocVector(REALSXP, 7));
    SEXP names = PROTECT(allocVector(STRSXP, 7));
    for (int i = 0; i < 7; i++) SET_STRING_ELT(names, i, mkChar(time_names[i]));
    REAL(times)[0] = fa2_now() - time_start;
    REAL(times)[1] = time_tree;
    REAL(times)[2] = time_repulsion;
    REAL(times)[3] = gpu_times[0];
    REAL(times)[4] = gpu_times[1];
    REAL(times)[5] = gpu_times[2];
    REAL(times)[6] = REAL(times)[0] - time_tree - time_repulsion;
    setAttrib(times, R_NamesSymbol, names);
    setAttrib(ans, install("timings"), times);

    UNPROTECT(3);
    return ans;
}

/*
 * The all-GPU layout. The arguments are those of C_forceatlas2, except:
 * flags_      c(strong gravity, linlog, outbound distribution, adjust sizes)
 * gpu_        c(device, double precision, profile)
 * source_     source of the OpenCL kernels
 */
SEXP C_forceatlas2_large(SEXP n_, SEXP from_, SEXP to_, SEXP w_, SEXP pos_,
                         SEXP size_, SEXP fixed_, SEXP iter_, SEXP dpar_,
                         SEXP flags_, SEXP gpu_, SEXP source_)
{
    const int n = asInteger(n_);
    const int m = LENGTH(from_);
    const int iterations = asInteger(iter_);

    if (n == NA_INTEGER || n < 2) error("invalid number of nodes");
    if (!isInteger(from_) || !isInteger(to_) || LENGTH(to_) != m)
        error("invalid edge list");
    if (!isReal(w_) || LENGTH(w_) != m) error("invalid edge weights");
    if (!isReal(pos_) || XLENGTH(pos_) != 2 * (R_xlen_t) n)
        error("invalid initial positions");
    if (!isReal(size_) || LENGTH(size_) != n) error("invalid node sizes");
    if (!isLogical(fixed_) || LENGTH(fixed_) != n)
        error("invalid 'fixed' vector");
    if (!isReal(dpar_) || LENGTH(dpar_) != 4) error("invalid parameters");
    if (!isLogical(flags_) || LENGTH(flags_) != 4) error("invalid flags");
    if (!isInteger(gpu_) || LENGTH(gpu_) != 3) error("invalid GPU settings");
    if (!isString(source_) || LENGTH(source_) != 1)
        error("invalid kernel source");
    if (iterations == NA_INTEGER || iterations < 0)
        error("invalid number of iterations");
    if (m > (1 << 30) - 1) error("too many edges");

    const int *from = INTEGER(from_), *to = INTEGER(to_);
    for (int e = 0; e < m; e++)
        if (from[e] < 0 || from[e] >= n || to[e] < 0 || to[e] >= n)
            error("edge %d refers to a node that does not exist", e + 1);

    SEXP ans = PROTECT(duplicate(pos_));
    const int outbound = LOGICAL(flags_)[2];
    const double *w = REAL(w_);

    /* As in Gephi, the mass of a node is its degree plus one. */
    double *mass = (double *) R_alloc(n, sizeof(double));
    double mean_mass = 0.0;
    for (int i = 0; i < n; i++) mass[i] = 1.0;
    for (int e = 0; e < m; e++) {
        mass[from[e]] += 1.0;
        mass[to[e]] += 1.0;
    }
    for (int i = 0; i < n; i++) mean_mass += mass[i];
    mean_mass /= n;
    const double coef = outbound ? mean_mass : 1.0;

    /* The edges of each node (self-loops have no effect). */
    int *off = (int *) R_alloc((size_t) n + 1, sizeof(int));
    int *inc_node = (int *) R_alloc(2 * (size_t) m + 1, sizeof(int));
    double *inc_coef = (double *) R_alloc(2 * (size_t) m + 1, sizeof(double));
    memset(off, 0, ((size_t) n + 1) * sizeof(int));
    for (int e = 0; e < m; e++) {
        if (from[e] == to[e]) continue;
        off[from[e] + 1]++;
        off[to[e] + 1]++;
    }
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    for (int e = 0; e < m; e++) {
        if (from[e] == to[e]) continue;
        const double c = outbound ? coef * w[e] / mass[from[e]] : coef * w[e];
        inc_node[off[from[e]]] = to[e];
        inc_coef[off[from[e]]++] = c;
        inc_node[off[to[e]]] = from[e];
        inc_coef[off[to[e]]++] = c;
    }
    for (int i = n; i > 0; i--) off[i] = off[i - 1];
    off[0] = 0;

    fa2_full_params p;
    memset(&p, 0, sizeof(p));
    p.n = n;
    p.inc_off = off;
    p.inc_node = inc_node;
    p.inc_coef = inc_coef;
    p.mass = mass;
    p.size = REAL(size_);
    p.fixed = LOGICAL(fixed_);
    for (int i = 0; i < n; i++)
        if (p.fixed[i]) p.any_fixed = 1;
    p.scaling = REAL(dpar_)[0];
    p.gravity = REAL(dpar_)[1];
    p.jitter_tolerance = REAL(dpar_)[2];
    p.theta = REAL(dpar_)[3];
    p.strong = LOGICAL(flags_)[0];
    p.linlog = LOGICAL(flags_)[1];
    p.adjust = LOGICAL(flags_)[3];
    p.iterations = iterations;
    p.device = INTEGER(gpu_)[0];
    p.use_double = INTEGER(gpu_)[1] != 0;
    p.profile = INTEGER(gpu_)[2] != 0;
    p.source = CHAR(STRING_ELT(source_, 0));
    p.interrupted = user_interrupted;

    /* fa2_full_run() does not use R: it cannot be interrupted by an R error
     * while it holds resources on the GPU. */
    double t[FA2_FULL_NTIMES];
    if (fa2_full_run(&p, REAL(ans), t)) error("%s", fa2_gpu_error());

    const char *time_names[FA2_FULL_NTIMES] = {
        "total", "codes", "sort", "tree", "repulsion", "attraction", "move"};
    SEXP times = PROTECT(allocVector(REALSXP, FA2_FULL_NTIMES));
    SEXP names = PROTECT(allocVector(STRSXP, FA2_FULL_NTIMES));
    for (int i = 0; i < FA2_FULL_NTIMES; i++) {
        REAL(times)[i] = t[i];
        SET_STRING_ELT(names, i, mkChar(time_names[i]));
    }
    setAttrib(times, R_NamesSymbol, names);
    setAttrib(ans, install("timings"), times);

    UNPROTECT(3);
    return ans;
}

/* The OpenCL devices of this computer, as a list of columns. */
SEXP C_gpu_devices(void)
{
    const int nd = fa2_gpu_ndevices();
    const char *names[] = {"name", "vendor", "platform", "version",
                           "type", "memory_gb", "double", ""};
    SEXP ans = PROTECT(mkNamed(VECSXP, names));
    SEXP name = PROTECT(allocVector(STRSXP, nd));
    SEXP vendor = PROTECT(allocVector(STRSXP, nd));
    SEXP platform = PROTECT(allocVector(STRSXP, nd));
    SEXP version = PROTECT(allocVector(STRSXP, nd));
    SEXP type = PROTECT(allocVector(STRSXP, nd));
    SEXP memory = PROTECT(allocVector(REALSXP, nd));
    SEXP dbl = PROTECT(allocVector(LGLSXP, nd));
    for (int i = 0; i < nd; i++) {
        fa2_gpu_devinfo info;
        memset(&info, 0, sizeof(info));
        fa2_gpu_device_info(i, &info);
        SET_STRING_ELT(name, i, mkChar(info.name));
        SET_STRING_ELT(vendor, i, mkChar(info.vendor));
        SET_STRING_ELT(platform, i, mkChar(info.platform));
        SET_STRING_ELT(version, i, mkChar(info.version));
        SET_STRING_ELT(type, i, mkChar(info.is_gpu ? "GPU" : "other"));
        REAL(memory)[i] = info.memory / 1073741824.0;
        LOGICAL(dbl)[i] = info.has_double;
    }
    SET_VECTOR_ELT(ans, 0, name);
    SET_VECTOR_ELT(ans, 1, vendor);
    SET_VECTOR_ELT(ans, 2, platform);
    SET_VECTOR_ELT(ans, 3, version);
    SET_VECTOR_ELT(ans, 4, type);
    SET_VECTOR_ELT(ans, 5, memory);
    SET_VECTOR_ELT(ans, 6, dbl);
    UNPROTECT(8);
    return ans;
}

/* Whether the package was compiled with OpenMP support. */
SEXP C_has_openmp(void)
{
#ifdef _OPENMP
    return ScalarLogical(1);
#else
    return ScalarLogical(0);
#endif
}

static const R_CallMethodDef call_methods[] = {
    {"C_forceatlas2", (DL_FUNC) &C_forceatlas2, 12},
    {"C_forceatlas2_large", (DL_FUNC) &C_forceatlas2_large, 12},
    {"C_gpu_devices", (DL_FUNC) &C_gpu_devices, 0},
    {"C_has_openmp", (DL_FUNC) &C_has_openmp, 0},
    {NULL, NULL, 0}};

void R_init_forceatlas2r(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, call_methods, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
    R_forceSymbols(dll, TRUE);
}

void R_unload_forceatlas2r(DllInfo *dll)
{
    (void) dll;
    fa2_gpu_shutdown();
}

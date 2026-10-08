/*
 * ForceAtlas2 entirely on the GPU: OpenCL kernels of forceatlas2_large().
 *
 * One iteration is the following sequence. Everything stays on the device;
 * the host only reads back two numbers (step 7) to adjust the global speed.
 *
 *   1. bbox_partial, bbox_final   bounding box of the nodes
 *   2. morton                     Morton code of each node (16 bits per axis)
 *   3. sort_hist, sort_scan,      radix sort of the nodes by Morton code,
 *      sort_scatter               four passes of 8 bits
 *   4. gather, tree, aggregate,   binary radix tree over the sorted nodes
 *      finalize                   (Karras 2012), with the mass, the centre of
 *                                 mass and the bounding box of every region
 *   5. repulse                    Barnes-Hut repulsion: one traversal of the
 *                                 tree per node
 *   6. attract                    gravity, and attraction along the edges
 *   7. swing, sum2                swinging and traction of the nodes
 *   8. apply                      displacement of the nodes
 *
 * Arrays indexed by node hold the nodes in their original order; `psort`
 * holds them in the order of the tree. Several kernels process the nodes by
 * blocks of S consecutive nodes, one work item per block, so that sums are
 * always made in the same order and the results are reproducible.
 *
 * Tree layout, for n nodes: the regions (internal nodes of the tree) are
 * numbered 0 .. n-2, region 0 being the root, and the nodes themselves
 * (leaves) n-1 .. 2n-2 in sorted order.
 */

#ifdef FA2_DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double real;
#define REAL(x) x
#else
typedef float real;
#define REAL(x) x##f
#endif
#pragma OPENCL FP_CONTRACT OFF

/* Regions with at most this many nodes are not opened node by node: when one
 * is too close to be approximated, its nodes, which are contiguous in psort,
 * are visited directly. */
#define FA2_LEAF_SIZE 8

/* Repulsion factor between two nodes (see src/forceatlas2.c). */
real rep_factor(real d2, real mm, real ss, real scaling, int adjust)
{
    if (adjust) {
        const real d = sqrt(d2) - ss;
        if (d > (real) 0) return scaling * mm / (d * d);
        if (d < (real) 0) return (real) 100 * scaling * mm;
        return (real) 0;
    }
    return d2 > (real) 0 ? scaling * mm / d2 : (real) 0;
}

/* ------------------------------------------------------------------------ */
/* 1. Bounding box                                                          */
/* ------------------------------------------------------------------------ */

__kernel void bbox_partial(const int n, const int S, __global const real *pos,
                           __global real *partial)
{
    const int b = (int) get_global_id(0);
    const int lo = b * S;
    const int hi = min(n, lo + S);
    if (lo >= n) return;
    real minx = pos[2 * lo], maxx = minx;
    real miny = pos[2 * lo + 1], maxy = miny;
    for (int i = lo + 1; i < hi; i++) {
        const real x = pos[2 * i], y = pos[2 * i + 1];
        minx = min(minx, x);
        maxx = max(maxx, x);
        miny = min(miny, y);
        maxy = max(maxy, y);
    }
    partial[4 * b] = minx;
    partial[4 * b + 1] = miny;
    partial[4 * b + 2] = maxx;
    partial[4 * b + 3] = maxy;
}

__kernel void bbox_final(const int B, __global const real *partial,
                         __global real *bbox)
{
    if (get_global_id(0) != 0) return;
    real minx = partial[0], miny = partial[1];
    real maxx = partial[2], maxy = partial[3];
    for (int b = 1; b < B; b++) {
        minx = min(minx, partial[4 * b]);
        miny = min(miny, partial[4 * b + 1]);
        maxx = max(maxx, partial[4 * b + 2]);
        maxy = max(maxy, partial[4 * b + 3]);
    }
    bbox[0] = minx;
    bbox[1] = miny;
    bbox[2] = maxx;
    bbox[3] = maxy;
}

/* ------------------------------------------------------------------------ */
/* 2. Morton codes                                                          */
/* ------------------------------------------------------------------------ */

/* Insert a zero bit before each of the 16 low bits of v. */
uint spread_bits(uint v)
{
    v &= 0x0000FFFFu;
    v = (v | (v << 8)) & 0x00FF00FFu;
    v = (v | (v << 4)) & 0x0F0F0F0Fu;
    v = (v | (v << 2)) & 0x33333333u;
    v = (v | (v << 1)) & 0x55555555u;
    return v;
}

__kernel void morton(const int n, __global const real *pos,
                     __global const real *bbox, __global uint *key,
                     __global uint *idx)
{
    const int i = (int) get_global_id(0);
    if (i >= n) return;
    const real minx = bbox[0], miny = bbox[1];
    const real extent = max(bbox[2] - minx, bbox[3] - miny);
    const real scale = extent > (real) 0 ? (real) 65535 / extent : (real) 0;
    const real fx = clamp((pos[2 * i] - minx) * scale, (real) 0, (real) 65535);
    const real fy =
        clamp((pos[2 * i + 1] - miny) * scale, (real) 0, (real) 65535);
    key[i] = spread_bits((uint) fx) | (spread_bits((uint) fy) << 1);
    idx[i] = (uint) i;
}

/* ------------------------------------------------------------------------ */
/* 3. Radix sort (one pass sorts on the 8 bits starting at `shift`)         */
/* ------------------------------------------------------------------------ */

/* hist[d * B + b]: number of keys of block b whose digit is d. */
__kernel void sort_hist(const int n, const int S, const int B, const int shift,
                        __global const uint *key, __global uint *hist)
{
    const int b = (int) get_global_id(0);
    const int lo = b * S;
    const int hi = min(n, lo + S);
    if (b >= B) return;
    uint h[256];
    for (int d = 0; d < 256; d++) h[d] = 0;
    for (int i = lo; i < hi; i++) h[(key[i] >> shift) & 255u]++;
    for (int d = 0; d < 256; d++) hist[d * B + b] = h[d];
}

/* For each digit d: hist[d * B + b] becomes the number of keys with digit d
 * in the blocks before b, and total[d] the number of keys with digit d. */
__kernel void sort_scan(const int B, __global uint *hist, __global uint *total)
{
    const int d = (int) get_global_id(0);
    if (d >= 256) return;
    uint sum = 0;
    for (int b = 0; b < B; b++) {
        const uint count = hist[d * B + b];
        hist[d * B + b] = sum;
        sum += count;
    }
    total[d] = sum;
}

__kernel void sort_scatter(const int n, const int S, const int B,
                           const int shift, __global const uint *key,
                           __global const uint *idx,
                           __global const uint *hist,
                           __global const uint *total, __global uint *key_out,
                           __global uint *idx_out)
{
    const int b = (int) get_global_id(0);
    const int lo = b * S;
    const int hi = min(n, lo + S);
    if (b >= B) return;
    uint offset[256];
    uint base = 0;
    for (int d = 0; d < 256; d++) {
        offset[d] = base + hist[d * B + b];
        base += total[d];
    }
    for (int i = lo; i < hi; i++) {
        const uint k = key[i];
        const uint p = offset[(k >> shift) & 255u]++;
        key_out[p] = k;
        idx_out[p] = idx[i];
    }
}

/* ------------------------------------------------------------------------ */
/* 4. Tree                                                                  */
/* ------------------------------------------------------------------------ */

/* psort: x, y, mass and radius of the nodes, in sorted order. */
__kernel void gather(const int n, const int adjust,
                     __global const uint *order, __global const real *pos,
                     __global const real *mass, __global const real *size,
                     __global real *psort)
{
    const int k = (int) get_global_id(0);
    if (k >= n) return;
    const uint i = order[k];
    psort[4 * k] = pos[2 * i];
    psort[4 * k + 1] = pos[2 * i + 1];
    psort[4 * k + 2] = mass[i];
    psort[4 * k + 3] = adjust ? size[i] : (real) 0;
}

/* Length of the common prefix of the keys of the sorted nodes i and j, or -1
 * if j is not a node. Equal keys are told apart by their position. */
int common_prefix(__global const uint *key, int n, int i, int j)
{
    if (j < 0 || j >= n) return -1;
    const uint a = key[i], b = key[j];
    if (a == b) return 32 + (int) clz((uint) (i ^ j));
    return (int) clz(a ^ b);
}

/* Binary radix tree of Karras (2012), "Maximizing Parallelism in the
 * Construction of BVHs, Octrees, and k-d Trees": each region is found
 * independently of the others.
 *   child[2r], child[2r+1]   the two children of region r
 *   range[2r], range[2r+1]   first and last sorted node of region r
 *   parent[c]                the region that c (region or node) belongs to
 *   visits[r]                reset here, used by `aggregate`              */
__kernel void tree(const int n, __global const uint *key, __global int *child,
                   __global int *range, __global int *parent,
                   __global int *visits)
{
    const int i = (int) get_global_id(0);
    if (i >= n - 1) return;

    /* Direction of the range of keys covered by the region */
    const int d = common_prefix(key, n, i, i + 1) >
                          common_prefix(key, n, i, i - 1)
                      ? 1
                      : -1;

    /* Upper bound of its length, then its length by bisection */
    const int prefix_min = common_prefix(key, n, i, i - d);
    int lmax = 2;
    while (common_prefix(key, n, i, i + lmax * d) > prefix_min) lmax *= 2;
    int l = 0;
    for (int t = lmax / 2; t >= 1; t /= 2)
        if (common_prefix(key, n, i, i + (l + t) * d) > prefix_min) l += t;
    const int j = i + l * d;

    /* Where the range is split between the two children */
    const int prefix_node = common_prefix(key, n, i, j);
    int s = 0;
    int t = l;
    do {
        t = (t + 1) >> 1;
        if (common_prefix(key, n, i, i + (s + t) * d) > prefix_node) s += t;
    } while (t > 1);
    const int split = i + s * d + min(d, 0);

    const int first = min(i, j), last = max(i, j);
    const int left = first == split ? n - 1 + split : split;
    const int right = last == split + 1 ? n - 1 + split + 1 : split + 1;

    child[2 * i] = left;
    child[2 * i + 1] = right;
    range[2 * i] = first;
    range[2 * i + 1] = last;
    parent[left] = i;
    parent[right] = i;
    if (i == 0) parent[0] = -1;
    visits[i] = 0;
}

/* nd, for each region and node: bounding box (min x, min y, max x, max y),
 * mass, and sums of mass * x and mass * y. Each node goes up the tree from
 * its leaf; a region is computed by whichever of its two children arrives
 * second, when both of them are complete. */
__kernel void aggregate(const int n, __global const real *psort,
                        __global const int *child, __global const int *parent,
                        volatile __global int *visits, __global real *nd)
{
    const int k = (int) get_global_id(0);
    if (k >= n) return;
    const int leaf = n - 1 + k;
    const real x = psort[4 * k], y = psort[4 * k + 1], m = psort[4 * k + 2];
    nd[8 * leaf] = x;
    nd[8 * leaf + 1] = y;
    nd[8 * leaf + 2] = x;
    nd[8 * leaf + 3] = y;
    nd[8 * leaf + 4] = m;
    nd[8 * leaf + 5] = x * m;
    nd[8 * leaf + 6] = y * m;

    int r = parent[leaf];
    while (r >= 0) {
        if (atomic_inc(&visits[r]) == 0) return;
        const int a = child[2 * r], b = child[2 * r + 1];
        nd[8 * r] = min(nd[8 * a], nd[8 * b]);
        nd[8 * r + 1] = min(nd[8 * a + 1], nd[8 * b + 1]);
        nd[8 * r + 2] = max(nd[8 * a + 2], nd[8 * b + 2]);
        nd[8 * r + 3] = max(nd[8 * a + 3], nd[8 * b + 3]);
        nd[8 * r + 4] = nd[8 * a + 4] + nd[8 * b + 4];
        nd[8 * r + 5] = nd[8 * a + 5] + nd[8 * b + 5];
        nd[8 * r + 6] = nd[8 * a + 6] + nd[8 * b + 6];
        r = parent[r];
    }
}

/* ni, for each region: centre of mass (x, y), mass, and squared size. The
 * size of a region is the diagonal of its bounding box times sqrt(2), which
 * makes a given theta about as precise as it is with the regions of Gephi
 * (whose size is twice the largest distance between the centre of mass and a
 * node, i.e. between one and two diagonals). */
__kernel void finalize(const int n, __global const real *nd, __global real *ni)
{
    const int r = (int) get_global_id(0);
    if (r >= n - 1) return;
    const real m = nd[8 * r + 4];
    const real w = nd[8 * r + 2] - nd[8 * r];
    const real h = nd[8 * r + 3] - nd[8 * r + 1];
    ni[4 * r] = nd[8 * r + 5] / m;
    ni[4 * r + 1] = nd[8 * r + 6] / m;
    ni[4 * r + 2] = m;
    ni[4 * r + 3] = (real) 2 * (w * w + h * h);
}

/* ------------------------------------------------------------------------ */
/* 5. Repulsion                                                             */
/* ------------------------------------------------------------------------ */

/* A region is approximated by a single node placed at its centre of mass
 * when distance * theta > size. The force on each node is written to f, in
 * the original order of the nodes. */
__kernel void repulse(const int n, const real scaling, const real theta2,
                      const int adjust, __global const real *psort,
                      __global const real *ni, __global const int *child,
                      __global const int *range, __global const uint *order,
                      __global real *f)
{
    const int k = (int) get_global_id(0);
    if (k >= n) return;
    const real xi = psort[4 * k], yi = psort[4 * k + 1];
    const real mi = psort[4 * k + 2], si = psort[4 * k + 3];
    real ax = (real) 0, ay = (real) 0;

    /* The tree is at most 64 levels deep (32 bits of key, then the bits of
     * the positions), and going down one level adds one entry. */
    int stack[72];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const int r = stack[--sp];
        int first, last;
        if (r >= n - 1) {
            first = last = r - (n - 1); /* a single node */
        } else {
            const real xd = xi - ni[4 * r], yd = yi - ni[4 * r + 1];
            const real d2 = xd * xd + yd * yd;
            if (d2 * theta2 > ni[4 * r + 3]) {
                const real g = scaling * (mi * ni[4 * r + 2]) / d2;
                ax += xd * g;
                ay += yd * g;
                continue;
            }
            first = range[2 * r];
            last = range[2 * r + 1];
            if (last - first >= FA2_LEAF_SIZE) {
                stack[sp++] = child[2 * r + 1];
                stack[sp++] = child[2 * r];
                continue;
            }
        }
        for (int j = first; j <= last; j++) {
            if (j == k) continue;
            const real xd = xi - psort[4 * j], yd = yi - psort[4 * j + 1];
            const real g = rep_factor(xd * xd + yd * yd, mi * psort[4 * j + 2],
                                      si + psort[4 * j + 3], scaling, adjust);
            ax += xd * g;
            ay += yd * g;
        }
    }
    const uint i = order[k];
    f[2 * i] = ax;
    f[2 * i + 1] = ay;
}

/* ------------------------------------------------------------------------ */
/* 6. Gravity and attraction                                                */
/* ------------------------------------------------------------------------ */

/* The edges of node i are the entries inc_off[i] .. inc_off[i + 1] - 1 of
 * inc_node (the node at the other end) and inc_coef (the strength of the
 * attraction along that edge). */
__kernel void attract(const int n, const real gravity, const int strong,
                      const int linlog, const int adjust,
                      __global const real *pos, __global const real *mass,
                      __global const real *size, __global const int *inc_off,
                      __global const int *inc_node,
                      __global const real *inc_coef, __global real *f)
{
    const int i = (int) get_global_id(0);
    if (i >= n) return;
    const real xi = pos[2 * i], yi = pos[2 * i + 1];
    real ax = f[2 * i], ay = f[2 * i + 1];

    if (gravity != (real) 0) {
        const real dist = sqrt(xi * xi + yi * yi);
        if (dist > (real) 0) {
            const real g = strong ? gravity * mass[i] : gravity * mass[i] / dist;
            ax -= xi * g;
            ay -= yi * g;
        }
    }

    const real si = adjust ? size[i] : (real) 0;
    for (int e = inc_off[i]; e < inc_off[i + 1]; e++) {
        const int j = inc_node[e];
        const real xd = xi - pos[2 * j], yd = yi - pos[2 * j + 1];
        real g = -inc_coef[e];
        if (adjust || linlog) {
            real dist = sqrt(xd * xd + yd * yd);
            if (adjust) dist -= si + size[j];
            if (!(dist > (real) 0)) continue;
            if (linlog) g *= log1p(dist) / dist;
        }
        ax += xd * g;
        ay += yd * g;
    }
    f[2 * i] = ax;
    f[2 * i + 1] = ay;
}

/* ------------------------------------------------------------------------ */
/* 7. Swinging and traction                                                 */
/* ------------------------------------------------------------------------ */

/* sw[i]: how much node i swings, i.e. how much the force applied to it
 * changed since the previous iteration (f_old). partial[2b], partial[2b+1]:
 * total swinging and total traction of the nodes of block b that can move. */
__kernel void swing(const int n, const int S, const int use_fixed,
                    __global const real *f, __global const real *f_old,
                    __global const real *mass, __global const uchar *fixed,
                    __global real *sw, __global real *partial)
{
    const int b = (int) get_global_id(0);
    const int lo = b * S;
    const int hi = min(n, lo + S);
    if (lo >= n) return;
    real total_swinging = (real) 0, total_traction = (real) 0;
    for (int i = lo; i < hi; i++) {
        const real sx = f_old[2 * i] - f[2 * i];
        const real sy = f_old[2 * i + 1] - f[2 * i + 1];
        const real tx = f_old[2 * i] + f[2 * i];
        const real ty = f_old[2 * i + 1] + f[2 * i + 1];
        const real s = mass[i] * sqrt(sx * sx + sy * sy);
        sw[i] = s;
        if (use_fixed && fixed[i]) continue;
        total_swinging += s;
        total_traction += REAL(0.5) * mass[i] * sqrt(tx * tx + ty * ty);
    }
    partial[2 * b] = total_swinging;
    partial[2 * b + 1] = total_traction;
}

__kernel void sum2(const int B, __global const real *partial,
                   __global real *total)
{
    if (get_global_id(0) != 0) return;
    real a = (real) 0, b = (real) 0;
    for (int i = 0; i < B; i++) {
        a += partial[2 * i];
        b += partial[2 * i + 1];
    }
    total[0] = a;
    total[1] = b;
}

/* ------------------------------------------------------------------------ */
/* 8. Displacement                                                          */
/* ------------------------------------------------------------------------ */

__kernel void apply(const int n, const real speed, const int adjust,
                    const int use_fixed, __global const real *f,
                    __global real *f_old, __global const real *sw,
                    __global const uchar *fixed, __global real *pos)
{
    const int i = (int) get_global_id(0);
    if (i >= n) return;
    const real fx = f[2 * i], fy = f[2 * i + 1];
    f_old[2 * i] = fx;
    f_old[2 * i + 1] = fy;
    if (use_fixed && fixed[i]) return;

    real factor;
    if (adjust) {
        const real df = sqrt(fx * fx + fy * fy);
        factor = REAL(0.1) * speed / ((real) 1 + sqrt(speed * sw[i]));
        if (df > (real) 0) {
            if (factor * df > (real) 10) factor = (real) 10 / df;
        } else {
            factor = (real) 0;
        }
    } else {
        factor = speed / ((real) 1 + sqrt(speed * sw[i]));
    }
    pos[2 * i] += fx * factor;
    pos[2 * i + 1] += fy * factor;
}

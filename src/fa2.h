/*
 * Shared declarations for the ForceAtlas2 layout: Barnes-Hut tree and GPU
 * backend. See forceatlas2.c for the licence.
 */
#ifndef FA2_H
#define FA2_H

/*
 * Barnes-Hut regions are stored in one flat array. A region covers the nodes
 * perm[start .. start + count - 1]; its children (if any) are the regions
 * child .. child + nchild - 1. As in Gephi, a region is split in four around
 * its centre of mass and its "size" is twice the largest distance between
 * the centre of mass and one of its nodes. A region is approximated by a
 * single node of the same mass when distance * theta > size.
 *
 * On the GPU the tree is traversed without a stack: from a region, one either
 * goes down to its last child, or jumps to `skip`, the next region to visit
 * that is not one of its descendants: the previous sibling if there is one,
 * otherwise the `skip` of the parent (-1 when the traversal is over).
 */
typedef struct {
    double mass, cx, cy, size;
    int start, count;
    int child, nchild;
    int skip;
} fa2_region;

typedef struct {
    fa2_region *reg;
    int nreg;
    int *perm; /* node indices, grouped by region */
    int *tmp;  /* scratch space for partitioning */
    int *cnt;  /* scratch space: nodes per quadrant, for a level of the tree */
    /* Positions, masses and radii of the nodes in the order of perm, so that
     * the nodes of a region are contiguous in memory. */
    double *px, *py, *pm, *ps;
} fa2_tree;

/* ------------------------------------------------------------------------ */
/* GPU backend (gpu.c). Nothing here depends on R.                          */
/* ------------------------------------------------------------------------ */

typedef struct {
    char name[128];
    char vendor[128];
    char platform[128];
    char version[128];
    int is_gpu;
    int has_double;
    double memory; /* bytes */
} fa2_gpu_devinfo;

typedef struct fa2_gpu fa2_gpu;

/* Description of the last error. */
const char *fa2_gpu_error(void);

/* Number of OpenCL devices (0 if there is no OpenCL runtime). */
int fa2_gpu_ndevices(void);
/* Information about device `idx` (0-based); returns 0 on success. */
int fa2_gpu_device_info(int idx, fa2_gpu_devinfo *info);

/* Prepare device `device` for a layout of n nodes; NULL on error. */
/* `threads` is the number of CPU threads used to prepare the data. */
fa2_gpu *fa2_gpu_open(int device, int use_double, int n, int adjust,
                      int barnes_hut, int threads);
/* Time spent, in seconds, since the device was opened: sending the data,
 * computing on the device, and fetching the result. */
void fa2_gpu_times(const fa2_gpu *g, double times[3]);
void fa2_gpu_close(fa2_gpu *g);

/* Both add the repulsion forces to dx and dy and return 0 on success. `size`
 * may be NULL when node sizes are not used. */
int fa2_gpu_repulse_exact(fa2_gpu *g, const double *x, const double *y,
                          const double *mass, const double *size,
                          double scaling, double *dx, double *dy);
int fa2_gpu_repulse_bh(fa2_gpu *g, const fa2_tree *t, double scaling,
                       double theta, double *dx, double *dy);

/* Release everything that is cached between calls. */
void fa2_gpu_shutdown(void);

/* A monotonic clock, in seconds (ocl_loader.c). */
double fa2_now(void);

#endif

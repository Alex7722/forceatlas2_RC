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
    int buf; /* during construction: which buffer holds the nodes */
} fa2_region;

/* A node as it is moved around while the tree is built. */
typedef struct {
    double x, y, m;
    int idx;
} fa2_node;

typedef struct {
    fa2_region *reg;
    int nreg;
    int *perm; /* node indices, grouped by region */
    int *cnt;  /* scratch space: nodes per quadrant, for a level of the tree */
    fa2_node *buf[2]; /* scratch space: the nodes, grouped by region */
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

/* ------------------------------------------------------------------------ */
/* All-GPU layout (gpu_full.c)                                              */
/* ------------------------------------------------------------------------ */

typedef struct {
    int n;
    /* The edges of node i are the entries inc_off[i] .. inc_off[i + 1] - 1 of
     * inc_node (the node at the other end) and inc_coef (the strength of the
     * attraction along the edge). */
    const int *inc_off, *inc_node;
    const double *inc_coef;
    const double *mass;
    const double *size; /* node radii, used only if adjust */
    const int *fixed;   /* non-zero for the nodes that must not move */
    int any_fixed;
    double scaling, gravity, theta, jitter_tolerance;
    int strong, linlog, adjust;
    int iterations;
    int device, use_double;
    int profile;                /* measure the time spent in each phase */
    const char *source;         /* source of the OpenCL kernels */
    int (*interrupted)(void);   /* returns non-zero to stop the layout */
} fa2_full_params;

enum {
    FA2_FULL_T_TOTAL,
    FA2_FULL_T_CODES,
    FA2_FULL_T_SORT,
    FA2_FULL_T_TREE,
    FA2_FULL_T_REPULSION,
    FA2_FULL_T_ATTRACTION,
    FA2_FULL_T_MOVE,
    FA2_FULL_NTIMES
};

/* Run the layout. `pos` holds the x then the y coordinates of the nodes, and
 * receives the result; `times` (FA2_FULL_NTIMES values) the time spent, in
 * seconds. Returns 0 on success. */
int fa2_full_run(const fa2_full_params *p, double *pos, double *times);

/* Adaptive global speed of ForceAtlas2 (forceatlas2.c): update the speed
 * from the total swinging and the total traction of the nodes. */
void fa2_update_speed(int n, double total_swinging, double total_traction,
                      double jitter_tolerance, double *speed,
                      double *speed_efficiency);

#endif

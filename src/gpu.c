/*
 * GPU backend of the ForceAtlas2 layout (OpenCL).
 *
 * Only the repulsion, which is where nearly all the time goes, is computed on
 * the GPU. Two kernels are provided:
 *
 *   - fa2_exact: the repulsion between every pair of nodes;
 *   - fa2_bh: the traversal of a Barnes-Hut tree. The tree itself is built on
 *     the CPU at every iteration (which is cheap) and uploaded.
 *
 * Both perform the same operations, in the same order, as their CPU
 * counterparts in forceatlas2.c.
 *
 * This file does not depend on R. See forceatlas2.c for the licence.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fa2.h"
#include "ocl.h"

/* ------------------------------------------------------------------------ */
/* Kernels                                                                  */
/* ------------------------------------------------------------------------ */

static const char *const kernel_source =
    "#ifdef FA2_DOUBLE\n"
    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
    "typedef double real;\n"
    "#else\n"
    "typedef float real;\n"
    "#endif\n"
    "#pragma OPENCL FP_CONTRACT OFF\n"
    "\n"
    "/* Repulsion factor between two nodes (see forceatlas2.c). */\n"
    "real rep_factor(real d2, real mm, real ss, real scaling, int adjust)\n"
    "{\n"
    "    if (adjust) {\n"
    "        const real d = sqrt(d2) - ss;\n"
    "        if (d > (real) 0) return scaling * mm / (d * d);\n"
    "        if (d < (real) 0) return (real) 100 * scaling * mm;\n"
    "        return (real) 0;\n"
    "    }\n"
    "    return d2 > (real) 0 ? scaling * mm / d2 : (real) 0;\n"
    "}\n"
    "\n"
    "/* p holds x, y, mass and radius of each node; f receives the force on\n"
    " * each node. Only the nodes j0 <= j < j1 are taken into account: the\n"
    " * kernel is called for successive slices so that no single call runs\n"
    " * for long. */\n"
    "__kernel void fa2_exact(const int n, const int j0, const int j1,\n"
    "                        const real scaling, const int adjust,\n"
    "                        __global const real *p, __global real *f)\n"
    "{\n"
    "    const int i = (int) get_global_id(0);\n"
    "    if (i >= n) return;\n"
    "    const real xi = p[4 * i], yi = p[4 * i + 1];\n"
    "    const real mi = p[4 * i + 2], si = p[4 * i + 3];\n"
    "    real ax = (real) 0, ay = (real) 0;\n"
    "    if (j0 > 0) {\n"
    "        ax = f[2 * i];\n"
    "        ay = f[2 * i + 1];\n"
    "    }\n"
    "    for (int j = j0; j < j1; j++) {\n"
    "        if (j == i) continue;\n"
    "        const real xd = xi - p[4 * j], yd = yi - p[4 * j + 1];\n"
    "        const real g = rep_factor(xd * xd + yd * yd, mi * p[4 * j + 2],\n"
    "                                  si + p[4 * j + 3], scaling, adjust);\n"
    "        ax += xd * g;\n"
    "        ay += yd * g;\n"
    "    }\n"
    "    f[2 * i] = ax;\n"
    "    f[2 * i + 1] = ay;\n"
    "}\n"
    "\n"
    "/* p and f are as above, with the nodes in the order of the tree. rr\n"
    " * holds the centre (x, y), the mass and the squared size of each region;\n"
    " * ri its first node, its number of nodes, its last child (or -1) and\n"
    " * the region to visit next when it is not opened (or -1). */\n"
    "__kernel void fa2_bh(const int n, const real scaling, const real theta2,\n"
    "                     const int adjust, __global const real *p,\n"
    "                     __global const real *rr, __global const int *ri,\n"
    "                     __global real *f)\n"
    "{\n"
    "    const int k = (int) get_global_id(0);\n"
    "    if (k >= n) return;\n"
    "    const real xi = p[4 * k], yi = p[4 * k + 1];\n"
    "    const real mi = p[4 * k + 2], si = p[4 * k + 3];\n"
    "    real ax = (real) 0, ay = (real) 0;\n"
    "    int cur = 0;\n"
    "    while (cur >= 0) {\n"
    "        const int start = ri[4 * cur], count = ri[4 * cur + 1];\n"
    "        if (count > 1) {\n"
    "            const real xd = xi - rr[4 * cur], yd = yi - rr[4 * cur + 1];\n"
    "            const real d2 = xd * xd + yd * yd;\n"
    "            if (d2 * theta2 > rr[4 * cur + 3]) {\n"
    "                const real g = scaling * (mi * rr[4 * cur + 2]) / d2;\n"
    "                ax += xd * g;\n"
    "                ay += yd * g;\n"
    "                cur = ri[4 * cur + 3];\n"
    "                continue;\n"
    "            }\n"
    "            const int child = ri[4 * cur + 2];\n"
    "            if (child >= 0) {\n"
    "                cur = child;\n"
    "                continue;\n"
    "            }\n"
    "        }\n"
    "        for (int j = start; j < start + count; j++) {\n"
    "            if (j == k) continue;\n"
    "            const real xd = xi - p[4 * j], yd = yi - p[4 * j + 1];\n"
    "            const real g = rep_factor(xd * xd + yd * yd, mi * p[4 * j + 2],\n"
    "                                      si + p[4 * j + 3], scaling, adjust);\n"
    "            ax += xd * g;\n"
    "            ay += yd * g;\n"
    "        }\n"
    "        cur = ri[4 * cur + 3];\n"
    "    }\n"
    "    f[2 * k] = ax;\n"
    "    f[2 * k + 1] = ay;\n"
    "}\n";

/* ------------------------------------------------------------------------ */
/* Errors                                                                   */
/* ------------------------------------------------------------------------ */

static char errbuf[2048] = "";

const char *fa2_gpu_error(void) { return errbuf; }

static void set_error(const char *what, cl_int code)
{
    snprintf(errbuf, sizeof(errbuf), "%s (OpenCL error %d)", what, (int) code);
}

/* ------------------------------------------------------------------------ */
/* Devices                                                                  */
/* ------------------------------------------------------------------------ */

#define FA2_MAX_DEVICES 64

static const fa2_cl_api *cl = NULL;
static cl_device_id devices[FA2_MAX_DEVICES];
static cl_platform_id device_platform[FA2_MAX_DEVICES];
static int ndevices = -1; /* -1: not enumerated yet */

int fa2_gpu_ndevices(void)
{
    if (ndevices >= 0) return ndevices;

    cl = fa2_cl_load();
    if (!cl) {
        snprintf(errbuf, sizeof(errbuf),
                 "no OpenCL runtime was found on this computer");
        return 0; /* not cached: a driver may be installed later */
    }

    cl_platform_id platforms[16];
    cl_uint nplat = 0;
    ndevices = 0;
    if (cl->GetPlatformIDs(16, platforms, &nplat) != CL_SUCCESS) return 0;
    if (nplat > 16) nplat = 16;

    for (cl_uint p = 0; p < nplat; p++) {
        cl_device_id devs[16];
        cl_uint ndev = 0;
        if (cl->GetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 16, devs,
                             &ndev) != CL_SUCCESS)
            continue;
        if (ndev > 16) ndev = 16;
        for (cl_uint d = 0; d < ndev && ndevices < FA2_MAX_DEVICES; d++) {
            devices[ndevices] = devs[d];
            device_platform[ndevices] = platforms[p];
            ndevices++;
        }
    }
    return ndevices;
}

int fa2_gpu_device_info(int idx, fa2_gpu_devinfo *info)
{
    if (idx < 0 || idx >= fa2_gpu_ndevices()) return 1;
    memset(info, 0, sizeof(*info));

    size_t len;
#define DEV_STRING(what, field)                                              \
    do {                                                                     \
        len = 0;                                                             \
        if (cl->GetDeviceInfo(devices[idx], what, sizeof(info->field) - 1,   \
                              info->field, &len) != CL_SUCCESS)              \
            len = 0;                                                         \
        info->field[len < sizeof(info->field) ? len                          \
                                              : sizeof(info->field) - 1] = 0; \
    } while (0)
    DEV_STRING(CL_DEVICE_NAME, name);
    DEV_STRING(CL_DEVICE_VENDOR, vendor);
    DEV_STRING(CL_DEVICE_VERSION, version);
#undef DEV_STRING
    len = 0;
    if (cl->GetPlatformInfo(device_platform[idx], CL_PLATFORM_NAME,
                            sizeof(info->platform) - 1, info->platform,
                            &len) != CL_SUCCESS)
        len = 0;
    info->platform[len < sizeof(info->platform) ? len
                                                : sizeof(info->platform) - 1] = 0;

    cl_bitfield type = 0, fp = 0;
    cl_ulong mem = 0;
    if (cl->GetDeviceInfo(devices[idx], CL_DEVICE_TYPE, sizeof(type), &type,
                          NULL) == CL_SUCCESS)
        info->is_gpu = (type & CL_DEVICE_TYPE_GPU) != 0;
    if (cl->GetDeviceInfo(devices[idx], CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp),
                          &fp, NULL) == CL_SUCCESS)
        info->has_double = fp != 0;
    if (cl->GetDeviceInfo(devices[idx], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem),
                          &mem, NULL) == CL_SUCCESS)
        info->memory = (double) mem;
    return 0;
}

/* ------------------------------------------------------------------------ */
/* Context, kept between calls                                              */
/* ------------------------------------------------------------------------ */

/* Creating a context and compiling the kernels takes a noticeable time, so
 * they are kept for the next layout on the same device. */
static struct {
    int device; /* -1: none */
    cl_context context;
    cl_command_queue queue;
    cl_program program[2]; /* single, double precision */
} cache = {-1, NULL, NULL, {NULL, NULL}};

static void cache_release(void)
{
    if (!cl) return;
    for (int i = 0; i < 2; i++) {
        if (cache.program[i]) cl->ReleaseProgram(cache.program[i]);
        cache.program[i] = NULL;
    }
    if (cache.queue) cl->ReleaseCommandQueue(cache.queue);
    if (cache.context) cl->ReleaseContext(cache.context);
    cache.queue = NULL;
    cache.context = NULL;
    cache.device = -1;
}

void fa2_gpu_shutdown(void)
{
    cache_release();
    fa2_cl_unload();
    cl = NULL;
    ndevices = -1;
}

static int cache_prepare(int device, int use_double)
{
    cl_int err = CL_SUCCESS;

    if (cache.device != device) {
        cache_release();
        cache.context =
            cl->CreateContext(NULL, 1, &devices[device], NULL, NULL, &err);
        if (!cache.context || err != CL_SUCCESS) {
            cache.context = NULL;
            set_error("could not create an OpenCL context on this device", err);
            return 1;
        }
        cache.queue =
            cl->CreateCommandQueue(cache.context, devices[device], 0, &err);
        if (!cache.queue || err != CL_SUCCESS) {
            cache.queue = NULL;
            cache_release();
            set_error("could not create an OpenCL command queue", err);
            return 1;
        }
        cache.device = device;
    }

    if (!cache.program[use_double]) {
        cl_program prog = cl->CreateProgramWithSource(
            cache.context, 1, (const char **) &kernel_source, NULL, &err);
        if (!prog || err != CL_SUCCESS) {
            set_error("could not create the OpenCL program", err);
            return 1;
        }
        err = cl->BuildProgram(prog, 1, &devices[device],
                               use_double ? "-DFA2_DOUBLE" : "", NULL, NULL);
        if (err != CL_SUCCESS) {
            char log[1500] = "";
            size_t len = 0;
            cl->GetProgramBuildInfo(prog, devices[device],
                                    CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log,
                                    &len);
            log[len < sizeof(log) ? len : sizeof(log) - 1] = '\0';
            snprintf(errbuf, sizeof(errbuf),
                     "the OpenCL kernels could not be compiled for this "
                     "device (OpenCL error %d):\n%s",
                     (int) err, log);
            cl->ReleaseProgram(prog);
            return 1;
        }
        cache.program[use_double] = prog;
    }
    return 0;
}

/* ------------------------------------------------------------------------ */
/* Sessions                                                                 */
/* ------------------------------------------------------------------------ */

struct fa2_gpu {
    int n, use_double, adjust, barnes_hut, threads;
    double t_send, t_compute, t_fetch;
    size_t real_size;
    cl_kernel kernel;
    cl_mem buf_p, buf_f, buf_rr, buf_ri;
    /* Host-side copies, in the precision of the device computations */
    void *host_p, *host_f, *host_rr;
    int *host_ri;
};

void fa2_gpu_close(fa2_gpu *g)
{
    if (!g) return;
    if (cl) {
        if (cache.queue) cl->Finish(cache.queue);
        if (g->buf_p) cl->ReleaseMemObject(g->buf_p);
        if (g->buf_f) cl->ReleaseMemObject(g->buf_f);
        if (g->buf_rr) cl->ReleaseMemObject(g->buf_rr);
        if (g->buf_ri) cl->ReleaseMemObject(g->buf_ri);
        if (g->kernel) cl->ReleaseKernel(g->kernel);
    }
    free(g->host_p);
    free(g->host_f);
    free(g->host_rr);
    free(g->host_ri);
    free(g);
}

void fa2_gpu_times(const fa2_gpu *g, double times[3])
{
    times[0] = g->t_send;
    times[1] = g->t_compute;
    times[2] = g->t_fetch;
}

fa2_gpu *fa2_gpu_open(int device, int use_double, int n, int adjust,
                      int barnes_hut, int threads)
{
    if (device < 0 || device >= fa2_gpu_ndevices()) {
        if (cl)
            snprintf(errbuf, sizeof(errbuf), "there is no OpenCL device %d",
                     device + 1);
        return NULL;
    }
    if (n < 1 || n > (1 << 27)) {
        snprintf(errbuf, sizeof(errbuf),
                 "the GPU backend handles at most %d nodes", 1 << 27);
        return NULL;
    }
    use_double = use_double ? 1 : 0;
    if (use_double) {
        fa2_gpu_devinfo info;
        fa2_gpu_device_info(device, &info);
        if (!info.has_double) {
            snprintf(errbuf, sizeof(errbuf),
                     "this device does not support double precision");
            return NULL;
        }
    }
    if (cache_prepare(device, use_double)) return NULL;

    fa2_gpu *g = (fa2_gpu *) calloc(1, sizeof(fa2_gpu));
    if (!g) {
        snprintf(errbuf, sizeof(errbuf), "out of memory");
        return NULL;
    }
    g->n = n;
    g->use_double = use_double;
    g->adjust = adjust ? 1 : 0;
    g->barnes_hut = barnes_hut ? 1 : 0;
    g->threads = threads > 1 ? threads : 1;
    g->real_size = use_double ? sizeof(double) : sizeof(float);

    const size_t nn = (size_t) n;
    const size_t nreg = 2 * nn; /* upper bound on the number of regions */
    cl_int err = CL_SUCCESS;

    g->host_p = malloc(4 * nn * g->real_size);
    g->host_f = malloc(2 * nn * g->real_size);
    if (!g->host_p || !g->host_f) goto nomem;
    if (g->barnes_hut) {
        g->host_rr = malloc(4 * nreg * g->real_size);
        g->host_ri = (int *) malloc(4 * nreg * sizeof(int));
        if (!g->host_rr || !g->host_ri) goto nomem;
    }

    g->kernel = cl->CreateKernel(cache.program[use_double],
                                 g->barnes_hut ? "fa2_bh" : "fa2_exact", &err);
    if (!g->kernel || err != CL_SUCCESS) {
        g->kernel = NULL;
        set_error("could not create the OpenCL kernel", err);
        goto fail;
    }

#define MAKE_BUFFER(field, flags, bytes)                                     \
    do {                                                                     \
        g->field = cl->CreateBuffer(cache.context, flags, bytes, NULL, &err); \
        if (!g->field || err != CL_SUCCESS) {                                \
            g->field = NULL;                                                 \
            set_error("could not allocate memory on the device", err);       \
            goto fail;                                                       \
        }                                                                    \
    } while (0)
    MAKE_BUFFER(buf_p, CL_MEM_READ_ONLY, 4 * nn * g->real_size);
    MAKE_BUFFER(buf_f, CL_MEM_READ_WRITE, 2 * nn * g->real_size);
    if (g->barnes_hut) {
        MAKE_BUFFER(buf_rr, CL_MEM_READ_ONLY, 4 * nreg * g->real_size);
        MAKE_BUFFER(buf_ri, CL_MEM_READ_ONLY, 4 * nreg * sizeof(int));
    }
#undef MAKE_BUFFER
    return g;

nomem:
    snprintf(errbuf, sizeof(errbuf), "out of memory");
fail:
    fa2_gpu_close(g);
    return NULL;
}

/* The loops that convert between the doubles of the CPU and the reals of the
 * device are written for both types with this macro. They run in parallel:
 * for large graphs they would otherwise take longer than the GPU itself. */
#ifdef _OPENMP
#define FA2_PARALLEL_FOR                                                      \
    _Pragma("omp parallel for num_threads(g->threads) schedule(static) if (g->threads > 1)")
#else
#define FA2_PARALLEL_FOR
#endif

#define DEFINE_CONVERTERS(REAL, SUFFIX)                                       \
    static void nodes_to_device_##SUFFIX(const fa2_gpu *g, const double *x,   \
                                         const double *y, const double *mass, \
                                         const double *size)                  \
    {                                                                         \
        REAL *p = (REAL *) g->host_p;                                         \
        FA2_PARALLEL_FOR                                                      \
        for (int i = 0; i < g->n; i++) {                                      \
            p[4 * i] = (REAL) x[i];                                           \
            p[4 * i + 1] = (REAL) y[i];                                       \
            p[4 * i + 2] = (REAL) mass[i];                                    \
            p[4 * i + 3] = size ? (REAL) size[i] : (REAL) 0;                  \
        }                                                                     \
    }                                                                         \
    static void regions_to_device_##SUFFIX(const fa2_gpu *g,                  \
                                           const fa2_tree *t)                 \
    {                                                                         \
        REAL *rr = (REAL *) g->host_rr;                                       \
        int *ri = g->host_ri;                                                 \
        FA2_PARALLEL_FOR                                                      \
        for (int r = 0; r < t->nreg; r++) {                                   \
            const fa2_region *reg = &t->reg[r];                               \
            rr[4 * r] = (REAL) reg->cx;                                       \
            rr[4 * r + 1] = (REAL) reg->cy;                                   \
            rr[4 * r + 2] = (REAL) reg->mass;                                 \
            rr[4 * r + 3] = (REAL) (reg->size * reg->size);                   \
            ri[4 * r] = reg->start;                                           \
            ri[4 * r + 1] = reg->count;                                       \
            ri[4 * r + 2] =                                                   \
                reg->nchild > 0 ? reg->child + reg->nchild - 1 : -1;          \
            ri[4 * r + 3] = reg->skip;                                        \
        }                                                                     \
    }                                                                         \
    /* perm[k] is the node that element k of the result belongs to (NULL:    \
     * node k). It is a permutation, so the iterations are independent. */   \
    static void forces_from_device_##SUFFIX(const fa2_gpu *g,                 \
                                            const int *perm, double *dx,      \
                                            double *dy)                       \
    {                                                                         \
        const REAL *f = (const REAL *) g->host_f;                             \
        FA2_PARALLEL_FOR                                                      \
        for (int k = 0; k < g->n; k++) {                                      \
            const int i = perm ? perm[k] : k;                                 \
            dx[i] += (double) f[2 * k];                                       \
            dy[i] += (double) f[2 * k + 1];                                   \
        }                                                                     \
    }

DEFINE_CONVERTERS(float, single)
DEFINE_CONVERTERS(double, double)

static void nodes_to_device(const fa2_gpu *g, const double *x, const double *y,
                            const double *mass, const double *size)
{
    if (g->use_double)
        nodes_to_device_double(g, x, y, mass, size);
    else
        nodes_to_device_single(g, x, y, mass, size);
}

static void regions_to_device(const fa2_gpu *g, const fa2_tree *t)
{
    if (g->use_double)
        regions_to_device_double(g, t);
    else
        regions_to_device_single(g, t);
}

static void forces_from_device(const fa2_gpu *g, const int *perm, double *dx,
                               double *dy)
{
    if (g->use_double)
        forces_from_device_double(g, perm, dx, dy);
    else
        forces_from_device_single(g, perm, dx, dy);
}

static cl_int set_arg_real(const fa2_gpu *g, cl_uint idx, double v)
{
    if (g->use_double) return cl->SetKernelArg(g->kernel, idx, sizeof(v), &v);
    const float f = (float) v;
    return cl->SetKernelArg(g->kernel, idx, sizeof(f), &f);
}

static cl_int set_arg_int(const fa2_gpu *g, cl_uint idx, int v)
{
    const cl_int i = (cl_int) v;
    return cl->SetKernelArg(g->kernel, idx, sizeof(i), &i);
}

static cl_int set_arg_mem(const fa2_gpu *g, cl_uint idx, cl_mem m)
{
    return cl->SetKernelArg(g->kernel, idx, sizeof(m), &m);
}

int fa2_gpu_repulse_exact(fa2_gpu *g, const double *x, const double *y,
                          const double *mass, const double *size,
                          double scaling, double *dx, double *dy)
{
    const size_t n = (size_t) g->n;
    cl_int err = CL_SUCCESS;

    double t0 = fa2_now(), t1;
    nodes_to_device(g, x, y, mass, size);
    err = cl->EnqueueWriteBuffer(cache.queue, g->buf_p, CL_FALSE, 0,
                                 4 * n * g->real_size, g->host_p, 0, NULL,
                                 NULL);
    if (err != CL_SUCCESS) goto fail;
    t1 = fa2_now();
    g->t_send += t1 - t0;
    t0 = t1;

    err = set_arg_int(g, 0, g->n);
    if (!err) err = set_arg_real(g, 3, scaling);
    if (!err) err = set_arg_int(g, 4, g->adjust);
    if (!err) err = set_arg_mem(g, 5, g->buf_p);
    if (!err) err = set_arg_mem(g, 6, g->buf_f);
    if (err != CL_SUCCESS) goto fail;

    /* About 2^30 pairs of nodes per call, so that each call is short: on
     * Windows, the display driver is reset when a call takes seconds. */
    size_t slice = ((size_t) 1 << 30) / n;
    if (slice < 256) slice = 256;
    for (size_t j0 = 0; j0 < n; j0 += slice) {
        const size_t j1 = j0 + slice < n ? j0 + slice : n;
        err = set_arg_int(g, 1, (int) j0);
        if (!err) err = set_arg_int(g, 2, (int) j1);
        if (!err)
            err = cl->EnqueueNDRangeKernel(cache.queue, g->kernel, 1, NULL, &n,
                                           NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) goto fail;
        cl->Flush(cache.queue);
    }

    err = cl->Finish(cache.queue);
    if (err != CL_SUCCESS) goto fail;
    t1 = fa2_now();
    g->t_compute += t1 - t0;
    t0 = t1;

    err = cl->EnqueueReadBuffer(cache.queue, g->buf_f, CL_TRUE, 0,
                                2 * n * g->real_size, g->host_f, 0, NULL, NULL);
    if (err != CL_SUCCESS) goto fail;
    forces_from_device(g, NULL, dx, dy);
    g->t_fetch += fa2_now() - t0;
    return 0;

fail:
    set_error("the computation on the GPU failed", err);
    return 1;
}

int fa2_gpu_repulse_bh(fa2_gpu *g, const fa2_tree *t, double scaling,
                       double theta, double *dx, double *dy)
{
    const size_t n = (size_t) g->n;
    const size_t nreg = (size_t) t->nreg;
    cl_int err = CL_SUCCESS;

    double t0 = fa2_now(), t1;
    nodes_to_device(g, t->px, t->py, t->pm, t->ps);
    regions_to_device(g, t);

    /* The writes are not blocking: the host arrays are left untouched until
     * the result has been read, which waits for everything before it. */
    err = cl->EnqueueWriteBuffer(cache.queue, g->buf_p, CL_FALSE, 0,
                                 4 * n * g->real_size, g->host_p, 0, NULL,
                                 NULL);
    if (!err)
        err = cl->EnqueueWriteBuffer(cache.queue, g->buf_rr, CL_FALSE, 0,
                                     4 * nreg * g->real_size, g->host_rr, 0,
                                     NULL, NULL);
    if (!err)
        err = cl->EnqueueWriteBuffer(cache.queue, g->buf_ri, CL_FALSE, 0,
                                     4 * nreg * sizeof(int), g->host_ri, 0,
                                     NULL, NULL);
    if (err != CL_SUCCESS) goto fail;
    t1 = fa2_now();
    g->t_send += t1 - t0;
    t0 = t1;

    err = set_arg_int(g, 0, g->n);
    if (!err) err = set_arg_real(g, 1, scaling);
    if (!err) err = set_arg_real(g, 2, theta * theta);
    if (!err) err = set_arg_int(g, 3, g->adjust);
    if (!err) err = set_arg_mem(g, 4, g->buf_p);
    if (!err) err = set_arg_mem(g, 5, g->buf_rr);
    if (!err) err = set_arg_mem(g, 6, g->buf_ri);
    if (!err) err = set_arg_mem(g, 7, g->buf_f);
    if (err != CL_SUCCESS) goto fail;

    /* At most 2^20 nodes per call, for the same reason as above. */
    const size_t slice = (size_t) 1 << 20;
    for (size_t k0 = 0; k0 < n; k0 += slice) {
        const size_t count = k0 + slice < n ? slice : n - k0;
        err = cl->EnqueueNDRangeKernel(cache.queue, g->kernel, 1, &k0, &count,
                                       NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) goto fail;
        cl->Flush(cache.queue);
    }

    err = cl->Finish(cache.queue);
    if (err != CL_SUCCESS) goto fail;
    t1 = fa2_now();
    g->t_compute += t1 - t0;
    t0 = t1;

    err = cl->EnqueueReadBuffer(cache.queue, g->buf_f, CL_TRUE, 0,
                                2 * n * g->real_size, g->host_f, 0, NULL, NULL);
    if (err != CL_SUCCESS) goto fail;
    forces_from_device(g, t->perm, dx, dy);
    g->t_fetch += fa2_now() - t0;
    return 0;

fail:
    set_error("the computation on the GPU failed", err);
    return 1;
}

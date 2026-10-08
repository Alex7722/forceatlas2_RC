/*
 * ForceAtlas2 entirely on the GPU: host side of forceatlas2_large().
 *
 * The kernels are in inst/opencl/pipeline.cl, where the sequence of an
 * iteration is described. This file uploads the graph, enqueues the kernels,
 * adjusts the global speed between the iterations and downloads the final
 * positions. It does not depend on R. See forceatlas2.c for the licence.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fa2.h"
#include "gpu_internal.h"

enum {
    K_BBOX_PARTIAL,
    K_BBOX_FINAL,
    K_MORTON,
    K_SORT_HIST,
    K_SORT_SCAN,
    K_SORT_SCATTER,
    K_GATHER,
    K_TREE,
    K_AGGREGATE,
    K_FINALIZE,
    K_REPULSE,
    K_ATTRACT,
    K_SWING,
    K_SUM2,
    K_APPLY,
    K_COUNT
};

static const char *const kernel_names[K_COUNT] = {
    "bbox_partial", "bbox_final", "morton",   "sort_hist", "sort_scan",
    "sort_scatter", "gather",     "tree",     "aggregate", "finalize",
    "repulse",      "attract",    "swing",    "sum2",      "apply"};

enum {
    M_POS,
    M_MASS,
    M_SIZE,
    M_FIXED,
    M_F,
    M_F_OLD,
    M_SW,
    M_INC_OFF,
    M_INC_NODE,
    M_INC_COEF,
    M_PARTIAL,
    M_BBOX,
    M_TOTAL,
    M_KEY_A,
    M_IDX_A,
    M_KEY_B,
    M_IDX_B,
    M_HIST,
    M_DIGIT_TOTAL,
    M_PSORT,
    M_CHILD,
    M_RANGE,
    M_PARENT,
    M_VISITS,
    M_ND,
    M_NI,
    M_COUNT
};

typedef struct {
    const fa2_cl_api *cl;
    cl_command_queue queue;
    cl_kernel kernel[K_COUNT];
    cl_mem mem[M_COUNT];
    int use_double;
    size_t real_size;
    cl_int err; /* first OpenCL error met */
} session;

static void session_free(session *s)
{
    if (!s->cl) return;
    if (s->queue) s->cl->Finish(s->queue);
    for (int i = 0; i < M_COUNT; i++)
        if (s->mem[i]) s->cl->ReleaseMemObject(s->mem[i]);
    for (int i = 0; i < K_COUNT; i++)
        if (s->kernel[i]) s->cl->ReleaseKernel(s->kernel[i]);
}

/* The helpers below do nothing once an error has been met, so that a
 * sequence of calls can be checked once at its end. */

static void arg_int(session *s, int k, cl_uint idx, int v)
{
    const cl_int i = (cl_int) v;
    if (!s->err) s->err = s->cl->SetKernelArg(s->kernel[k], idx, sizeof(i), &i);
}

static void arg_real(session *s, int k, cl_uint idx, double v)
{
    if (s->err) return;
    if (s->use_double) {
        s->err = s->cl->SetKernelArg(s->kernel[k], idx, sizeof(v), &v);
    } else {
        const float f = (float) v;
        s->err = s->cl->SetKernelArg(s->kernel[k], idx, sizeof(f), &f);
    }
}

static void arg_mem(session *s, int k, cl_uint idx, int m)
{
    if (!s->err)
        s->err = s->cl->SetKernelArg(s->kernel[k], idx, sizeof(cl_mem),
                                     &s->mem[m]);
}

/* Run a kernel over `count` work items. */
static void run(session *s, int k, size_t count)
{
    if (!s->err)
        s->err = s->cl->EnqueueNDRangeKernel(s->queue, s->kernel[k], 1, NULL,
                                             &count, NULL, 0, NULL, NULL);
}

static void make_buffer(session *s, cl_context context, int m, size_t bytes)
{
    if (s->err) return;
    cl_int err = CL_SUCCESS;
    s->mem[m] = s->cl->CreateBuffer(context, CL_MEM_READ_WRITE,
                                    bytes > 0 ? bytes : 1, NULL, &err);
    if (!s->mem[m] && err == CL_SUCCESS) err = -5; /* CL_OUT_OF_RESOURCES */
    s->err = err;
}

static void upload(session *s, int m, size_t bytes, const void *data)
{
    if (!s->err && bytes > 0)
        s->err = s->cl->EnqueueWriteBuffer(s->queue, s->mem[m], CL_TRUE, 0,
                                           bytes, data, 0, NULL, NULL);
}

/* Upload doubles as the reals of the device. `scratch` holds count floats. */
static void upload_reals(session *s, int m, size_t count, const double *data,
                         float *scratch)
{
    if (s->use_double) {
        upload(s, m, count * sizeof(double), data);
    } else {
        for (size_t i = 0; i < count; i++) scratch[i] = (float) data[i];
        upload(s, m, count * sizeof(float), scratch);
    }
}

/* Wait for the device and add the time elapsed since *t0 to *slot. */
static void lap(session *s, double *t0, double *slot)
{
    if (!s->err) s->err = s->cl->Finish(s->queue);
    const double t1 = fa2_now();
    *slot += t1 - *t0;
    *t0 = t1;
}

int fa2_full_run(const fa2_full_params *p, double *pos, double *times)
{
    const int n = p->n;
    const size_t nn = (size_t) n;
    const size_t ninc = (size_t) p->inc_off[n];
    session s;
    int status = 1;
    double *work = NULL; /* host scratch space */
    unsigned char *fixed8 = NULL;

    memset(&s, 0, sizeof(s));
    for (int i = 0; i < FA2_FULL_NTIMES; i++) times[i] = 0.0;

    if (n < 2 || n > (1 << 26)) {
        fa2_gpu_set_message(
            "the all-GPU layout handles between 2 and 67 million nodes");
        return 1;
    }

    cl_context context;
    cl_program program;
    if (fa2_gpu_acquire(p->device, 1, p->use_double, p->source, &context,
                        &s.queue, &program))
        return 1;
    s.cl = fa2_gpu_cl();
    s.use_double = p->use_double ? 1 : 0;
    s.real_size = s.use_double ? sizeof(double) : sizeof(float);
    const size_t rs = s.real_size;

    /* Blocks of S consecutive nodes, B of them, none empty. */
    int B = n / 256;
    if (B < 1) B = 1;
    if (B > 16384) B = 16384;
    const int S = (n + B - 1) / B;
    B = (n + S - 1) / S;

    for (int k = 0; k < K_COUNT; k++) {
        cl_int err = CL_SUCCESS;
        s.kernel[k] = s.cl->CreateKernel(program, kernel_names[k], &err);
        if (!s.kernel[k] || err != CL_SUCCESS) {
            s.kernel[k] = NULL;
            fa2_gpu_set_error("could not create an OpenCL kernel", err);
            goto done;
        }
    }

    make_buffer(&s, context, M_POS, 2 * nn * rs);
    make_buffer(&s, context, M_MASS, nn * rs);
    make_buffer(&s, context, M_SIZE, p->adjust ? nn * rs : 1);
    make_buffer(&s, context, M_FIXED, p->any_fixed ? nn : 1);
    make_buffer(&s, context, M_F, 2 * nn * rs);
    make_buffer(&s, context, M_F_OLD, 2 * nn * rs);
    make_buffer(&s, context, M_SW, nn * rs);
    make_buffer(&s, context, M_INC_OFF, (nn + 1) * sizeof(int));
    make_buffer(&s, context, M_INC_NODE, ninc * sizeof(int));
    make_buffer(&s, context, M_INC_COEF, ninc * rs);
    make_buffer(&s, context, M_PARTIAL, 4 * (size_t) B * rs);
    make_buffer(&s, context, M_BBOX, 4 * rs);
    make_buffer(&s, context, M_TOTAL, 2 * rs);
    make_buffer(&s, context, M_KEY_A, nn * sizeof(cl_uint));
    make_buffer(&s, context, M_IDX_A, nn * sizeof(cl_uint));
    make_buffer(&s, context, M_KEY_B, nn * sizeof(cl_uint));
    make_buffer(&s, context, M_IDX_B, nn * sizeof(cl_uint));
    make_buffer(&s, context, M_HIST, 256 * (size_t) B * sizeof(cl_uint));
    make_buffer(&s, context, M_DIGIT_TOTAL, 256 * sizeof(cl_uint));
    make_buffer(&s, context, M_PSORT, 4 * nn * rs);
    make_buffer(&s, context, M_CHILD, 2 * (nn - 1) * sizeof(int));
    make_buffer(&s, context, M_RANGE, 2 * (nn - 1) * sizeof(int));
    make_buffer(&s, context, M_PARENT, (2 * nn - 1) * sizeof(int));
    make_buffer(&s, context, M_VISITS, (nn - 1) * sizeof(int));
    make_buffer(&s, context, M_ND, 8 * (2 * nn - 1) * rs);
    make_buffer(&s, context, M_NI, 4 * (nn - 1) * rs);
    if (s.err) {
        fa2_gpu_set_error("could not allocate memory on the device", s.err);
        goto done;
    }

    /* Upload the graph. The positions are stored as x then y on the host,
     * and interleaved on the device. */
    {
        size_t len = 2 * nn > ninc ? 2 * nn : ninc;
        work = (double *) malloc(len * sizeof(double));
        if (!work) {
            fa2_gpu_set_message("out of memory");
            goto done;
        }
        float *scratch = (float *) work;

        if (s.use_double) {
            for (size_t i = 0; i < nn; i++) {
                work[2 * i] = pos[i];
                work[2 * i + 1] = pos[nn + i];
            }
        } else {
            for (size_t i = 0; i < nn; i++) {
                scratch[2 * i] = (float) pos[i];
                scratch[2 * i + 1] = (float) pos[nn + i];
            }
        }
        upload(&s, M_POS, 2 * nn * rs, work);

        memset(work, 0, 2 * nn * sizeof(double));
        upload(&s, M_F_OLD, 2 * nn * rs, work);

        upload_reals(&s, M_MASS, nn, p->mass, scratch);
        if (p->adjust) upload_reals(&s, M_SIZE, nn, p->size, scratch);
        upload_reals(&s, M_INC_COEF, ninc, p->inc_coef, scratch);
        upload(&s, M_INC_OFF, (nn + 1) * sizeof(int), p->inc_off);
        upload(&s, M_INC_NODE, ninc * sizeof(int), p->inc_node);

        if (p->any_fixed) {
            fixed8 = (unsigned char *) malloc(nn);
            if (!fixed8) {
                fa2_gpu_set_message("out of memory");
                goto done;
            }
            for (size_t i = 0; i < nn; i++) fixed8[i] = p->fixed[i] != 0;
            upload(&s, M_FIXED, nn, fixed8);
        }
        if (s.err) {
            fa2_gpu_set_error("could not send the graph to the device", s.err);
            goto done;
        }
    }

    /* Arguments that do not change from one iteration to the next. */
    arg_int(&s, K_BBOX_PARTIAL, 0, n);
    arg_int(&s, K_BBOX_PARTIAL, 1, S);
    arg_mem(&s, K_BBOX_PARTIAL, 2, M_POS);
    arg_mem(&s, K_BBOX_PARTIAL, 3, M_PARTIAL);

    arg_int(&s, K_BBOX_FINAL, 0, B);
    arg_mem(&s, K_BBOX_FINAL, 1, M_PARTIAL);
    arg_mem(&s, K_BBOX_FINAL, 2, M_BBOX);

    arg_int(&s, K_MORTON, 0, n);
    arg_mem(&s, K_MORTON, 1, M_POS);
    arg_mem(&s, K_MORTON, 2, M_BBOX);
    arg_mem(&s, K_MORTON, 3, M_KEY_A);
    arg_mem(&s, K_MORTON, 4, M_IDX_A);

    arg_int(&s, K_SORT_HIST, 0, n);
    arg_int(&s, K_SORT_HIST, 1, S);
    arg_int(&s, K_SORT_HIST, 2, B);
    arg_mem(&s, K_SORT_HIST, 5, M_HIST);

    arg_int(&s, K_SORT_SCAN, 0, B);
    arg_mem(&s, K_SORT_SCAN, 1, M_HIST);
    arg_mem(&s, K_SORT_SCAN, 2, M_DIGIT_TOTAL);

    arg_int(&s, K_SORT_SCATTER, 0, n);
    arg_int(&s, K_SORT_SCATTER, 1, S);
    arg_int(&s, K_SORT_SCATTER, 2, B);
    arg_mem(&s, K_SORT_SCATTER, 6, M_HIST);
    arg_mem(&s, K_SORT_SCATTER, 7, M_DIGIT_TOTAL);

    /* After the four passes of the sort, the result is in the A buffers. */
    arg_int(&s, K_GATHER, 0, n);
    arg_int(&s, K_GATHER, 1, p->adjust);
    arg_mem(&s, K_GATHER, 2, M_IDX_A);
    arg_mem(&s, K_GATHER, 3, M_POS);
    arg_mem(&s, K_GATHER, 4, M_MASS);
    arg_mem(&s, K_GATHER, 5, M_SIZE);
    arg_mem(&s, K_GATHER, 6, M_PSORT);

    arg_int(&s, K_TREE, 0, n);
    arg_mem(&s, K_TREE, 1, M_KEY_A);
    arg_mem(&s, K_TREE, 2, M_CHILD);
    arg_mem(&s, K_TREE, 3, M_RANGE);
    arg_mem(&s, K_TREE, 4, M_PARENT);
    arg_mem(&s, K_TREE, 5, M_VISITS);

    arg_int(&s, K_AGGREGATE, 0, n);
    arg_mem(&s, K_AGGREGATE, 1, M_PSORT);
    arg_mem(&s, K_AGGREGATE, 2, M_CHILD);
    arg_mem(&s, K_AGGREGATE, 3, M_PARENT);
    arg_mem(&s, K_AGGREGATE, 4, M_VISITS);
    arg_mem(&s, K_AGGREGATE, 5, M_ND);

    arg_int(&s, K_FINALIZE, 0, n);
    arg_mem(&s, K_FINALIZE, 1, M_ND);
    arg_mem(&s, K_FINALIZE, 2, M_NI);

    arg_int(&s, K_REPULSE, 0, n);
    arg_real(&s, K_REPULSE, 1, p->scaling);
    arg_real(&s, K_REPULSE, 2, p->theta * p->theta);
    arg_int(&s, K_REPULSE, 3, p->adjust);
    arg_mem(&s, K_REPULSE, 4, M_PSORT);
    arg_mem(&s, K_REPULSE, 5, M_NI);
    arg_mem(&s, K_REPULSE, 6, M_CHILD);
    arg_mem(&s, K_REPULSE, 7, M_RANGE);
    arg_mem(&s, K_REPULSE, 8, M_IDX_A);
    arg_mem(&s, K_REPULSE, 9, M_F);

    arg_int(&s, K_ATTRACT, 0, n);
    arg_real(&s, K_ATTRACT, 1, p->gravity);
    arg_int(&s, K_ATTRACT, 2, p->strong);
    arg_int(&s, K_ATTRACT, 3, p->linlog);
    arg_int(&s, K_ATTRACT, 4, p->adjust);
    arg_mem(&s, K_ATTRACT, 5, M_POS);
    arg_mem(&s, K_ATTRACT, 6, M_MASS);
    arg_mem(&s, K_ATTRACT, 7, M_SIZE);
    arg_mem(&s, K_ATTRACT, 8, M_INC_OFF);
    arg_mem(&s, K_ATTRACT, 9, M_INC_NODE);
    arg_mem(&s, K_ATTRACT, 10, M_INC_COEF);
    arg_mem(&s, K_ATTRACT, 11, M_F);

    arg_int(&s, K_SWING, 0, n);
    arg_int(&s, K_SWING, 1, S);
    arg_int(&s, K_SWING, 2, p->any_fixed);
    arg_mem(&s, K_SWING, 3, M_F);
    arg_mem(&s, K_SWING, 4, M_F_OLD);
    arg_mem(&s, K_SWING, 5, M_MASS);
    arg_mem(&s, K_SWING, 6, M_FIXED);
    arg_mem(&s, K_SWING, 7, M_SW);
    arg_mem(&s, K_SWING, 8, M_PARTIAL);

    arg_int(&s, K_SUM2, 0, B);
    arg_mem(&s, K_SUM2, 1, M_PARTIAL);
    arg_mem(&s, K_SUM2, 2, M_TOTAL);

    arg_int(&s, K_APPLY, 0, n);
    arg_int(&s, K_APPLY, 2, p->adjust);
    arg_int(&s, K_APPLY, 3, p->any_fixed);
    arg_mem(&s, K_APPLY, 4, M_F);
    arg_mem(&s, K_APPLY, 5, M_F_OLD);
    arg_mem(&s, K_APPLY, 6, M_SW);
    arg_mem(&s, K_APPLY, 7, M_FIXED);
    arg_mem(&s, K_APPLY, 8, M_POS);
    if (s.err) {
        fa2_gpu_set_error("could not set up the OpenCL kernels", s.err);
        goto done;
    }

    double speed = 1.0, speed_efficiency = 1.0;
    double work_done = 0.0;
    const double start = fa2_now();
    double t0 = start;

    for (int it = 0; it < p->iterations; it++) {
        work_done += (double) n + (double) ninc;
        if (work_done >= 1e6) {
            work_done = 0.0;
            if (p->interrupted && p->interrupted()) {
                fa2_gpu_set_message("the layout was interrupted");
                goto done;
            }
        }

        /* 1-2. Bounding box and Morton codes */
        run(&s, K_BBOX_PARTIAL, (size_t) B);
        run(&s, K_BBOX_FINAL, 1);
        run(&s, K_MORTON, nn);
        if (p->profile) lap(&s, &t0, &times[FA2_FULL_T_CODES]);

        /* 3. Sort, alternating between the A and B buffers */
        for (int pass = 0; pass < 4; pass++) {
            const int key_in = pass % 2 ? M_KEY_B : M_KEY_A;
            const int idx_in = pass % 2 ? M_IDX_B : M_IDX_A;
            const int key_out = pass % 2 ? M_KEY_A : M_KEY_B;
            const int idx_out = pass % 2 ? M_IDX_A : M_IDX_B;
            arg_int(&s, K_SORT_HIST, 3, 8 * pass);
            arg_mem(&s, K_SORT_HIST, 4, key_in);
            run(&s, K_SORT_HIST, (size_t) B);
            run(&s, K_SORT_SCAN, 256);
            arg_int(&s, K_SORT_SCATTER, 3, 8 * pass);
            arg_mem(&s, K_SORT_SCATTER, 4, key_in);
            arg_mem(&s, K_SORT_SCATTER, 5, idx_in);
            arg_mem(&s, K_SORT_SCATTER, 8, key_out);
            arg_mem(&s, K_SORT_SCATTER, 9, idx_out);
            run(&s, K_SORT_SCATTER, (size_t) B);
        }
        if (p->profile) lap(&s, &t0, &times[FA2_FULL_T_SORT]);

        /* 4. Tree */
        run(&s, K_GATHER, nn);
        run(&s, K_TREE, nn - 1);
        run(&s, K_AGGREGATE, nn);
        run(&s, K_FINALIZE, nn - 1);
        if (p->profile) lap(&s, &t0, &times[FA2_FULL_T_TREE]);

        /* 5. Repulsion, at most 2^20 nodes per call so that no call runs for
         *    long: on Windows, the display driver is reset when a call takes
         *    seconds. */
        {
            const size_t slice = (size_t) 1 << 20;
            for (size_t k0 = 0; k0 < nn && !s.err; k0 += slice) {
                const size_t count = k0 + slice < nn ? slice : nn - k0;
                s.err = s.cl->EnqueueNDRangeKernel(
                    s.queue, s.kernel[K_REPULSE], 1, &k0, &count, NULL, 0,
                    NULL, NULL);
                if (!s.err) s.cl->Flush(s.queue);
            }
        }
        if (p->profile) lap(&s, &t0, &times[FA2_FULL_T_REPULSION]);

        /* 6. Gravity and attraction */
        run(&s, K_ATTRACT, nn);
        if (p->profile) lap(&s, &t0, &times[FA2_FULL_T_ATTRACTION]);

        /* 7. Swinging and traction: the only data read back */
        run(&s, K_SWING, (size_t) B);
        run(&s, K_SUM2, 1);
        double total_swinging = 0.0, total_traction = 0.0;
        if (!s.err) {
            if (s.use_double) {
                double total[2] = {0.0, 0.0};
                s.err = s.cl->EnqueueReadBuffer(s.queue, s.mem[M_TOTAL],
                                                CL_TRUE, 0, sizeof(total),
                                                total, 0, NULL, NULL);
                total_swinging = total[0];
                total_traction = total[1];
            } else {
                float total[2] = {0.0f, 0.0f};
                s.err = s.cl->EnqueueReadBuffer(s.queue, s.mem[M_TOTAL],
                                                CL_TRUE, 0, sizeof(total),
                                                total, 0, NULL, NULL);
                total_swinging = (double) total[0];
                total_traction = (double) total[1];
            }
        }
        if (s.err) break;
        if (!(total_swinging == total_swinging) ||
            !(total_traction == total_traction)) {
            fa2_gpu_set_message(
                "the computation on the GPU produced invalid numbers");
            goto done;
        }
        fa2_update_speed(n, total_swinging, total_traction,
                         p->jitter_tolerance, &speed, &speed_efficiency);

        /* 8. Displacement */
        arg_real(&s, K_APPLY, 1, speed);
        run(&s, K_APPLY, nn);
        if (p->profile) lap(&s, &t0, &times[FA2_FULL_T_MOVE]);
    }
    if (s.err) {
        fa2_gpu_set_error("the computation on the GPU failed", s.err);
        goto done;
    }

    /* Download the positions. */
    s.err = s.cl->EnqueueReadBuffer(s.queue, s.mem[M_POS], CL_TRUE, 0,
                                    2 * nn * rs, work, 0, NULL, NULL);
    if (s.err) {
        fa2_gpu_set_error("could not fetch the layout from the device", s.err);
        goto done;
    }
    if (s.use_double) {
        for (size_t i = 0; i < nn; i++) {
            pos[i] = work[2 * i];
            pos[nn + i] = work[2 * i + 1];
        }
    } else {
        const float *result = (const float *) work;
        for (size_t i = 0; i < nn; i++) {
            pos[i] = (double) result[2 * i];
            pos[nn + i] = (double) result[2 * i + 1];
        }
    }
    times[FA2_FULL_T_TOTAL] = fa2_now() - start;
    status = 0;

done:
    session_free(&s);
    free(work);
    free(fixed8);
    return status;
}

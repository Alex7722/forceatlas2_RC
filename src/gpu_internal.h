/*
 * What gpu.c and gpu_full.c share: the OpenCL API and the context, command
 * queue and compiled programs that are kept between calls.
 */
#ifndef FA2_GPU_INTERNAL_H
#define FA2_GPU_INTERNAL_H

#include "ocl.h"

/* The OpenCL API; valid once fa2_gpu_ndevices() has returned a positive
 * number. */
const fa2_cl_api *fa2_gpu_cl(void);

/* Context, queue and compiled program of a device. `which` is 0 for the
 * repulsion kernels of gpu.c and 1 for the pipeline of gpu_full.c, whose
 * source is given by `source` (it is compiled the first time only). Returns 0
 * on success; they belong to the cache and must not be released. */
int fa2_gpu_acquire(int device, int which, int use_double, const char *source,
                    cl_context *context, cl_command_queue *queue,
                    cl_program *program);

/* Set the message returned by fa2_gpu_error(). */
void fa2_gpu_set_error(const char *what, cl_int code);
void fa2_gpu_set_message(const char *message);

#endif

/*
 * A minimal declaration of the part of the OpenCL 1.2 C API used by this
 * package, and a loader that binds it at run time.
 *
 * The OpenCL library is not linked when the package is built: it is looked up
 * when a GPU layout is first requested. The package therefore builds without
 * any OpenCL header or library, and works on machines without a GPU.
 */
#ifndef FA2_OCL_H
#define FA2_OCL_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define FA2_CL_CALL __stdcall
#else
#define FA2_CL_CALL
#endif

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef cl_uint cl_bool;
typedef cl_ulong cl_bitfield;

typedef struct _cl_platform_id *cl_platform_id;
typedef struct _cl_device_id *cl_device_id;
typedef struct _cl_context *cl_context;
typedef struct _cl_command_queue *cl_command_queue;
typedef struct _cl_mem *cl_mem;
typedef struct _cl_program *cl_program;
typedef struct _cl_kernel *cl_kernel;
typedef struct _cl_event *cl_event;

#define CL_SUCCESS 0
#define CL_TRUE 1
#define CL_FALSE 0
#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_DEVICE_TYPE_ALL 0xFFFFFFFF
#define CL_PLATFORM_NAME 0x0902
#define CL_DEVICE_TYPE 0x1000
#define CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_VENDOR 0x102C
#define CL_DEVICE_VERSION 0x102F
#define CL_DEVICE_DOUBLE_FP_CONFIG 0x1032
#define CL_MEM_READ_WRITE (1 << 0)
#define CL_MEM_READ_ONLY (1 << 2)
#define CL_PROGRAM_BUILD_LOG 0x1183

typedef struct {
    cl_int (FA2_CL_CALL *GetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
    cl_int (FA2_CL_CALL *GetPlatformInfo)(cl_platform_id, cl_uint, size_t,
                                          void *, size_t *);
    cl_int (FA2_CL_CALL *GetDeviceIDs)(cl_platform_id, cl_bitfield, cl_uint,
                                       cl_device_id *, cl_uint *);
    cl_int (FA2_CL_CALL *GetDeviceInfo)(cl_device_id, cl_uint, size_t, void *,
                                        size_t *);
    cl_context (FA2_CL_CALL *CreateContext)(
        const intptr_t *, cl_uint, const cl_device_id *,
        void (FA2_CL_CALL *)(const char *, const void *, size_t, void *),
        void *, cl_int *);
    cl_command_queue (FA2_CL_CALL *CreateCommandQueue)(cl_context,
                                                       cl_device_id,
                                                       cl_bitfield, cl_int *);
    cl_program (FA2_CL_CALL *CreateProgramWithSource)(cl_context, cl_uint,
                                                      const char **,
                                                      const size_t *,
                                                      cl_int *);
    cl_int (FA2_CL_CALL *BuildProgram)(cl_program, cl_uint,
                                       const cl_device_id *, const char *,
                                       void (FA2_CL_CALL *)(cl_program, void *),
                                       void *);
    cl_int (FA2_CL_CALL *GetProgramBuildInfo)(cl_program, cl_device_id,
                                              cl_uint, size_t, void *,
                                              size_t *);
    cl_kernel (FA2_CL_CALL *CreateKernel)(cl_program, const char *, cl_int *);
    cl_mem (FA2_CL_CALL *CreateBuffer)(cl_context, cl_bitfield, size_t, void *,
                                       cl_int *);
    cl_int (FA2_CL_CALL *SetKernelArg)(cl_kernel, cl_uint, size_t,
                                       const void *);
    cl_int (FA2_CL_CALL *EnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool,
                                             size_t, size_t, const void *,
                                             cl_uint, const cl_event *,
                                             cl_event *);
    cl_int (FA2_CL_CALL *EnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool,
                                            size_t, size_t, void *, cl_uint,
                                            const cl_event *, cl_event *);
    cl_int (FA2_CL_CALL *EnqueueNDRangeKernel)(cl_command_queue, cl_kernel,
                                               cl_uint, const size_t *,
                                               const size_t *, const size_t *,
                                               cl_uint, const cl_event *,
                                               cl_event *);
    cl_int (FA2_CL_CALL *Flush)(cl_command_queue);
    cl_int (FA2_CL_CALL *Finish)(cl_command_queue);
    cl_int (FA2_CL_CALL *ReleaseMemObject)(cl_mem);
    cl_int (FA2_CL_CALL *ReleaseKernel)(cl_kernel);
    cl_int (FA2_CL_CALL *ReleaseProgram)(cl_program);
    cl_int (FA2_CL_CALL *ReleaseCommandQueue)(cl_command_queue);
    cl_int (FA2_CL_CALL *ReleaseContext)(cl_context);
} fa2_cl_api;

/* The OpenCL API, or NULL if no OpenCL runtime is installed. */
const fa2_cl_api *fa2_cl_load(void);
void fa2_cl_unload(void);

#endif

/*
 * Run-time binding of the OpenCL library. This file must not include any R
 * header (they clash with <windows.h>).
 */
#include "ocl.h"

#ifdef _WIN32
#include <windows.h>
#ifndef LOAD_LIBRARY_SEARCH_SYSTEM32
#define LOAD_LIBRARY_SEARCH_SYSTEM32 0x00000800
#endif
typedef HMODULE fa2_lib;
static fa2_lib lib_open(void)
{
    /* Graphics drivers install OpenCL.dll in the system directory; only look
     * there, so that a file of the same name elsewhere is never loaded. */
    return LoadLibraryExA("OpenCL.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
}
static void *lib_symbol(fa2_lib lib, const char *name)
{
    return (void *) GetProcAddress(lib, name);
}
static void lib_close(fa2_lib lib) { FreeLibrary(lib); }
#else
#include <dlfcn.h>
typedef void *fa2_lib;
static fa2_lib lib_open(void)
{
    static const char *const names[] = {
#ifdef __APPLE__
        "/System/Library/Frameworks/OpenCL.framework/OpenCL",
#else
        "libOpenCL.so.1", "libOpenCL.so",
#endif
        NULL};
    for (int i = 0; names[i]; i++) {
        fa2_lib lib = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
        if (lib) return lib;
    }
    return NULL;
}
static void *lib_symbol(fa2_lib lib, const char *name)
{
    return dlsym(lib, name);
}
static void lib_close(fa2_lib lib) { dlclose(lib); }
#endif

static fa2_lib the_lib = NULL;
static fa2_cl_api the_api;
static int loaded = 0;

/* Function pointers are fetched as data pointers and copied bytewise, which
 * avoids casting between object and function pointers. */
#define BIND(field, symbol)                                        \
    do {                                                           \
        void *p = lib_symbol(the_lib, symbol);                     \
        if (!p) goto fail;                                         \
        *(void **) (void *) &the_api.field = p;                    \
    } while (0)

const fa2_cl_api *fa2_cl_load(void)
{
    if (loaded) return &the_api;
    the_lib = lib_open();
    if (!the_lib) return NULL;

    BIND(GetPlatformIDs, "clGetPlatformIDs");
    BIND(GetPlatformInfo, "clGetPlatformInfo");
    BIND(GetDeviceIDs, "clGetDeviceIDs");
    BIND(GetDeviceInfo, "clGetDeviceInfo");
    BIND(CreateContext, "clCreateContext");
    BIND(CreateCommandQueue, "clCreateCommandQueue");
    BIND(CreateProgramWithSource, "clCreateProgramWithSource");
    BIND(BuildProgram, "clBuildProgram");
    BIND(GetProgramBuildInfo, "clGetProgramBuildInfo");
    BIND(CreateKernel, "clCreateKernel");
    BIND(CreateBuffer, "clCreateBuffer");
    BIND(SetKernelArg, "clSetKernelArg");
    BIND(EnqueueWriteBuffer, "clEnqueueWriteBuffer");
    BIND(EnqueueReadBuffer, "clEnqueueReadBuffer");
    BIND(EnqueueNDRangeKernel, "clEnqueueNDRangeKernel");
    BIND(Flush, "clFlush");
    BIND(Finish, "clFinish");
    BIND(ReleaseMemObject, "clReleaseMemObject");
    BIND(ReleaseKernel, "clReleaseKernel");
    BIND(ReleaseProgram, "clReleaseProgram");
    BIND(ReleaseCommandQueue, "clReleaseCommandQueue");
    BIND(ReleaseContext, "clReleaseContext");

    loaded = 1;
    return &the_api;

fail:
    lib_close(the_lib);
    the_lib = NULL;
    return NULL;
}

void fa2_cl_unload(void)
{
    if (the_lib) lib_close(the_lib);
    the_lib = NULL;
    loaded = 0;
}

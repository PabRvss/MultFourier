#pragma once

#define CL_MINIMUM_OPENCL_VERSION 120
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_ENABLE_EXCEPTIONS

#include <CL/cl.h>
#include <assert.h>
#include "common_cl.h"

#define LOCAL_WORK_SIZE 256
#define GROUPS MAX_ITER

static cl_context context;
static cl_program program;
static cl_command_queue queue;
static cl_kernel kernel_group;
static cl_kernel kernel_fast;

//__LINE__ works with GNU GCC.
#define CL_CHECK_ERROR(status,expr) \
	status = expr; \
	if(status != CL_SUCCESS) { \
		printf("Error %d in " #expr " at line: %d\n",status,__LINE__); \
		return 1; \
	}

#define CL_CHECK_STATUS(status,expr) \
	if(status != CL_SUCCESS) { \
		printf("Status error %d in " #expr " at line: %d\n",status,__LINE__); \
		return 1; \
	}

static void create_program(const char* program_name) {
    FILE* file = fopen(program_name, "rb");
    assert(file != NULL);

    fseek(file, 0, SEEK_END);
    size_t sourceSize = ftell(file);

    rewind(file);
    assert(sourceSize > 0);
    char* source = malloc(sourceSize*sizeof(char));
    size_t result = fread(source, 1, sourceSize, file);
    fclose(file);
    assert(result == sourceSize);
    program = clCreateProgramWithSource(context, 1, (const char**)&source, &sourceSize, NULL);
    free(source);
}

static int load_kernel() {
    int plat_id = 0, dev_id = 0;
    //query platforms
    cl_uint numPlatforms;
    cl_int status;
    CL_CHECK_ERROR(status,clGetPlatformIDs(0, NULL, &numPlatforms))
    assert(numPlatforms > 0);
    cl_platform_id* platforms = malloc(numPlatforms* sizeof(cl_platform_id));
    CL_CHECK_ERROR(status,clGetPlatformIDs(numPlatforms,platforms,NULL))
    cl_platform_id platform = platforms[plat_id];
    free(platforms);

    //query devices on platforms
    cl_uint numDevices;
    CL_CHECK_ERROR(status,clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,0,NULL,&numDevices))
    assert(numDevices > 0);

    cl_device_id* devices = malloc(numDevices*sizeof(cl_device_id));
    CL_CHECK_ERROR(status,clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,numDevices,devices,NULL))

    context = clCreateContext(NULL,numDevices,devices,NULL,NULL,NULL);
    cl_device_id device_id = devices[dev_id];

    queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &status);
    CL_CHECK_STATUS(status,"Create command queue")

    //load proper kernel
    create_program("src/multinomial_cl/kernel.cl");

    //build kernel
#ifdef USE_DOUBLE_PRECISION
    status = clBuildProgram(program,1, devices, "-D NO_LINTER -D USE_DOUBLE_PRECISION", NULL, NULL);
#else
    status = clBuildProgram(program,1, devices, "-D NO_LINTER", NULL, NULL);
#endif

    //check for kernel errors
    if (status != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("%s\n", log);
        free(log);
        return 1;
    }
    //kernel
    kernel_group = clCreateKernel(program, "kernel_eval_fourier", &status); CL_CHECK_STATUS(status, "Create kernel group")
    kernel_fast = clCreateKernel(program, "kernel_eval_fourier_fast", &status); CL_CHECK_STATUS(status, "Create kernel group")
    free(devices);

    return 0;
}

static int free_kernel() {
    cl_int status;

    CL_CHECK_ERROR(status,clReleaseKernel(kernel_group))
    CL_CHECK_ERROR(status,clReleaseKernel(kernel_fast))
    CL_CHECK_ERROR(status,clReleaseProgram(program))
    CL_CHECK_ERROR(status,clReleaseContext(context))
    CL_CHECK_ERROR(status,clReleaseCommandQueue(queue))

    return 0;
}

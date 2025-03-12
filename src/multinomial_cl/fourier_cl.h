#include "base_cl.h"
#pragma once

typedef struct {
    int_t N;
    int_t K;
    real_t gamma;
    real_t T;

    cl_mem matrix;
    cl_mem res_final;
} multinomial_cl;

static int create_multinomial_cl(multinomial_cl* mult, multinomial* mult0) {
    const cl_long N = mult0->N;
    const cl_long K = mult0->K;

    mult->N = mult0->N;
    mult->K = mult0->K;
    mult->gamma = mult0->gamma;
    mult->T = mult0->T;

    cl_int status;

    const size_t bytes_matrix = K*(N+1)*sizeof(global_cl_t);
    const size_t bytes_res_final = GROUPS*sizeof(result_cl_t);

    const size_t bytes_matrix_loc = (N+1)*sizeof(local_cl_t);
    const size_t bytes_res_in_loc = (N+1)*sizeof(entry_cl_t);
    const size_t bytes_res_out_loc = (N+1)*sizeof(entry_cl_t);
    if (verbose) {
        printf("GPU memory:\n");
        printf("\tglobal: %.3f KB\n", (bytes_matrix + bytes_res_final)/1024.0);
        printf("\tlocal : %zu B\n", bytes_matrix_loc + bytes_res_in_loc + bytes_res_out_loc);
    }

    //memory objects
    mult->matrix = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes_matrix, NULL, &status);
    mult->res_final = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes_res_final, NULL, &status);

    global_cl_t* matrix_host = malloc(bytes_matrix);

    for (size_t i=0; i<K*(N+1); i++) {
        matrix_host[i].reward = mult0->matrix[i].reward;
        matrix_host[i].shift = mult0->matrix[i].shift;
        matrix_host[i].log_prob = mult0->matrix[i].log_prob;
    }

    status = clEnqueueWriteBuffer(queue, mult->matrix, CL_FALSE, 0, bytes_matrix, matrix_host, 0, NULL, NULL);
    CL_CHECK_STATUS(status, "Copy matrix")

    free(matrix_host);

    //kernel arguments
    status = clSetKernelArg(kernel_group, 0, sizeof(cl_long), &N); CL_CHECK_STATUS(status, "arg 0")
    status = clSetKernelArg(kernel_group, 1, sizeof(cl_long), &K); CL_CHECK_STATUS(status, "arg 1")
    status = clSetKernelArg(kernel_group, 2, sizeof(real_t), &mult->gamma); CL_CHECK_STATUS(status, "arg 2")
    status = clSetKernelArg(kernel_group, 3, sizeof(real_t), &mult->T); CL_CHECK_STATUS(status, "arg 3")

    status = clSetKernelArg(kernel_group, 5, sizeof(cl_mem), &mult->matrix); CL_CHECK_STATUS(status, "arg 5")
    status = clSetKernelArg(kernel_group, 6, sizeof(cl_mem), &mult->res_final); CL_CHECK_STATUS(status, "arg 6")

    status = clSetKernelArg(kernel_group, 7, (N+1)*sizeof(local_cl_t), NULL); CL_CHECK_STATUS(status, "arg 7")
    status = clSetKernelArg(kernel_group, 8, (N+1)*sizeof(entry_cl_t), NULL); CL_CHECK_STATUS(status, "arg 8")
    status = clSetKernelArg(kernel_group, 9, (N+1)*sizeof(entry_cl_t), NULL); CL_CHECK_STATUS(status, "arg 9")

    return 0;
}

static int calculate_multinomial_cl(int_t trial, multinomial_cl* mult, result_cl_t* res, double* res_double) {
    cl_int status;
    cl_long offset = trial*GROUPS;
    status = clSetKernelArg(kernel_group, 4, sizeof(cl_long), &offset); CL_CHECK_STATUS(status, "arg 4")

    //execute
    const size_t lwSize = LOCAL_WORK_SIZE;
    const size_t gwSize = LOCAL_WORK_SIZE*GROUPS;
    cl_event event;
    int current = 0;
    status = clEnqueueNDRangeKernel(queue, kernel_group, 1, NULL, &gwSize, &lwSize, 0, NULL, &event);
    if (status != CL_SUCCESS) {
        printf("Kernel enqueue failed! GW Size %zu, LW Size %zu, Error code %d\n", gwSize, lwSize, status);
        return -1;
    }
    clWaitForEvents(current, &event);

    //read results
    cl_ulong bytes = GROUPS*sizeof(result_cl_t);
    status = clEnqueueReadBuffer(
            queue, mult->res_final, CL_TRUE, 0,
            bytes, res,
            0, NULL, &event);
    CL_CHECK_STATUS(status, "Enqueue read buffer")
    status = clWaitForEvents(1, &event);
    CL_CHECK_STATUS(status, "Read buffer")

    clFlush(queue);
    clFinish(queue);
    clReleaseEvent(event);

    double p = 0;
    int_t N = mult->N;
    for (int_t i=0; i<GROUPS; i++) {
        double pi = exp((double)res[i].log_mod + lgac[N])*res[i].mantissa;
        if ((trial==0) && (i==0)) pi/=2;
        p += pi;
    }
    printf("p = %.10e\n", p);

    return 0;
}

static int free_multinomial_cl(const multinomial_cl* mult) {
    cl_int status;

    CL_CHECK_ERROR(status,clReleaseMemObject(mult->matrix))
    CL_CHECK_ERROR(status,clReleaseMemObject(mult->res_final))

    return 0;
}

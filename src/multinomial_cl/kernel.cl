#ifndef NO_LINTER
#include <stdio.h>
#include <opencl-c.h>
#include "common_cl.h"
#define __kernel static
#define __constant const
#define __global
#define __local
#else
#include "src/multinomial_cl/common_cl.h"
#endif

#define LOCAL_MEMORY
#ifdef LOCAL_MEMORY
#define MEMORY __local
#else
#define MEMORY __global
#endif

#ifdef USE_DOUBLE_PRECISION
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error "Double precision floating point not supported by OpenCL implementation."
#endif
#define PI M_PI
#else
#define PI M_PI_F
#endif

typedef long int_t;

__kernel void kernel_eval_fourier(
        const int_t N,
        const int_t K,
        const real_t gamma,
        const real_t T,
        const int_t n_offset,

        __global global_cl_t* restrict matrix_glob,  // size K*(N+1)
        __global result_cl_t* restrict res_final,  // size t

        MEMORY local_cl_t* restrict matrix_loc,  // size (N+1), reused K times
        MEMORY entry_cl_t* restrict res_in_loc,  // size (N+1), reused K times
        MEMORY entry_cl_t* restrict res_out_loc  // size (N+1), reused K times
) {
    const int_t t = (int_t)get_group_id(0);
    const int_t n = n_offset + t;

    const int_t row = (int_t)get_local_id(0);
    const int_t stride = (int_t)get_local_size(0);
    const entry_cl_t res_zero = {-INFINITY, 1, 0};

    for (int_t i=row; i<=N; i+=stride) res_in_loc[i] = res_zero;
    if (row == 0) res_in_loc[0].log_mod = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int_t k=K-1; k>=0; k--) {  // reversed
        const int_t index_k = k*(N+1);
        for (int_t i=row; i<=N; i+=stride) {
            matrix_loc[i].log_prob = matrix_glob[index_k + i].log_prob;
            const real_t arg = (real_t)n*matrix_glob[index_k + i].reward;
            matrix_loc[i].sin = sincos(arg, &matrix_loc[i].cos);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int_t i=row; i<=N; i+=stride) {
            const real_t shift_i = -matrix_glob[index_k + i].shift;
            real_t res_re = 0;
            real_t res_im = 0;

            for (int_t j=0; j<=i; j++) {
                const local_cl_t cell_ij = matrix_loc[i-j];
                const real_t exponent_ij = cell_ij.log_prob;

                const real_t term_unit_re = cell_ij.cos;
                const real_t term_unit_im = cell_ij.sin;

                const entry_cl_t resj = res_in_loc[j];
                const real_t z_re = term_unit_re*resj.unit_re + term_unit_im*resj.unit_im;
                const real_t z_im = term_unit_re*resj.unit_im - term_unit_im*resj.unit_re;
                const real_t exponent_re = exponent_ij + shift_i + resj.log_mod;
                const real_t term_mod = exp(exponent_re);

                res_re += term_mod*z_re;
                res_im += term_mod*z_im;
            }

            res_out_loc[i].log_mod = -shift_i;
            res_out_loc[i].unit_re = res_re;
            res_out_loc[i].unit_im = res_im;
        }

        MEMORY entry_cl_t* temp = res_in_loc;
        res_in_loc = res_out_loc;
        res_out_loc = temp;

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row==0) {
        const entry_cl_t result = res_in_loc[N];

        const real_t s2_re = PI*gamma/T;
        const real_t s2_im = (real_t)n*PI;

        const real_t unit_re = result.unit_re;
        const real_t unit_im = result.unit_im;

        const int_t alt = -1 + 2*(n % 2);
        const real_t sinc_num = 1 + (real_t)alt*exp(-s2_re);
        const real_t modsq = s2_re*s2_re + s2_im*s2_im;
        real_t sinc_re = sinc_num*s2_re/modsq;
        real_t sinc_im = -sinc_num*s2_im/modsq;

        // Numerical issues close to the origin, use Taylor expansions instead
        if ((n == 0) && (fabs(s2_re)<=EPS)) {
            sinc_re = 1 - s2_re*(0.5 - s2_re/6) - s2_im*s2_im/6;
            sinc_im = -s2_im*(0.5 - s2_re/3);
        }

        const real_t log_mod = result.log_mod;
        const real_t mantissa = unit_re*sinc_re - unit_im*sinc_im;
        res_final[t].log_mod = log_mod;
        res_final[t].mantissa = mantissa;

//        if (row==0) printf(
//                "########################################\n"
//                "n=%lld:\n"
//                "\tgamma = %.6e\n"
//                "\tgamma/T = %.6e\n"
//                "\tlog_mod = %f\n"
//                "\tunit = %.6e + i*%.6e\n"
//                "\ts2 = %.6e + i*%.6e\n"
//                "\tsinc = %.6e + i*%.6e\n"
//                "\tres = %.6e, %.6e -> %.6e\n"
//                ,
//                n,
//                gamma,
//                gamma/T,
//                result.log_mod,
//                unit_re, unit_im,
//                s2_re, s2_im,
//                sinc_re, sinc_im,
//                res_final[t].log_mod, res_final[t].mantissa,
//                exp(res_final[t].log_mod)*mantissa
//        );
    }
}

__kernel void kernel_eval_fourier_fast(
        const int_t N,
        const int_t K,
        const real_t gamma,
        const real_t T,
        const int_t n_offset,

        __global global_cl_t* restrict matrix,  // size K*(N+1)
        __global result_cl_t* restrict res_final,  // size t

        MEMORY local_cl_t* restrict matrix_loc,  // size (N+1), reused K times
        MEMORY entry_cl_t* restrict res_in_loc,  // size (N+1), reused K times
        MEMORY entry_cl_t* restrict res_out_loc  // size (N+1), reused K times
) {
    
}

#pragma once

#define USE_DOUBLE_PRECISION
#define USE_SIMD
//#define USE_OPENCL

#define EXP exp
#define EXPC cexp
#define LOG log

#ifdef USE_SIMD
#define SIMDE_ENABLE_NATIVE_ALIASES
#define SIMDE_FAST_CONVERSION_RANGE
#include "simde/x86/sse.h"
#include "simde/x86/sse2.h"
#include "sse_mathfun/ssemathlib.c"
#endif

#ifdef USE_DOUBLE_PRECISION

typedef double complex complex_t;
typedef double real_t;
#define ABS fabs
#define MIN fmin
#define MAX fmax

#ifdef USE_SIMD

#define STRIDE 2
typedef __m128d vec_t;
#define VEC_ZERO _mm_setzero_pd()
#define VEC_ONE _mm_set_pd1(1)
#define VEC_MAX _mm_max_pd
#define VEC_EXP exp_pd
#define VEC_LOG log_pd

#endif

#define MAX_EXP 700
#define CONV_COUNT 30

#else

typedef float complex complex_t;
typedef float real_t;
#define ABS fabsf
#define MIN fminf
#define MAX fmaxf

#ifdef USE_SIMD

#define STRIDE 4
typedef __m128 vec_t;
#define VEC_ZERO _mm_setzero_ps()
#define VEC_ONE _mm_set_ps1(1)
#define VEC_MAX _mm_max_ps
#define VEC_EXP exp_ps
#define VEC_LOG log_ps

#endif

#define MAX_EXP 85
#define CONV_COUNT 40

#endif

#ifndef USE_SIMD

#define STRIDE 1
typedef real_t vec_t;
#define VEC_ZERO 0
#define VEC_ONE 1
#define VEC_MAX MAX
#define VEC_EXP EXP
#define VEC_LOG LOG

#endif

#if defined(__GNUC__)
#define EXTERNAL __attribute__((visibility("default")))
#define PACK( __Declaration__ ) __Declaration__ __attribute__((__packed__))
#elif defined(_MSC_VER)
#define EXTERNAL __declspec(dllexport)
#define PACK( __Declaration__ ) __pragma( pack(push, 1) ) __Declaration__ __pragma( pack(pop))
#endif

typedef int64_t int_t;

#define EPS 1e-10
static real_t* restrict lgac;

static inline double get_dt(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec-start.tv_sec) + (double)(end.tv_nsec-start.tv_nsec)*1e-9;
}

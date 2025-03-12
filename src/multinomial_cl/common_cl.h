#pragma once

#ifdef USE_DOUBLE_PRECISION
typedef double real_t;
#else
typedef float real_t;
#endif

#if defined(__GNUC__)
#define EXTERNAL __attribute__((visibility("default")))
//#define PACK( __Declaration__ ) __Declaration__ __attribute__((__packed__))
//#define PACK typedef struct __attribute__((__packed__))
#elif defined(_MSC_VER)
#define EXTERNAL __declspec(dllexport)
#define PACK( __Declaration__ ) __pragma( pack(push, 1) ) __Declaration__ __pragma( pack(pop))
#endif

#define EPS 1e-10

typedef struct __attribute__((__packed__)) {
    real_t log_prob;  // term i-j
    real_t reward;
    real_t shift;  // exponent shift i
} global_cl_t;

typedef struct __attribute__((__packed__)) {
    real_t log_prob;  // term i-j
    real_t cos;  // cosine i-j
    real_t sin;  // sine i-j
} local_cl_t;

typedef struct __attribute__((__packed__)) {
    real_t log_mod;
    real_t unit_re;
    real_t unit_im;
} entry_cl_t;

typedef struct __attribute__((__packed__)) {
    real_t log_mod;
    real_t mantissa;
} result_cl_t;

#pragma once
#include "base_fast.h"
#include <string.h>

static void fast_convolution(
        const int_t N,
        complex_t* A, complex_t* B, complex_t* C,
        complex_t* F,
        complex_t f, const int_t threshold_prod
    ) {
    F[0] = f;

    int_t N_curr = N;
    int_t blocks = 1;

    for (; N_curr > threshold_prod; N_curr>>=1) {
        const int_t width = N_curr/2;
        for (int_t i=0; i<blocks; i++) {
            const int_t base = i*N_curr;
            const complex_t f_sqrt = sqrt(F[base]);

            for (int_t j = 0; j<width; j++) {
                const complex_t a1 = A[base + j];
                const complex_t a2 = f_sqrt*A[base + width + j];
                A[base + j] = a1 + a2;
                A[base + width + j] = a1 - a2;

                const complex_t b1 = f_sqrt*B[base + j];
                const complex_t b2 = B[base + width + j];
                B[base + j] = b1 + b2;
                B[base + width + j] = b1 - b2;
            }

            F[base] = f_sqrt;
            F[base + width] = -f_sqrt;
        }
        blocks <<= 1;
    }

    for (int_t b=0; b<blocks; b++) {
        const int_t width = N_curr;
        const int_t base = b*width;

        for (int_t i=0; i<width; i++) {
            complex_t acc = 0;
            for (int_t k=0; k<i; k++) acc += A[base + width + k - i]*B[base + k];
            acc *= F[base];
            for (int_t k=i; k<width; k++) acc += A[base + k - i]*B[base + k];
            C[base + i] = acc;
        }
    }

    blocks >>= 1;
    for (; N_curr < N; N_curr<<=1) {
        const int_t width2 = N_curr;
        for (int_t i=0; i<blocks; i++) {
            const int_t base = i*width2*2;
            const complex_t f_sqrt = F[base];
            const complex_t denom = 1/f_sqrt;

            for (int_t j = 0; j<width2; j++) {
                const complex_t m1 = C[base + j];
                const complex_t m2 = C[base + width2 + j];
                C[base + j] = (m1 + m2)*denom;
                C[base + width2 + j] = m1 - m2;
            }

            F[base] = f_sqrt*f_sqrt;
        }
        blocks >>= 1;
    }
}

static void fast_convolution0(
        const int_t N,
        complex_t* A, complex_t* B, complex_t* C,
        complex_t* A1, complex_t* B1, complex_t* C1,
        complex_t* F,
        const int_t threshold_prod
) {
    real_t factor = 0.5;
    for (int_t N_curr = N; N_curr > threshold_prod; N_curr>>=1) factor /= 2;
    for (int_t i=0; i<N; i++) B[i] = B[i]*factor;

    size_t bytes = N*sizeof(complex_t);
    memcpy(A1, A, bytes);
    memcpy(B1, B, bytes);

    fast_convolution(N, A1, B1, C1, F, 1, threshold_prod);
    fast_convolution(N, A, B, C, F, -1, threshold_prod);
    for (int_t i=0; i<N; i++) C[i] += C1[i];
}

static void slow_convolution(int_t N, const complex_t* A, const complex_t* B, complex_t* C) {
    for (int_t i=0; i<N; i++) {
        complex_t acc = 0;
        for (int_t j=0; j<=i; j++) acc += A[i - j]*B[j];
        C[i] = acc;
    }
}

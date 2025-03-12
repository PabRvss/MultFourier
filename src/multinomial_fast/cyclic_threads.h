#include "base_fast.h"
#pragma once

typedef struct __attribute__((__packed__)) {
    complex_t a;
    complex_t b;
    complex_t f;
    complex_t c;
} cell_fast_t;

static void fast_convolution_threads(  // TODO: check correct results
        const int_t N, const int_t threads,
        const int_t threshold_prod,
        cell_fast_t* restrict matrix, cell_fast_t* restrict matrix_loc
) {
    real_t factor = 1;
    for (int_t N_curr = N; N_curr > threshold_prod; N_curr>>=1) factor /= 2;
    for (int_t i=0; i<N; i++) {
        matrix[i].b = matrix[i].b*factor;
        matrix[i].f = 1;
    }

    int_t N_curr = N;
    int_t blocks = 1;

    for (; N_curr>threshold_prod; N_curr>>=1) {
        const int_t width = N_curr/2;

        for (int_t t=0; t<threads; t++) {
            for (int_t b=0; b<blocks; b++) {

                const int_t i_min = b*N_curr + t;
                const int_t i_max = b*N_curr + width;
                const complex_t f_sqrt = sqrt(matrix[i_min].f);
                const complex_t f_sqrt_min = -f_sqrt;

                for (int_t i = i_min; i < i_max; i += threads) {
                    const complex_t a1 = matrix[i].a;
                    const complex_t a2 = f_sqrt * matrix[i+width].a;
                    matrix[i].a = a1 + a2;
                    matrix[i+width].a = a1 - a2;

                    const complex_t b1 = f_sqrt * matrix[i].b;
                    const complex_t b2 = matrix[i+width].b;
                    matrix[i].b = b1 + b2;
                    matrix[i+width].b = b1 - b2;

                    matrix[i_min].f = f_sqrt;
                    matrix[i_min+width].f = f_sqrt_min;
                }
            }
        }

        blocks <<= 1;
        // sync barrier here
    }

    int_t width = N_curr;

    for (int_t b = 0; b < blocks; b++) {  // TODO: use local matrix
        const int_t base = b*N_curr;

        for (int_t t=0; t<threads; t++) {
            const int_t i_min = base + t;
            const int_t i_max = base + width;
            const complex_t f = matrix[i_min].f;

            for (int_t i = i_min; i < i_max; i += threads) {
                const int_t i0 = i - base;
                complex_t acc = 0;
                for (int_t k = 0; k < i0; k++) acc += matrix[base + width + k - i0].a * matrix[base + k].b;
                acc *= f;
                for (int_t k = i0; k < width; k++) acc += matrix[base + k - i0].a * matrix[base + k].b;
                matrix[i].c = acc;
            }
        }

        // sync barrier here
    }

    blocks >>= 1;
    for (; N_curr < N; N_curr<<=1) {
        const int_t width2 = N_curr;

        for (int_t t=0; t<threads; t++) {
            for (int_t b=0; b<blocks; b++) {
                const int_t base = b*2*N_curr;
                const int_t i_min = base + t;
                const int_t i_max = base + width2;
                const complex_t f_sqrt = matrix[i_min].f;
                const complex_t f = f_sqrt*f_sqrt;
                const complex_t denom = 1/f_sqrt;

                for (int_t i = i_min; i < i_max; i += threads) {
                    const complex_t m1 = matrix[i].c;
                    const complex_t m2 = matrix[i + width2].c;
                    matrix[i].c = (m1 + m2)*denom;
                    matrix[i].f = f;
                    matrix[width2 + i].f = f;
                    matrix[width2 + i].c = m1 - m2;
                }
            }

        }

        blocks >>= 1;
        // sync barrier here
    }

    for (int_t i=N/2; i<N; i++) matrix[i].c = 0;
}

void main_fast() {
    int_t N0 = 1 << 10;
    int_t trials = 1000;
    int_t threshold_prod = 1 << 3;
    int_t threads = 1;
    printf("N = %lld\n", N0);

    size_t bytes = N0*sizeof(complex_t);
    complex_t *A, *B, *C;
    A = malloc(bytes);
    B = malloc(bytes);
    C = malloc(bytes);

    for (int_t i=0; i<N0; i++) {
        A[i] = i+1;
        B[i] = i+1;
    }
//    printf("A = ["); for (int_t i=0; i<N0; i++) printf("%.3f, ", REAL(A[i])); printf("]\n");
//    printf("b = ["); for (int_t i=0; i<N0; i++) printf("%.3f, ", REAL(B[i])); printf("]\n");

    struct timespec start, end;
    timespec_get(&start, TIME_UTC);
    for (int_t t=0; t<trials; t++) slow_convolution(N0, A, B, C);
    timespec_get(&end, TIME_UTC);
    double t_slow = get_dt(start, end);

//    printf("C = ["); for (int_t i=0; i<N0; i++) printf("%.3f, ", REAL(C[i])); printf("]\n");
    printf("slow convolution time: %.6f seconds\n", t_slow);

    complex_t* A1 = malloc(bytes);
    complex_t* B1 = malloc(bytes);
    complex_t* C1 = malloc(bytes);
    complex_t* F = malloc(bytes);

    for (int_t i=0; i<N0; i++) {
        A[i] = i+1;
        B[N0-1-i] = i+1;  // reversed B values
    }

    cell_fast_t* matrix = malloc(2*N0*sizeof(cell_fast_t));
    cell_fast_t* matrix_loc = malloc(2*N0*sizeof(cell_fast_t));
    for (int_t i=0; i<N0; i++) {
        matrix[i].a = A[i];
        matrix[i].b = B[i];

        matrix[i+N0].a = 0;
        matrix[i+N0].b = 0;
    }
    matrix[0].f = 1;

//    printf("A1 = ["); for (int_t i=0; i<N0; i++) printf("%.3f, ", REAL(A[i])); printf("]\n");
//    printf("b1 = ["); for (int_t i=0; i<N0; i++) printf("%.3f, ", REAL(B[i])); printf("]\n");

    timespec_get(&start, TIME_UTC);
    for (int_t t=0; t<trials; t++) fast_convolution0(N0, A, B, C, A1, B1, C1, F, threshold_prod);
    timespec_get(&end, TIME_UTC);
    double t_fast = get_dt(start, end);

    printf("fast convolution time: %.6f seconds\n", t_fast);
    printf("speedup: %.3f\n", t_slow/t_fast);

//    printf("A2 = ["); for (int_t i=0; i<2*N0; i++) printf("%.3f, ", REAL(matrix[i].a)); printf("]\n");
//    printf("b2 = ["); for (int_t i=0; i<2*N0; i++) printf("%.3f, ", REAL(matrix[i].b)); printf("]\n");

    timespec_get(&start, TIME_UTC);
    for (int_t t=0; t<trials; t++) fast_convolution_threads(2*N0, threads, threshold_prod, matrix, matrix_loc);
    timespec_get(&end, TIME_UTC);
    double t_threads = get_dt(start, end);
    printf("threads convolution time: %.6f seconds\n", t_threads);
    printf("\n");

//    printf("C1 = ["); for (int_t i=0; i<2*N0; i++) printf("%.3f, ", REAL(C[i])); printf("]\n");
//    printf("C2 = ["); for (int_t i=0; i<2*N0; i++) printf("%.3f, ", REAL(matrix[i].c)); printf("]\n");

    free(matrix); free(matrix_loc);
    free(A); free(B); free(C);
    free(A1); free(B1); free(C1); free(F);
}

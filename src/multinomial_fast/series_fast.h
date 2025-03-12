#pragma once
#include "base_fast.h"

static real_t eval_fourier_fast(
        const multinomial* restrict mult, const int_t n,
        result_t* restrict res_arr
) {
    return 0;
}

static void series_fast(multinomial* mult, multinomial_result* mult_res, options_t* options) {
    const int_t verbose = options->verbose;
    int_t N = mult->N;
    result_t* res_arr = malloc((N+1)*sizeof(result_t));

    int_t max_count = CONV_COUNT;
    int_t max_iter = options->max_iter;
    if (verbose) printf("Running scalar\n");

    if (verbose) printf("p0 = %.6e\n", mult->p0);

    const real_t threshold = 1 + options->eps_rel/(real_t)max_count;
    real_t delta_acc = 0;
    real_t bound = INFINITY;

    real_t p = 0;
    real_t c = 0;

    int converged_full = 0;
    int converged_count = 0;
    real_t p_prev = INFINITY;

    int_t n = 0;
    for (; n <= max_iter; n++) {
        if (!isfinite(p) || isnan(p)) break;

        real_t delta = eval_fourier_fast(mult, n, res_arr);
        if (n==0) {
            real_t denom = ABS(1-EXPC(I*mult->T));
            bound = delta/denom;
            delta/=2;
        }
        real_t y = delta - c;
        real_t t = p + y;
        c = (t - p) - y;
        p = t;

        const real_t tail_bound = bound*(1+LOG(n+1))/(real_t)(n+1);

        if (verbose) {
            delta_acc += delta;
            if (n % options->print_freq == 0) {
                printf("%04lld\t%.6e\t%.6e\t%.6e\n", n, p, delta_acc/options->print_freq, tail_bound);
                delta_acc = 0;
            }
        }

        const real_t frac = p/p_prev;
        const int converged = (p > 0.0) & (frac <= threshold) & (1/frac <= threshold);
        const int converged_tail = (p > 0.0) & (tail_bound/p <= options->eps_rel);
        converged_count = converged ? converged_count+1 : 0;
        p_prev = p;

        if ((converged_count >= max_count) | converged_tail) {
            if (verbose) printf("%04lld\t%.6e\t%.6e\n", n, p, delta_acc);
            converged_full = 1;
            break;
        }
    }
    p += mult->p0/2;
    if (verbose) {
        printf("p = %.9e\n", p);
        printf("converged: "); printf(converged_full ? "true" : "false"); printf("\n");
    }

    mult_res->pval = p;
    mult_res->converged = converged_full;
    mult_res->terms = n-1;

    free(res_arr);
}

#include "R.h"
#include "Rinternals.h"
#include <stdint.h>

#include <stdio.h>
#include <stdlib.h>
#include <tgmath.h>
#include <time.h>
#include <pthread.h>

#include "base.h"
#include "options.h"

#include "rewards.h"
#include "multinomial/multinomial.h"
#include "multinomial/fourier.h"

#include "multinomial_fast/multinomial_fast.h"
#include "multinomial_fast/cyclic.h"
#include "multinomial_fast/cyclic_threads.h"
#include "multinomial_fast/series_fast.h"

#ifdef USE_OPENCL
#include "multinomial_cl/base_cl.h"
#include "multinomial_cl/fourier_cl.h"
#endif

#include "exhaustive.h"

// TODO:
// OpenCL global/local memory selection
// OpenCL chunks of series terms
// gamma optimization with OpenCL
// enumeration recursive -> iterative

extern void init_ext(int_t fact_max, int_t verbose_ext) {
  if (lgac!=NULL) free(lgac);

  lgac = malloc(fact_max*sizeof(real_t));
  lgac[0] = 0;

  double s = 0;
  double d = 0;
  for (size_t i=1; i <= fact_max; i++) {
    double l = log((double)i);

    double y = l - d;
    double t = s + y;
    d = (t - s) - y;
    s = t;
    lgac[i] = (real_t)s;
  }
}

extern void finish() {
  if (lgac) {
    free(lgac);
    lgac = NULL;
  }
}


extern void main_int(
    const int_t N, const int_t K,
    const int_t* x0, const real_t* probs,
    multinomial_result* mult_res,
    options_t* options
) {
  const int_t verbose = options->verbose;
  real_t supp = (lgac[N + K - 1] - lgac[N] - lgac[K - 1]) / LOG(10.0);

  if (verbose) {
    printf("p = [");
    //        for (int_t k=0; k<K; k++) printf("%.8f, ", probs[k]);
    for (int_t k = 0; k < K; k++) printf("%.8f ", probs[k]);
    printf("]\n");
    printf("x0 = [");
    for (int_t k = 0; k < K; k++) printf("%lld, ", x0[k]);
    printf("]\n");
    printf("expected = [");
    for (int_t k = 0; k < K; k++) printf("%.3f, ", N*probs[k]);
    printf("]\n");
    printf("N = %lld\n", N);
    printf("K = %lld\n", K);
    printf("log10 support = %.6f\n", supp);
    printf("log10 k! = %.6f\n", lgac[K] / LOG(10));
    printf("\n");
  }

  struct timespec t0, t1, t2;
  timespec_get(&t0, TIME_UTC);

  multinomial* mult = get_mult(N, K, probs, options);
  fill_mult1(mult, x0, options);
  if (supp >= options->enum_cutoff) fill_mult2(mult, x0, options);
  mult_res->p0 = mult->p0;
  timespec_get(&t1, TIME_UTC);
  if (verbose) printf("CPU init time: %.6f seconds\n\n", get_dt(t0, t1));
  if (supp >= options->enum_cutoff) {
    series(mult, mult_res, options);
    timespec_get(&t2, TIME_UTC);
    if (verbose) printf("CPU series time: %.6f seconds\n", get_dt(t1, t2));
  } else {
    exhaustive(mult, mult_res, options);
    timespec_get(&t2, TIME_UTC);
    if (verbose) printf("exhaustive time: %.6f seconds\n", get_dt(t1, t2));
  }
  if (verbose) printf("CPU total time: %.6f seconds\n\n", get_dt(t0, t2));

#ifdef USE_OPENCL
  load_kernel();

  timespec_get(&t0, TIME_UTC);
  multinomial_cl* mult_cl = malloc(sizeof(multinomial_cl));
  create_multinomial_cl(mult_cl, mult);
  timespec_get(&t1, TIME_UTC);
  printf("CL init time: %.6f seconds\n", get_dt(t0, t1));

  result_cl_t* res = malloc(GROUPS*sizeof(result_cl_t));
  double* res_double = malloc(GROUPS*sizeof(double));
  calculate_multinomial_cl(0, mult_cl, res, res_double);
  free_multinomial_cl(mult_cl);

  free(mult_cl);
  free(res);
  free(res_double);
  free_kernel();

  timespec_get(&t2, TIME_UTC);
  printf("CL series time: %.6f seconds\n", get_dt(t0, t2));
#endif

  free_mult(mult);
  mult = NULL;
}

extern void main_ext(
    const int_t N, const int_t K,
    const int_t* x0, const real_t* probs,
    double* restrict p_value,
    int_t* restrict converged,
    int_t* restrict terms,
    double* restrict p0,
    real_t rel_eps,
    int_t max_terms,
    real_t undersampling,
    int_t verbose
) {
  multinomial_result* mult_res = malloc(sizeof(multinomial_result));
  options_t options = {0};  // Sets initial values at 0
  set_options(&options, rel_eps, max_terms, undersampling, 1e-2, 6.5, 8, 1,
              verbose, 10);

  main_int(N, K, x0, probs, mult_res, &options);

  p_value[0] = mult_res->pval;
  converged[0] = mult_res->converged;
  terms[0] = mult_res->terms;
  p0[0] = mult_res->p0;

  free(mult_res);
}

extern SEXP pval_flexible(SEXP N, SEXP K, SEXP x0, SEXP probs, SEXP max_terms,
                          SEXP rel_eps, SEXP undersampling, SEXP verbose) {
  // Converts parameters to int64_t and double
  int64_t n = (int64_t) REAL(N)[0];
  int64_t k = (int64_t) REAL(K)[0];
  real_t rel_eps_val = REAL(rel_eps)[0];          // real_t is the same as double
  int64_t max_terms_val = (int64_t) REAL(max_terms)[0]; // Converts to int_t
  real_t undersampling_val = REAL(undersampling)[0]; // real_t is the same as double
  int64_t verbose_val = (int64_t) REAL(verbose)[0];  // Converts to int_t

  // Convert x0 to an int64_t array
  int64_t *x0_ptr = (int64_t*) malloc(k * sizeof(int64_t));
  if (x0_ptr == NULL) {
    error("Error: Failed to allocate memory for x0.");
  }
  for (int i = 0; i < k; i++) {
    x0_ptr[i] = (int64_t) REAL(x0)[i];  // Cast to int64_t
  }

  // Convert probs to a double array
  double *probs_ptr = (double*) malloc(k * sizeof(double));
  if (probs_ptr == NULL) {
    free(x0_ptr);
    error("Error: Failed to allocate memory for probs.");
  }
  for (int i = 0; i < k; i++) {
    probs_ptr[i] = REAL(probs)[i];  // Directly double
  }

  // Initialize lgac
  init_ext(n + k, 1);

  // Output variables
  double p_value;
  int64_t converged;
  int64_t terms;
  double p0;

  // Call the C function
  main_ext(n, k, x0_ptr, probs_ptr, &p_value, &converged, &terms, &p0,
           rel_eps_val, max_terms_val, undersampling_val, verbose_val);

  // Free allocated memory
  free(x0_ptr);
  free(probs_ptr);

  // Finalize
  finish();

  // Create an R list with the results
  SEXP result = PROTECT(allocVector(VECSXP, 4));
  SET_VECTOR_ELT(result, 0, ScalarReal(p_value));
  SET_VECTOR_ELT(result, 1, ScalarInteger(converged));
  SET_VECTOR_ELT(result, 2, ScalarInteger(terms));
  SET_VECTOR_ELT(result, 3, ScalarReal(p0));

  UNPROTECT(1);
  return result;
}

// Función que utiliza exclusivamente el método exhaustive
extern void main_exhaustive(
    const int_t N, const int_t K,
    const int_t* x0, const real_t* probs,
    double* restrict p_value,
    int_t* restrict converged,
    int_t* restrict terms,
    double* restrict p0,
    real_t rel_eps,
    int_t max_terms,
    real_t undersampling,
    int_t verbose
) {
  multinomial_result* mult_res = malloc(sizeof(multinomial_result));
  options_t options = {0};  // Sets initial values at 0
  set_options(&options, rel_eps, max_terms, undersampling, 1e-2, 6.5, 8, 1,
              verbose, 10);

  // Forzar el uso del método exhaustive
  options.enum_cutoff = INFINITY;  // Esto asegura que siempre se use exhaustive

  main_int(N, K, x0, probs, mult_res, &options);

  p_value[0] = mult_res->pval;
  converged[0] = mult_res->converged;
  terms[0] = mult_res->terms;
  p0[0] = mult_res->p0;

  free(mult_res);
}

// Función que utiliza exclusivamente el método series
extern void main_series(
    const int_t N, const int_t K,
    const int_t* x0, const real_t* probs,
    double* restrict p_value,
    int_t* restrict converged,
    int_t* restrict terms,
    double* restrict p0,
    real_t rel_eps,
    int_t max_terms,
    real_t undersampling,
    int_t verbose
) {
  multinomial_result* mult_res = malloc(sizeof(multinomial_result));
  options_t options = {0};  // Sets initial values at 0
  set_options(&options, rel_eps, max_terms, undersampling, 1e-2, 6.5, 8, 1,
              verbose, 10);

  // Forzar el uso del método series
  options.enum_cutoff = -INFINITY;  // Esto asegura que siempre se use series

  main_int(N, K, x0, probs, mult_res, &options);

  p_value[0] = mult_res->pval;
  converged[0] = mult_res->converged;
  terms[0] = mult_res->terms;
  p0[0] = mult_res->p0;

  free(mult_res);
}

// Funciones R que llaman a las nuevas funciones C
extern SEXP pval_exhaustive(SEXP N, SEXP K, SEXP x0, SEXP probs, SEXP max_terms,
                            SEXP rel_eps, SEXP undersampling, SEXP verbose) {
  // Convierte los parámetros a int64_t y double
  int64_t n = (int64_t) REAL(N)[0];
  int64_t k = (int64_t) REAL(K)[0];
  real_t rel_eps_val = REAL(rel_eps)[0];
  int64_t max_terms_val = (int64_t) REAL(max_terms)[0];
  real_t undersampling_val = REAL(undersampling)[0];
  int64_t verbose_val = (int64_t) REAL(verbose)[0];

  // Convierte x0 a un array de int64_t
  int64_t *x0_ptr = (int64_t*) malloc(k * sizeof(int64_t));
  if (x0_ptr == NULL) {
    error("Error: Failed to allocate memory for x0.");
  }
  for (int i = 0; i < k; i++) {
    x0_ptr[i] = (int64_t) REAL(x0)[i];
  }

  // Convierte probs a un array de double
  double *probs_ptr = (double*) malloc(k * sizeof(double));
  if (probs_ptr == NULL) {
    free(x0_ptr);
    error("Error: Failed to allocate memory for probs.");
  }
  for (int i = 0; i < k; i++) {
    probs_ptr[i] = REAL(probs)[i];
  }

  // Inicializa lgac
  init_ext(n + k, 1);

  // Variables de salida
  double p_value;
  int64_t converged;
  int64_t terms;
  double p0;

  // Llama a la función C que usa exhaustive
  main_exhaustive(n, k, x0_ptr, probs_ptr, &p_value, &converged, &terms, &p0,
                  rel_eps_val, max_terms_val, undersampling_val, verbose_val);

  // Libera la memoria asignada
  free(x0_ptr);
  free(probs_ptr);

  // Finaliza
  finish();

  // Crea una lista R con los resultados
  SEXP result = PROTECT(allocVector(VECSXP, 4));
  SET_VECTOR_ELT(result, 0, ScalarReal(p_value));
  SET_VECTOR_ELT(result, 1, ScalarInteger(converged));
  SET_VECTOR_ELT(result, 2, ScalarInteger(terms));
  SET_VECTOR_ELT(result, 3, ScalarReal(p0));

  UNPROTECT(1);
  return result;
}

extern SEXP pval_series(SEXP N, SEXP K, SEXP x0, SEXP probs, SEXP max_terms,
                        SEXP rel_eps, SEXP undersampling, SEXP verbose) {
  // Convierte los parámetros a int64_t y double
  int64_t n = (int64_t) REAL(N)[0];
  int64_t k = (int64_t) REAL(K)[0];
  real_t rel_eps_val = REAL(rel_eps)[0];
  int64_t max_terms_val = (int64_t) REAL(max_terms)[0];
  real_t undersampling_val = REAL(undersampling)[0];
  int64_t verbose_val = (int64_t) REAL(verbose)[0];

  // Convierte x0 a un array de int64_t
  int64_t *x0_ptr = (int64_t*) malloc(k * sizeof(int64_t));
  if (x0_ptr == NULL) {
    error("Error: Failed to allocate memory for x0.");
  }
  for (int i = 0; i < k; i++) {
    x0_ptr[i] = (int64_t) REAL(x0)[i];
  }

  // Convierte probs a un array de double
  double *probs_ptr = (double*) malloc(k * sizeof(double));
  if (probs_ptr == NULL) {
    free(x0_ptr);
    error("Error: Failed to allocate memory for probs.");
  }
  for (int i = 0; i < k; i++) {
    probs_ptr[i] = REAL(probs)[i];
  }

  // Inicializa lgac
  init_ext(n + k, 1);

  // Variables de salida
  double p_value;
  int64_t converged;
  int64_t terms;
  double p0;

  // Llama a la función C que usa series
  main_series(n, k, x0_ptr, probs_ptr, &p_value, &converged, &terms, &p0,
              rel_eps_val, max_terms_val, undersampling_val, verbose_val);

  // Libera la memoria asignada
  free(x0_ptr);
  free(probs_ptr);

  // Finaliza
  finish();

  // Crea una lista R con los resultados
  SEXP result = PROTECT(allocVector(VECSXP, 4));
  SET_VECTOR_ELT(result, 0, ScalarReal(p_value));
  SET_VECTOR_ELT(result, 1, ScalarInteger(converged));
  SET_VECTOR_ELT(result, 2, ScalarInteger(terms));
  SET_VECTOR_ELT(result, 3, ScalarReal(p0));

  UNPROTECT(1);
  return result;
}

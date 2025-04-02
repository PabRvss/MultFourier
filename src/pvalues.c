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

// Definición de constantes
#define DEFAULT_REL_EPS 1e-2
#define DEFAULT_ENUM_CUTOFF 6.5
#define DEFAULT_UNDERSAMPLING 1.0
#define DEFAULT_VERBOSE 0
#define DEFAULT_MAX_TERMS 1000000

// Initialize the lgac array for factorial calculations
extern void init_ext(int64_t fact_max, int64_t verbose_ext) {
  if (lgac != NULL) {
    free(lgac);
    lgac = NULL;
  }

  lgac = malloc(fact_max * sizeof(real_t));
  if (lgac == NULL) {
    error("Error: Failed to allocate memory for lgac array.");
    return;
  }

  lgac[0] = 0;

  double s = 0;
  double d = 0;
  for (size_t i = 1; i <= fact_max; i++) {
    double l = log((double)i);

    double y = l - d;
    double t = s + y;
    d = (t - s) - y;
    s = t;
    lgac[i] = (real_t)s;
  }
}

// Clean up and free the lgac array
extern void finish() {
  if (lgac) {
    free(lgac);
    lgac = NULL;
  }
}

// Main internal function to compute p-values
extern void main_int(
    const int64_t N, const int64_t K,
    const int64_t* x0, const real_t* probs,
    multinomial_result* mult_res,
    options_t* options
) {
  // Verificación de punteros NULL
  if (x0 == NULL || probs == NULL || mult_res == NULL || options == NULL) {
    error("Error: NULL pointer passed to main_int");
    return;
  }

  const int64_t verbose = options->verbose;
  real_t supp = (lgac[N + K - 1] - lgac[N] - lgac[K - 1]) / LOG(10.0);

  if (verbose) {
    printf("p = [");
    for (int64_t k = 0; k < K; k++) printf("%.8f ", probs[k]);
    printf("]\n");
    printf("x0 = [");
    for (int64_t k = 0; k < K; k++) printf("%lld, ", x0[k]);
    printf("]\n");
    printf("expected = [");
    for (int64_t k = 0; k < K; k++) printf("%.3f, ", N * probs[k]);
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
  if (mult == NULL) {
    error("Error: Failed to create multinomial structure");
    return;
  }

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
  if (!load_kernel()) {
    error("Error: Failed to load OpenCL kernel");
    free_mult(mult);
    return;
  }

  timespec_get(&t0, TIME_UTC);
  multinomial_cl* mult_cl = malloc(sizeof(multinomial_cl));
  if (mult_cl == NULL) {
    error("Error: Failed to allocate memory for multinomial_cl");
    free_mult(mult);
    return;
  }

  if (!create_multinomial_cl(mult_cl, mult)) {
    error("Error: Failed to create multinomial_cl structure");
    free(mult_cl);
    free_mult(mult);
    return;
  }

  timespec_get(&t1, TIME_UTC);
  printf("CL init time: %.6f seconds\n", get_dt(t0, t1));

  result_cl_t* res = malloc(GROUPS * sizeof(result_cl_t));
  double* res_double = malloc(GROUPS * sizeof(double));
  if (res == NULL || res_double == NULL) {
    error("Error: Failed to allocate memory for OpenCL results");
    if (res) free(res);
    if (res_double) free(res_double);
    free_multinomial_cl(mult_cl);
    free(mult_cl);
    free_mult(mult);
    return;
  }

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
}

// External function to compute p-values
extern void main_ext(
    const int64_t N, const int64_t K,
    const int64_t* x0, const real_t* probs,
    double* restrict p_value,
    int64_t* restrict converged,
    int64_t* restrict terms,
    double* restrict p0,
    real_t rel_eps,
    int64_t max_terms,
    real_t undersampling,
    int64_t verbose
) {
  if (p_value == NULL || converged == NULL || terms == NULL || p0 == NULL) {
    error("Error: NULL pointer passed to main_ext");
    return;
  }

  multinomial_result* mult_res = malloc(sizeof(multinomial_result));
  if (mult_res == NULL) {
    error("Error: Failed to allocate memory for multinomial_result");
    return;
  }

  options_t options = {0};
  set_options(&options, rel_eps, max_terms, undersampling, DEFAULT_REL_EPS,
              DEFAULT_ENUM_CUTOFF, 8, 1, verbose, 10);

  main_int(N, K, x0, probs, mult_res, &options);

  *p_value = mult_res->pval;
  *converged = mult_res->converged;
  *terms = mult_res->terms;
  *p0 = mult_res->p0;

  free(mult_res);
}

// R interface for the flexible p-value computation
extern SEXP pval_flexible(SEXP N, SEXP K, SEXP x0, SEXP probs, SEXP max_terms,
                          SEXP rel_eps, SEXP undersampling, SEXP verbose) {
  // Convert parameters to int64_t and double
  int64_t n = (int64_t) REAL(N)[0];
  int64_t k = (int64_t) REAL(K)[0];
  real_t rel_eps_val = REAL(rel_eps)[0];
  int64_t max_terms_val = (int64_t) REAL(max_terms)[0];
  real_t undersampling_val = REAL(undersampling)[0];
  int64_t verbose_val = (int64_t) REAL(verbose)[0];

  // Convert x0 to an int64_t array
  int64_t* x0_ptr = (int64_t*) malloc(k * sizeof(int64_t));
  if (x0_ptr == NULL) {
    error("Error: Failed to allocate memory for x0.");
    return R_NilValue;
  }
  for (int i = 0; i < k; i++) {
    x0_ptr[i] = (int64_t) REAL(x0)[i];
  }

  // Convert probs to a double array
  double* probs_ptr = (double*) malloc(k * sizeof(double));
  if (probs_ptr == NULL) {
    free(x0_ptr);
    error("Error: Failed to allocate memory for probs.");
    return R_NilValue;
  }
  for (int i = 0; i < k; i++) {
    probs_ptr[i] = REAL(probs)[i];
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

// Function to force the use of the series method
extern void main_series(
    const int64_t N, const int64_t K,
    const int64_t* x0, const real_t* probs,
    double* restrict p_value,
    int64_t* restrict converged,
    int64_t* restrict terms,
    double* restrict p0,
    real_t rel_eps,
    int64_t max_terms,
    real_t undersampling,
    int64_t verbose
) {
  if (p_value == NULL || converged == NULL || terms == NULL || p0 == NULL) {
    error("Error: NULL pointer passed to main_series");
    return;
  }

  multinomial_result* mult_res = malloc(sizeof(multinomial_result));
  if (mult_res == NULL) {
    error("Error: Failed to allocate memory for multinomial_result");
    return;
  }

  options_t options = {0};
  set_options(&options, rel_eps, max_terms, undersampling, DEFAULT_REL_EPS,
              -INFINITY, 8, 1, verbose, 10); // Force series method

  main_int(N, K, x0, probs, mult_res, &options);

  *p_value = mult_res->pval;
  *converged = mult_res->converged;
  *terms = mult_res->terms;
  *p0 = mult_res->p0;

  free(mult_res);
}

// R interface for the exhaustive p-value computation
// Nueva versión modificada de pval_exhaustive que recibe 6 parámetros
extern SEXP pval_exhaustive(SEXP N, SEXP K, SEXP x0, SEXP probs, SEXP max_time, SEXP verbose) {
  // Validación básica de parámetros
  if (length(N) != 1 || length(K) != 1 || length(max_time) != 1 || length(verbose) != 1) {
    error("Error: Scalar parameters expected for N, K, max_time, verbose");
    return R_NilValue;
  }

  if (length(x0) != length(probs)) {
    error("Error: x0 and probs must have the same length");
    return R_NilValue;
  }

  // Convertir parámetros
  int64_t n = (int64_t) REAL(N)[0];
  int64_t k = (int64_t) REAL(K)[0];
  real_t max_time_val = REAL(max_time)[0];
  int64_t verbose_val = (int64_t) REAL(verbose)[0];

  // Validaciones adicionales
  if (n <= 0 || k <= 0) {
    error("Error: N and K must be positive");
    return R_NilValue;
  }

  if (length(x0) != k) {
    error("Error: Length of x0 must match K");
    return R_NilValue;
  }

  // Convertir x0
  int64_t* x0_ptr = (int64_t*) malloc(k * sizeof(int64_t));
  if (x0_ptr == NULL) {
    error("Error: Failed to allocate memory for x0.");
    return R_NilValue;
  }

  for (int i = 0; i < k; i++) {
    x0_ptr[i] = (int64_t) REAL(x0)[i];
    if (x0_ptr[i] < 0) {
      free(x0_ptr);
      error("Error: All x0 values must be non-negative");
      return R_NilValue;
    }
  }

  // Convertir probs
  double* probs_ptr = (double*) malloc(k * sizeof(double));
  if (probs_ptr == NULL) {
    free(x0_ptr);
    error("Error: Failed to allocate memory for probs.");
    return R_NilValue;
  }

  double prob_sum = 0.0;
  for (int i = 0; i < k; i++) {
    probs_ptr[i] = REAL(probs)[i];
    if (probs_ptr[i] < 0.0 || probs_ptr[i] > 1.0) {
      free(x0_ptr);
      free(probs_ptr);
      error("Error: All probs values must be between 0 and 1");
      return R_NilValue;
    }
    prob_sum += probs_ptr[i];
  }

  // Normalizar probabilidades si es necesario
  if (fabs(prob_sum - 1.0) > 1e-8) {
    if (verbose_val) {
      Rprintf("Note: Probabilities sum to %f, normalizing to 1\n", prob_sum);
    }
    for (int i = 0; i < k; i++) {
      probs_ptr[i] /= prob_sum;
    }
  }

  // Inicializar lgac
  init_ext(n + k, verbose_val);

  // Variables de salida
  double p_value;
  int64_t converged;
  int64_t terms;
  double p0;

  // Configurar opciones con valores por defecto para los parámetros faltantes
  multinomial_result* mult_res = malloc(sizeof(multinomial_result));
  if (mult_res == NULL) {
    free(x0_ptr);
    free(probs_ptr);
    error("Error: Failed to allocate memory for results");
    finish();
    return R_NilValue;
  }

  // Usar valores por defecto para rel_eps y undersampling
  options_t options = {0};
  set_options(&options,
              DEFAULT_REL_EPS,    // rel_eps por defecto (1e-2)
              INT64_MAX,          // max_terms (usamos INT64_MAX como valor grande)
              1.0,               // undersampling por defecto
              DEFAULT_REL_EPS,
              1e308,             // enum_cutoff muy grande para forzar exhaustive
              8, 1, verbose_val, 10);

  // Llamar a la función principal
  main_int(n, k, x0_ptr, probs_ptr, mult_res, &options);

  p_value = mult_res->pval;
  converged = mult_res->converged;
  terms = mult_res->terms;
  p0 = mult_res->p0;

  // Liberar memoria
  free(mult_res);
  free(x0_ptr);
  free(probs_ptr);
  finish();

  // Crear lista de resultados para R
  SEXP result = PROTECT(allocVector(VECSXP, 4));
  SET_VECTOR_ELT(result, 0, ScalarReal(p_value));
  SET_VECTOR_ELT(result, 1, ScalarInteger(converged));
  SET_VECTOR_ELT(result, 2, ScalarInteger(terms));
  SET_VECTOR_ELT(result, 3, ScalarReal(p0));

  UNPROTECT(1);
  return result;
}

// R interface for the series p-value computation
extern SEXP pval_series(SEXP N, SEXP K, SEXP x0, SEXP probs, SEXP max_terms,
                        SEXP rel_eps, SEXP undersampling, SEXP verbose) {
  // Convert parameters to int64_t and double
  int64_t n = (int64_t) REAL(N)[0];
  int64_t k = (int64_t) REAL(K)[0];
  real_t rel_eps_val = REAL(rel_eps)[0];
  int64_t max_terms_val = (int64_t) REAL(max_terms)[0];
  real_t undersampling_val = REAL(undersampling)[0];
  int64_t verbose_val = (int64_t) REAL(verbose)[0];

  // Convert x0 to an int64_t array
  int64_t* x0_ptr = (int64_t*) malloc(k * sizeof(int64_t));
  if (x0_ptr == NULL) {
    error("Error: Failed to allocate memory for x0.");
    return R_NilValue;
  }
  for (int i = 0; i < k; i++) {
    x0_ptr[i] = (int64_t) REAL(x0)[i];
  }

  // Convert probs to a double array
  double* probs_ptr = (double*) malloc(k * sizeof(double));
  if (probs_ptr == NULL) {
    free(x0_ptr);
    error("Error: Failed to allocate memory for probs.");
    return R_NilValue;
  }
  for (int i = 0; i < k; i++) {
    probs_ptr[i] = REAL(probs)[i];
  }

  // Initialize lgac
  init_ext(n + k, 1);

  // Output variables
  double p_value;
  int64_t converged;
  int64_t terms;
  double p0;

  // Call the C function that uses the series method
  main_series(n, k, x0_ptr, probs_ptr, &p_value, &converged, &terms, &p0,
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

dyn.load("src/pvalues.so")
library(bit64)

#' Compute p-value for a Multinomial Test
#'
#' This function calculates the p-value based on the multinomial distribution
#' using a C function for efficient computation. The p-value is determined by
#' comparing observed counts (`x`) against expected probabilities (`p`)
#' for a multinomial test.
#'
#' @usage
#' pval_flexible(
#'   x,
#'   p,
#'   stat = "prob",
#'   lambda = 0,
#'   max_time = 600,
#'   max_terms = 300,
#'   rel_eps = 0.001,
#'   undersampling = 1,
#'   verbose = FALSE
#' )
#'
#' @param x An Integer vector with realizations for each category.
#' @param p A Numeric vector with the probabilities for each category. These should be non-negative and sum to one. It should be the same size as `x`.
#' @param stat String with the name of the statistic to compute. If `"prob"`, the exact Multinomial p-value is computed. If `"pearson"`, the Pearson's Chi-square p-value is computed. If `"llr"`, the log-likelihood ratio p-value is computed. If `"power_div"`, a Power Divergence p-value is computed, in which case a `lambda` parameter must be given. The default value is `"prob"`.
#' @param lambda A Numeric with the lambda value of the Power Divergence statistic. Only works if `stat = "power_div"`, otherwise is ignored.
#' @param max_time A Numeric with the maximum time limit in seconds. The default is 600.
#' @param max_terms An Integer indicating the number of terms to add in the Fourier series. The default is 300.
#' @param rel_eps A Numeric with the relative error tolerance. The default is 0.001.
#' @param undersampling An Integer with the undersampling value to use. The default and recommended value is 1. Values greater than one will speed up calculations but will sacrifice precision.
#' @param verbose Boolean. If `TRUE`, it prints intermediate results every 10 terms. If `FALSE`, it does not print intermediate computations. The default is `FALSE`.
#'
#' @return Returns a `MultF` object with the following attributes:
#' \itemize{
#'   \item `x`: The input vector of the observed realizations for each category.
#'   \item `p`: The input vector of the probabilities for each category.
#'   \item `pval`: The p-value computed.
#'   \item `gamma`: The optimal gamma obtained in the first part of the method.
#'   \item `n_terms`: The number of terms of the Fourier sum.
#'   \item `time`: The total execution time of the algorithm in seconds.
#'   \item `p0`: Probability mass function in `x`.
#'   \item `status`: The final status ID of the algorithm upon completion:
#'     \itemize{
#'       \item `0`: Converged.
#'       \item `1`: Maximum time reached.
#'       \item `2`: Maximum number of terms reached.
#'       \item `3`: Could not solve the optimization of gamma.
#'     }
#'   \item `message`: The finishing status displayed as a message.
#'   \item `method`: A String with value `"fourier"` or `"exhaustive"`, depending on the method used.
#' }
#'
#' @export
#' @useDynLib MultFourier, .registration = TRUE
#'
#' @examples
#' # Example 1: Compute p-value using the exact multinomial statistic
#' probs <- c(0.00040161, 0.00080321, 0.00200803, 0.00401606, 0.00682731,
#'            0.01044177, 0.01485944, 0.02008032, 0.02610442, 0.03293173,
#'            0.04056225, 0.04899598, 0.05823293, 0.06827309, 0.07911647,
#'            0.09076305, 0.10321285, 0.11646586, 0.13052209, 0.14538153)
#'
#' x0 <- rep(10, length(probs))
#'
#' result <- pval_flexible(x0, probs, max_terms = 300, verbose = TRUE)
#' print(result)
#'
#' # Example 2: Error case (mismatched lengths)
#' ## Not run:
#' x0_invalid <- c(10, 10, 10)
#' probs_invalid <- c(0.2, 0.3)  # Different lengths, should raise an error
#' result_invalid <- pval_flexible(x0_invalid, probs_invalid)
#' print(result_invalid)
#' ## End(Not run)
pval_flexible <- function(x, p, stat = "prob", lambda = 0, max_time = 600,
                          max_terms = 300, rel_eps = 0.001, undersampling = 1,
                          verbose = FALSE) {
  if (length(x) != length(p)) {
    stop("The lengths of 'x' and 'p' must be equal.")
  }

  N <- sum(x)
  K <- length(x)
  verbose_num <- 1
  if (verbose == FALSE){
    verbose_num <- 0
  }

  .Call("pval_flexible", as.double(N), as.double(K), as.double(x),
        as.double(p), as.double(max_terms), as.double(rel_eps),
        as.double(undersampling), as.double(verbose_num))
}

#' Compute p-value using the Exhaustive Method
#'
#' This function calculates the p-value for a multinomial test using the exhaustive method,
#' which evaluates all possible outcomes to compute the exact p-value. This method is
#' computationally intensive but provides precise results for small datasets.
#'
#' @usage
#' pval_exhaustive(
#'   x,
#'   p,
#'   stat = "prob",
#'   lambda = 0,
#'   max_time = 600,
#'   verbose = FALSE
#' )
#'
#' @param x An Integer vector with realizations for each category.
#' @param p A Numeric vector with the probabilities for each category. These should be non-negative and sum to one. It should be the same size as `x`.
#' @param stat String with the name of the statistic to compute. If `"prob"`, the exact Multinomial p-value is computed. If `"pearson"`, the Pearson's Chi-square p-value is computed. If `"llr"`, the log-likelihood ratio p-value is computed. If `"power_div"`, a Power Divergence p-value is computed, in which case a `lambda` parameter must be given. The default value is `"prob"`.
#' @param lambda A Numeric with the lambda value of the Power Divergence statistic. Only works if `stat = "power_div"`, otherwise is ignored.
#' @param max_time A Numeric with the maximum time limit in seconds. The default is 600.
#' @param verbose Boolean. If `TRUE`, it prints information on the run time. If `FALSE`, it does not print. The default is `FALSE`.
#'
#' @return Returns a `MultF` object with the following attributes:
#' \itemize{
#'   \item `x`: The input vector of the observed realizations for each category.
#'   \item `p`: The input vector of the probabilities for each category.
#'   \item `pval`: The p-value computed.
#'   \item `time`: The total execution time of the algorithm in seconds.
#'   \item `p0`: Probability mass function in `x`.
#'   \item `status`: The final status ID of the algorithm upon completion:
#'     \itemize{
#'       \item `0`: Successful computation.
#'       \item `1`: Maximum time reached.
#'     }
#'   \item `message`: The finishing status displayed as a message.
#'   \item `method`: A String with value `"exhaustive"`.
#' }
#'
#' @export
#'
#' @examples
#' # Example 1: Compute p-value using the exhaustive method
#' probs <- c(0.00040161, 0.00080321, 0.00200803, 0.00401606, 0.00682731,
#'            0.01044177, 0.01485944, 0.02008032, 0.02610442, 0.03293173,
#'            0.04056225, 0.04899598, 0.05823293, 0.06827309, 0.07911647,
#'            0.09076305, 0.10321285, 0.11646586, 0.13052209, 0.14538153)
#'
#' x0 <- rep(10, length(probs))
#'
#' result <- pval_exhaustive(x0, probs, verbose = TRUE)
#' print(result)
#'
#' # Example 2: Error case (mismatched lengths)
#' ## Not run:
#' x0_invalid <- c(10, 10, 10)
#' probs_invalid <- c(0.2, 0.3)  # Different lengths, should raise an error
#' result_invalid <- pval_exhaustive(x0_invalid, probs_invalid)
#' print(result_invalid)
#' ## End(Not run)
pval_exhaustive <- function(x, p, stat = "prob", lambda = 0, max_time = 600,
                            verbose = FALSE) {
  if (length(x) != length(p)) {
    stop("The lengths of 'x' and 'p' must be equal.")
  }

  N <- sum(x)
  K <- length(x)
  verbose_num <- 1
  if (verbose == FALSE){
    verbose_num <- 0
  }

  .Call("pval_exhaustive", as.double(N), as.double(K), as.double(x),
        as.double(p), as.double(max_time), as.double(verbose_num))
}

#' Compute p-value using the Fourier Method
#'
#' This function calculates the p-value for a multinomial test using the Fourier series method,
#' which approximates the p-value using a Fourier series expansion. This method is
#' more efficient for large datasets but may introduce some approximation error.
#'
#' @usage
#' pval_fourier(
#'   x,
#'   p,
#'   stat = "prob",
#'   lambda = 0,
#'   max_time = 600,
#'   max_terms = 300,
#'   rel_eps = 0.001,
#'   undersampling = 1,
#'   verbose = FALSE
#' )
#'
#' @param x An Integer vector with realizations for each category.
#' @param p A Numeric vector with the probabilities for each category. These should be non-negative and sum to one. It should be the same size as `x`.
#' @param stat String with the name of the statistic to compute. If `"prob"`, the exact Multinomial p-value is computed. If `"pearson"`, the Pearson's Chi-square p-value is computed. If `"llr"`, the log-likelihood ratio p-value is computed. If `"power_div"`, a Power Divergence p-value is computed, in which case a `lambda` parameter must be given. The default value is `"prob"`.
#' @param lambda A Numeric with the lambda value of the Power Divergence statistic. Only works if `stat = "power_div"`, otherwise is ignored.
#' @param max_time A Numeric with the maximum time limit in seconds. The default is 600.
#' @param max_terms An Integer indicating the number of terms to add in the Fourier series. The default is 300.
#' @param rel_eps A Numeric with the relative error tolerance. The default is 0.001.
#' @param undersampling An Integer with the undersampling value to use. The default and recommended value is 1. Values greater than one will speed up calculations but will sacrifice precision.
#' @param verbose Boolean. If `TRUE`, it prints intermediate results every 10 terms. If `FALSE`, it does not print intermediate computations. The default is `FALSE`.
#'
#' @return Returns a `MultF` object with the following attributes:
#' \itemize{
#'   \item `x`: The input vector of the observed realizations for each category.
#'   \item `p`: The input vector of the probabilities for each category.
#'   \item `pval`: The p-value computed.
#'   \item `gamma`: The optimal gamma obtained in the first part of the method.
#'   \item `n_terms`: The number of terms of the Fourier sum.
#'   \item `time`: The total execution time of the algorithm in seconds.
#'   \item `p0`: Probability mass function in `x`.
#'   \item `status`: The final status ID of the algorithm upon completion:
#'     \itemize{
#'       \item `0`: Converged.
#'       \item `1`: Maximum time reached.
#'       \item `2`: Maximum number of terms reached.
#'       \item `3`: Could not solve the optimization of gamma.
#'     }
#'   \item `message`: The finishing status displayed as a message.
#'   \item `method`: A String with value `"fourier"`.
#' }
#'
#' @export
#'
#' @examples
#' # Example 1: Compute p-value using the Fourier method
#' probs <- c(0.00040161, 0.00080321, 0.00200803, 0.00401606, 0.00682731,
#'            0.01044177, 0.01485944, 0.02008032, 0.02610442, 0.03293173,
#'            0.04056225, 0.04899598, 0.05823293, 0.06827309, 0.07911647,
#'            0.09076305, 0.10321285, 0.11646586, 0.13052209, 0.14538153)
#'
#' x0 <- rep(10, length(probs))
#'
#' result <- pval_fourier(x0, probs, max_terms = 300, verbose = TRUE)
#' print(result)
#'
#' # Example 2: Error case (mismatched lengths)
#' ## Not run:
#' x0_invalid <- c(10, 10, 10)
#' probs_invalid <- c(0.2, 0.3)  # Different lengths, should raise an error
#' result_invalid <- pval_fourier(x0_invalid, probs_invalid)
#' print(result_invalid)
#' ## End(Not run)
pval_fourier <- function(x, p, stat = "prob", lambda = 0, max_time = 600,
                         max_terms = 300, rel_eps = 0.001, undersampling = 1,
                         verbose = FALSE) {
  if (length(x) != length(p)) {
    stop("The lengths of 'x' and 'p' must be equal.")
  }

  N <- sum(x)
  K <- length(x)
  verbose_num <- 1
  if (verbose == FALSE){
    verbose_num <- 0
  }

  .Call("pval_series", as.double(N), as.double(K), as.double(x),
        as.double(p), as.double(max_terms), as.double(rel_eps),
        as.double(undersampling), as.double(verbose_num))
}

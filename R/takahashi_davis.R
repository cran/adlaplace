#' Symbolic unit-lower LDL pattern for a sparse positive-definite matrix
#'
#' Inflates the diagonal so \code{Matrix::Cholesky} can read the sparsity
#' pattern. Numeric values of \code{Q} are not the values that are factored
#' by \code{\link{takahashi_davis}}.
#' @keywords internal
#' @noRd
chol_pattern_ldl <- function(Q) {
  Q <- methods::as(methods::as(Q, "generalMatrix"), "CsparseMatrix")
  if (nrow(Q) != ncol(Q)) {
    stop("Q must be square", call. = FALSE)
  }
  Q <- Matrix::forceSymmetric(Q)
  d <- Matrix::diag(Q)
  if (any(d <= 0) || any(!is.finite(d))) {
    Matrix::diag(Q) <- pmax(abs(d), 1) + 1
  } else {
    Matrix::diag(Q) <- d + 1
  }
  ch <- Matrix::Cholesky(Q, perm = TRUE, LDL = TRUE)
  list(
    perm = as.integer(ch@perm),
    L1 = Matrix::expand2(ch)$L1
  )
}

#' Sparse inverse subset of a symmetric positive-definite matrix
#'
#' Entries of \eqn{Q^{-1}} on the sparsity pattern of the Cholesky factor, in
#' the original variable order. Positions the factor does not store are absent,
#' not zeros of the inverse.
#'
#' C++ callers `LinkingTo: adlaplace` and include
#' `adlaplace/takahashi_impl.hpp`. That function is
#' `adlaplace::chol::takahashi_davis` and takes the unit-lower factor \eqn{L}
#' and the LDL diagonal \eqn{D}. A caller that has only \eqn{Q} factors it
#' with `chol_update_csc` first.
#'
#' @param Q Symmetric sparse positive-definite matrix.
#' @param chol Optional list with `perm` (0-based) and `L1` (unit-lower
#'   factor). When `NULL`, the symbolic factor is computed from `Q`.
#' @return A `dgCMatrix` of the selected inverse.
#' @export
takahashi_davis <- function(Q, chol = NULL) {
  Q <- methods::as(Q, "CsparseMatrix")
  if (nrow(Q) != ncol(Q)) {
    stop("takahashi_davis: Q must be square", call. = FALSE)
  }
  Q <- Matrix::forceSymmetric(Q)
  if (is.null(chol)) {
    chol <- chol_pattern_ldl(Q)
  }
  if (is.null(chol$perm) || is.null(chol$L1)) {
    stop("chol must be a list with perm and L1", call. = FALSE)
  }
  U <- Matrix::triu(Q)
  U <- methods::as(methods::as(U, "generalMatrix"), "CsparseMatrix")
  takahashi_davis_matrix(U, chol$perm, chol$L1)
}

symbolic_chol <- function(Q) {
  Q <- methods::as(methods::as(Q, "generalMatrix"), "CsparseMatrix")
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

test_that("takahashi_davis matches a dense solve on the factor pattern", {
  n <- 6L
  Q <- Matrix::bandSparse(
    n,
    k = c(-1L, 0L, 1L),
    diagonals = list(rep(-0.4, n - 1L), rep(2, n), rep(-0.4, n - 1L))
  )
  Q <- methods::as(Q, "dgCMatrix")
  dense <- solve(as.matrix(Q))

  S <- takahashi_davis(Q)
  expect_s4_class(S, "dgCMatrix")
  expect_equal(dim(S), c(n, n))
  sm <- Matrix::summary(S)
  got <- mapply(function(i, j) S[i, j], sm$i, sm$j)
  ref <- mapply(function(i, j) dense[i, j], sm$i, sm$j)
  expect_equal(as.numeric(got), as.numeric(ref), tolerance = 1e-8)
  expect_equal(Matrix::diag(S), diag(dense), tolerance = 1e-8)

  S2 <- takahashi_davis(Q, chol = symbolic_chol(Q))
  expect_equal(as.matrix(S2), as.matrix(S), tolerance = 1e-12)
})

test_that("takahashi_davis matches a dense solve when the factor has fill-in", {
  set.seed(2)
  n <- 10L
  M <- Matrix::rsparsematrix(n, n, density = 0.35)
  Q <- Matrix::forceSymmetric(M %*% Matrix::t(M)) + Matrix::Diagonal(n, 0.2)
  Q <- methods::as(methods::as(Q, "generalMatrix"), "CsparseMatrix")
  dense <- solve(as.matrix(Q))
  S <- takahashi_davis(Q)
  max_err <- 0
  n_stored <- 0L
  for (j in seq_len(n)) {
    start <- S@p[j]
    end <- S@p[j + 1L]
    if (start >= end) {
      next
    }
    idx <- (start + 1L):end
    rows <- S@i[idx] + 1L
    max_err <- max(max_err, abs(S@x[idx] - dense[rows, j]))
    n_stored <- n_stored + length(idx)
  }
  expect_gt(n_stored, n)
  expect_lt(max_err, 1e-7)
})

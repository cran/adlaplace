test_that("symbolic inner Cholesky succeeds on a high-degree star pattern", {
  # Center 0 connected to 1..20; previously dummy diag=10 was not PD.
  n <- 21L
  rows <- c(0L, seq_len(n - 1L))
  cols <- c(0L, rep(0L, n - 1L))
  # include all diagonals
  rows <- c(rows, seq.int(0L, n - 1L))
  cols <- c(cols, seq.int(0L, n - 1L))
  innerMat <- data.frame(
    rowInner = rows,
    colInner = cols
  )
  res <- adlaplace:::hessian_map_build_chol_inner(innerMat, n)
  expect_true(length(res$chol_inner_list) > 0L)
  expect_true(!is.null(res$chol_inner_list$L1))
  expect_equal(nrow(res$chol_inner_list$L1), n)
})

test_that("symbolic inner Cholesky accepts both-triangle AD sparsity cells", {
  # get_sparse_pattern records (i,j) and (j,i) as separate cells; that used to
  # make sparseMatrix(..., symmetric = TRUE) fail validation on Matrix >= 1.6.
  n <- 21L
  rows <- c(rep(0L, n - 1L), seq_len(n - 1L), seq.int(0L, n - 1L))
  cols <- c(seq_len(n - 1L), rep(0L, n - 1L), seq.int(0L, n - 1L))
  innerMat <- data.frame(rowInner = rows, colInner = cols)
  res <- adlaplace:::hessian_map_build_chol_inner(innerMat, n)
  expect_true(length(res$chol_inner_list) > 0L)
  expect_equal(nrow(res$chol_inner_list$L1), n)
})

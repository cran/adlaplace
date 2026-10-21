expected_diagonal_outer_pattern <- function(nr, theta_index = nr) {
  pairs <- cbind(
    row = c(seq_len(nr) - 1L, seq_len(nr) - 1L, theta_index),
    col = c(seq_len(nr) - 1L, rep(theta_index, nr), theta_index)
  )
  pairs[order(pairs[, 1], pairs[, 2]), , drop = FALSE]
}

expected_diagonal_inner_pattern <- function(nr) {
  pairs <- cbind(row = seq_len(nr) - 1L, col = seq_len(nr) - 1L)
  pairs[order(pairs[, 1], pairs[, 2]), , drop = FALSE]
}

test_that("random_diagonal analytic sparsity is diagonal", {
  nr <- 5L
  model <- adlaplace:::density_data(
    gamma_map = Matrix::sparseMatrix(
      i = seq_len(nr), j = seq_len(nr), x = 1, dims = c(nr, nr)
    ),
    theta_map = c(1L, 1L),
    ad_kind = "random",
    density = "random_diagonal",
    precision = rep(2, nr)
  )
  config <- list(
    beta = numeric(0),
    gamma = rep(0, nr),
    theta = log(0.2),
    transform_theta = TRUE
  )
  ptr <- adlaplace::ad_pack_ptr(model, config)
  pat <- adlaplace:::get_sparse_pattern(ptr, 0L)

  exp_outer <- expected_diagonal_outer_pattern(nr)
  exp_inner <- expected_diagonal_inner_pattern(nr)
  act_outer <- cbind(row = pat$row_hess, col = pat$col_hess)
  act_inner <- cbind(row = pat$row_hess_inner, col = pat$col_hess_inner)

  expect_equal(
    unname(act_outer[order(act_outer[, 1], act_outer[, 2]), , drop = FALSE]),
    unname(exp_outer[order(exp_outer[, 1], exp_outer[, 2]), , drop = FALSE])
  )
  expect_equal(
    unname(act_inner[order(act_inner[, 1], act_inner[, 2]), , drop = FALSE]),
    unname(exp_inner[order(exp_inner[, 1], exp_inner[, 2]), , drop = FALSE])
  )
})

test_that("random_mult inner sparsity follows Q upper triangle (not CppAD conservative)", {
  nr <- 5L
  Q <- adlaplace:::as_dsC_upper(Matrix::bandSparse(
    nr,
    k = 0:1,
    diagonals = list(rep(2, nr), rep(-0.8, nr - 1L)),
    symmetric = TRUE
  ))
  model <- adlaplace:::density_data(
    gamma_map = Matrix::sparseMatrix(
      i = seq_len(nr), j = seq_len(nr), x = 1, dims = c(nr, nr)
    ),
    theta_map = c(1L, 1L),
    ad_kind = "random",
    density = "random_mult",
    precision = list(Q = Q, log_det = 0, rank = nr)
  )
  config <- list(gamma = rep(0, nr), theta = log(0.7), transform_theta = TRUE)
  ptr <- adlaplace::ad_pack_ptr(model, config)
  pat <- adlaplace:::get_sparse_pattern(ptr, 0L)

  Qc <- Matrix::summary(Q)
  Q_upper <- as.matrix(Qc[, c("i", "j")]) - 1L
  act_inner <- cbind(row = pat$row_hess_inner, col = pat$col_hess_inner)
  act_inner <- act_inner[order(act_inner[, 1], act_inner[, 2]), , drop = FALSE]
  Q_upper <- Q_upper[order(Q_upper[, 1], Q_upper[, 2]), , drop = FALSE]

  expect_equal(unname(act_inner), unname(Q_upper))
  expect_lt(nrow(act_inner), nr * (nr + 1L) / 2L)
})

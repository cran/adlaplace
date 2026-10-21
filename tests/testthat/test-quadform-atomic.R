test_that("random_mult inner Hessian matches -tau * Q", {
  set.seed(11)
  nr <- 5L
  Q <- Matrix::bandSparse(
    nr,
    k = 0:1,
    diagonals = list(rep(2, nr), rep(-0.8, nr - 1L)),
    symmetric = TRUE
  )
  Q <- adlaplace:::as_dsC_upper(Q)
  log_det <- as.numeric(Matrix::determinant(Q, logarithm = TRUE)$modulus)

  model <- adlaplace:::density_data(
    gamma_map = Matrix::sparseMatrix(
      i = seq_len(nr), j = seq_len(nr), x = 1, dims = c(nr, nr)
    ),
    theta_map = c(1L, 1L),
    ad_kind = "random",
    density = "random_mult",
    precision = list(Q = Q, log_det = log_det, rank = nr)
  )
  log_sd <- log(0.7)
  config <- list(gamma = rep(0, nr), theta = log_sd, transform_theta = TRUE)
  ptr <- adlaplace::ad_pack_ptr(model, config)

  u <- rnorm(nr)
  x <- c(u, log_sd)
  tau <- exp(-2 * log_sd)

  H_inner <- as.matrix(adlaplace::hessian(ptr, x, inner = TRUE, negative = FALSE))
  expect_equal(
    H_inner[seq_len(nr), seq_len(nr)],
    as.matrix(-tau * Q),
    tolerance = 1e-10
  )

  # Outer mixed gamma-theta vs finite differences
  H_outer <- as.matrix(adlaplace::hessian(ptr, x, inner = FALSE, negative = FALSE))
  eps <- 1e-5
  f <- function(xx) adlaplace::joint_log_dens(ptr, xx, negative = FALSE)
  H_fd <- matrix(0, length(x), length(x))
  for (i in seq_along(x)) {
    for (j in seq_along(x)) {
      xpp <- x
      xpm <- x
      xmp <- x
      xmm <- x
      xpp[i] <- xpp[i] + eps
      xpp[j] <- xpp[j] + eps
      xpm[i] <- xpm[i] + eps
      xpm[j] <- xpm[j] - eps
      xmp[i] <- xmp[i] - eps
      xmp[j] <- xmp[j] + eps
      xmm[i] <- xmm[i] - eps
      xmm[j] <- xmm[j] - eps
      H_fd[i, j] <- (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps * eps)
    }
  }
  expect_equal(H_outer, H_fd, tolerance = 5e-4)
})

test_that("random_mult tape size is O(n), not O(nnz)", {
  make_ptr <- function(nr) {
    Q <- adlaplace:::as_dsC_upper(Matrix::bandSparse(
      nr,
      k = 0:1,
      diagonals = list(rep(2, nr), rep(-0.5, nr - 1L)),
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
    config <- list(gamma = rep(0, nr), theta = 0, transform_theta = TRUE)
    adlaplace::ad_pack_ptr(model, config)
  }

  sz_small <- adlaplace:::get_tape_sizes(make_ptr(10L), 0L)
  sz_large <- adlaplace:::get_tape_sizes(make_ptr(80L), 0L)
  expect_lt(sz_large$size_op / 80, 5)
  Q80 <- adlaplace:::as_dsC_upper(Matrix::bandSparse(
    80L,
    k = 0:1,
    diagonals = list(rep(2, 80), rep(-0.5, 79)),
    symmetric = TRUE
  ))
  expect_lt(sz_large$size_op, Matrix::nnzero(Q80) * 2L)
  expect_lt(sz_large$size_op - sz_small$size_op, 200L)
})

test_that("BYM-style germany random_mult + random_diagonal evaluate", {
  data("germany", package = "adlaplace", envir = environment())
  n <- length(germany$Y)
  n_theta <- 2L
  n_gamma <- 2L * n

  model_struct <- adlaplace:::density_data(
    gamma_map = Matrix::sparseMatrix(
      i = seq_len(n), j = seq_len(n), x = 1,
      dims = c(n_gamma, n)
    ),
    theta_map = c(1L, n_theta),
    ad_kind = "random",
    density = "random_mult",
    precision = germany$prec
  )
  expect_true(inherits(model_struct@precision$Q, "dsCMatrix"))

  model_iid <- adlaplace:::density_data(
    gamma_map = Matrix::sparseMatrix(
      i = seq.int(n + 1L, length.out = n), j = seq_len(n), x = 1,
      dims = c(n_gamma, n)
    ),
    theta_map = c(2L, n_theta),
    ad_kind = "random",
    density = "random_diagonal",
    precision = rep(1, n)
  )
  config <- list(
    transform_theta = TRUE,
    gamma = rep(0, n_gamma),
    theta = log(c(1e-2, 0.1)),
    verbose = FALSE
  )

  ptr_struct <- adlaplace::ad_pack_ptr(model_struct, config)
  ptr_iid <- adlaplace::ad_pack_ptr(model_iid, config)

  sz <- adlaplace:::get_tape_sizes(ptr_struct, 0L)
  expect_lt(sz$size_op, Matrix::nnzero(model_struct@precision$Q) * 2L)

  ptr <- c(ptr_struct, ptr_iid)

  x <- c(rep(0, n_gamma), config$theta)
  ld <- adlaplace::joint_log_dens(ptr, x, negative = FALSE)
  expect_true(is.finite(ld))

  g <- as.numeric(adlaplace::grad(ptr, x, inner = FALSE, negative = FALSE))
  expect_equal(length(g), length(x))
  expect_true(all(is.finite(g)))
})

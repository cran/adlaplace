make_upper_Q <- function(nr, off = -0.8) {
  Q <- Matrix::bandSparse(
    nr,
    k = 0:1,
    diagonals = list(rep(2, nr), rep(off, nr - 1L)),
    symmetric = TRUE
  )
  adlaplace:::as_dsC_upper(Q)
}

test_that("random_mult matches the dense GMRF log density", {
  set.seed(42)
  nr <- 5L
  Q <- make_upper_Q(nr)
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
  expect_true(inherits(model@precision$Q, "dsCMatrix"))
  expect_identical(model@precision$Q@uplo, "U")

  log_sd <- log(0.7)
  config <- list(
    gamma = rep(0, nr),
    theta = log_sd,
    transform_theta = TRUE
  )
  ptr <- adlaplace::ad_pack_ptr(model, config)

  u <- rnorm(nr)
  x <- c(u, log_sd)
  tau <- exp(-2 * log_sd)
  manual <- 0.5 * (nr * log(tau) + log_det) -
    0.5 * tau * as.numeric(crossprod(u, Q %*% u)) -
    0.5 * nr * log(2 * pi)

  expect_equal(
    adlaplace::joint_log_dens(ptr, x, negative = FALSE),
    manual,
    tolerance = 1e-10
  )

  g <- as.numeric(adlaplace::grad(ptr, x, inner = FALSE, negative = FALSE))
  eps <- 1e-6
  g_fd <- vapply(
    seq_along(x),
    function(i) {
      xp <- x
      xm <- x
      xp[i] <- xp[i] + eps
      xm[i] <- xm[i] - eps
      (adlaplace::joint_log_dens(ptr, xp, negative = FALSE) -
        adlaplace::joint_log_dens(ptr, xm, negative = FALSE)) / (2 * eps)
    },
    numeric(1)
  )
  expect_equal(g, g_fd, tolerance = 1e-5)
})

test_that("random_mult handles an intrinsic (rank-deficient) structure matrix", {
  nr <- 6L
  # ICAR structure matrix of a path graph: rank nr - 1, null vector = constant
  Q_dense <- diag(c(1, rep(2, nr - 2L), 1))
  for (i in seq_len(nr - 1L)) {
    Q_dense[i, i + 1L] <- Q_dense[i + 1L, i] <- -1
  }
  ev <- eigen(Q_dense, symmetric = TRUE, only.values = TRUE)$values
  log_det_gen <- sum(log(ev[seq_len(nr - 1L)]))
  Q <- adlaplace:::as_dsC_upper(Matrix::Matrix(Q_dense, sparse = TRUE))

  model <- adlaplace:::density_data(
    gamma_map = Matrix::sparseMatrix(
      i = seq_len(nr), j = seq_len(nr), x = 1, dims = c(nr, nr)
    ),
    theta_map = c(1L, 1L),
    ad_kind = "random",
    density = "random_mult",
    precision = list(Q = Q, log_det = log_det_gen, rank = nr - 1L)
  )
  log_sd <- log(1.3)
  config <- list(gamma = rep(0, nr), theta = log_sd, transform_theta = TRUE)
  ptr <- adlaplace::ad_pack_ptr(model, config)

  set.seed(3)
  u <- rnorm(nr)
  x <- c(u, log_sd)
  tau <- exp(-2 * log_sd)
  manual <- 0.5 * ((nr - 1L) * log(tau) + log_det_gen) -
    0.5 * tau * as.numeric(crossprod(u, Q_dense %*% u)) -
    0.5 * (nr - 1L) * log(2 * pi)

  expect_equal(
    adlaplace::joint_log_dens(ptr, x, negative = FALSE),
    manual,
    tolerance = 1e-10
  )

  # invariance to adding a constant (the null direction) in the quadratic form
  x_shift <- c(u + 0.5, log_sd)
  expect_equal(
    adlaplace::joint_log_dens(ptr, x_shift, negative = FALSE),
    manual,
    tolerance = 1e-10
  )
})

test_that("random_mult rejects malformed precision payloads", {
  nr <- 3L
  gamma_map <- Matrix::sparseMatrix(
    i = seq_len(nr), j = seq_len(nr), x = 1, dims = c(nr, nr)
  )
  config <- list(gamma = rep(0, nr), theta = 0, transform_theta = TRUE)

  model_no_q <- adlaplace:::density_data(
    gamma_map = gamma_map,
    theta_map = c(1L, 1L),
    ad_kind = "random",
    density = "random_mult",
    precision = list(log_det = 0, rank = nr)
  )
  expect_error(
    adlaplace::ad_pack_ptr(model_no_q, config),
    "must contain Q"
  )

  model_vec <- adlaplace:::density_data(
    gamma_map = gamma_map,
    theta_map = c(1L, 1L),
    ad_kind = "random",
    density = "random_mult",
    precision = rep(1, nr)
  )
  expect_error(
    adlaplace::ad_pack_ptr(model_vec, config),
    "list"
  )
})

test_that("as_dsC_upper coerces dgCMatrix and C++ rejects raw dgCMatrix", {
  nr <- 4L
  Q_sym <- Matrix::bandSparse(
    nr,
    k = 0:1,
    diagonals = list(rep(2, nr), rep(-0.5, nr - 1L)),
    symmetric = TRUE
  )
  Q_dg <- methods::as(methods::as(Q_sym, "generalMatrix"), "CsparseMatrix")
  expect_true(inherits(Q_dg, "dgCMatrix"))

  Q_ds <- adlaplace:::as_dsC_upper(Q_dg)
  expect_true(inherits(Q_ds, "dsCMatrix"))
  expect_identical(Q_ds@uplo, "U")

  model_ok <- adlaplace:::density_data(
    gamma_map = Matrix::sparseMatrix(
      i = seq_len(nr), j = seq_len(nr), x = 1, dims = c(nr, nr)
    ),
    theta_map = c(1L, 1L),
    ad_kind = "random",
    density = "random_mult",
    precision = list(Q = Q_dg, log_det = 0, rank = nr)
  )
  expect_true(inherits(model_ok@precision$Q, "dsCMatrix"))
  ptr <- adlaplace::ad_pack_ptr(
    model_ok,
    list(gamma = rep(0, nr), theta = 0, transform_theta = TRUE)
  )
  expect_true(is(ptr, "ad_pack_ptr"))

  # Bypass R coerce: stuff a dgCMatrix into an already-built density_data
  model_bad <- model_ok
  model_bad@precision$Q <- Q_dg
  expect_error(
    adlaplace::ad_pack_ptr(
      model_bad,
      list(gamma = rep(0, nr), theta = 0, transform_theta = TRUE)
    ),
    "dsCMatrix"
  )
})

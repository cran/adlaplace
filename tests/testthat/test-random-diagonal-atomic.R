finite_diff_grad <- function(f, x, eps = 1e-6) {
  n <- length(x)
  g <- numeric(n)
  for (i in seq_len(n)) {
    xlo <- x
    xhi <- x
    xlo[i] <- xlo[i] - eps
    xhi[i] <- xhi[i] + eps
    g[i] <- (f(xhi) - f(xlo)) / (2 * eps)
  }
  g
}

finite_diff_hess <- function(f, x, eps = 1e-5) {
  n <- length(x)
  H <- matrix(0, n, n)
  for (i in seq_len(n)) {
    xlo <- x
    xhi <- x
    xlo[i] <- xlo[i] - eps
    xhi[i] <- xhi[i] + eps
    H[i, i] <- (f(xhi) - 2 * f(x) + f(xlo)) / eps^2
  }
  H
}

random_diagonal_fixture <- function(nr = 20L, n_gamma_full = NULL) {
  if (is.null(n_gamma_full)) {
    n_gamma_full <- nr
  }
  model <- adlaplace:::density_data(
    gamma_map = Matrix::sparseMatrix(
      i = seq_len(nr),
      j = seq_len(nr),
      x = 1,
      dims = c(n_gamma_full, nr)
    ),
    theta_map = c(1L, 1L),
    ad_kind = "random",
    density = "random_diagonal",
    precision = rep(2, nr)
  )
  config <- list(
    beta = numeric(0),
    gamma = rep(0, n_gamma_full),
    theta = log(0.2),
    transform_theta = TRUE,
    verbose = FALSE
  )
  list(model = model, config = config)
}

test_that("random_diagonal ad_pack_ptr builds", {
  fx <- random_diagonal_fixture(nr = 8L)
  ptr <- adlaplace::ad_pack_ptr(fx$model, fx$config)
  expect_true(is(ptr, "ad_pack_ptr"))
  af <- adlaplace::ad_pack(ptr, num_threads = 1L)
  expect_true(methods::is(af, "ad_pack"))
})

test_that("random_diagonal grad and inner hess match finite differences", {
  set.seed(42)
  nr <- 20L
  fx <- random_diagonal_fixture(nr = nr)
  af <- adlaplace::ad_pack(
    adlaplace::ad_pack_ptr(fx$model, fx$config),
    num_threads = 1L
  )
  x <- c(rnorm(nr, sd = 0.1), fx$config$theta)
  f <- function(xx) {
    adlaplace::joint_log_dens(af, xx, negative = FALSE)
  }
  gr_ad <- adlaplace::grad(af, x, negative = FALSE)
  gr_fd <- finite_diff_grad(f, x)
  expect_equal(gr_ad, gr_fd, tolerance = 5e-5)

  H_ad <- as.matrix(adlaplace::hessian(af, x, inner = TRUE, negative = FALSE))
  H_fd <- finite_diff_hess(f, x)
  expect_equal(diag(H_ad)[seq_len(nr)], diag(H_fd)[seq_len(nr)], tolerance = 5e-4)

  tau <- exp(-2 * x[length(x)])
  expect_equal(diag(H_ad)[seq_len(nr)], -tau * rep(2, nr), tolerance = 1e-5)
})

test_that("inner_opt works on random_diagonal-only ad_pack", {
  set.seed(7)
  nr <- 25L
  fx <- random_diagonal_fixture(nr = nr)
  af <- adlaplace::ad_pack(
    adlaplace::ad_pack_ptr(fx$model, fx$config),
    num_threads = 1L
  )
  parameters <- fx$config$theta
  gamma <- rnorm(nr, sd = 0.05)

  inner_false <- adlaplace::inner_opt(
    parameters = parameters,
    gamma = gamma,
    ad_pack = af,
    control = list(maxit = 20L, report.level = 0, report.freq = 0),
    deriv = FALSE
  )
  expect_true(is.finite(inner_false$log_lik))
  expect_length(inner_false$gradient$inner, nr)

  inner_true <- adlaplace::inner_opt(
    parameters = parameters,
    gamma = gamma,
    ad_pack = af,
    control = list(maxit = 20L, report.level = 0, report.freq = 0),
    deriv = TRUE
  )
  expect_true(is.finite(inner_true$log_lik))
  expect_true(all(is.finite(inner_true$hessian$trace3)))
})

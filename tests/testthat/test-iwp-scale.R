test_that("iwp basis and penalty are invariant to a units change", {
  knots_m <- seq(0, 2000, by = 200)
  x_m <- seq(0, 2000, length.out = 25)
  scale <- 1000
  term_m <- adlaplace::iwp(
    "elev", p = 2, knots = knots_m, ref_value = 600
  )
  term_k <- adlaplace::iwp(
    "elev", p = 2, knots = knots_m / scale, ref_value = 0.6
  )
  dat_m <- data.frame(elev = x_m)
  dat_k <- data.frame(elev = x_m / scale)

  design_m <- as.matrix(design(term_m[[1]], dat_m))
  design_k <- as.matrix(design(term_k[[1]], dat_k))
  expect_lt(max(abs(design_m)), 1e3)
  expect_equal(design_m, design_k, tolerance = 1e-8)

  prec_m <- as.matrix(precision(term_m[[1]], dat_m))
  prec_k <- as.matrix(precision(term_k[[1]], dat_k))
  expect_equal(prec_m, prec_k, tolerance = 1e-8)
  expect_lt(max(diag(prec_m)), 5)

  poly_m <- as.matrix(design(term_m[[2]], dat_m))
  poly_k <- as.matrix(design(term_k[[2]], dat_k))
  expect_equal(poly_m, poly_k, tolerance = 1e-8)
  expect_lt(max(abs(poly_m)), 20)
})

test_that("fitted iwp curve is invariant to a units change", {
  set.seed(21)
  n <- 36L
  x <- seq(0, 10, length.out = n)
  dat <- data.frame(
    y = sin(pi * x / 10) + rnorm(n, sd = 0.05),
    x = x
  )
  knots <- seq(0, 10, by = 2)
  ctrl <- list(maxit = 40L, trace = 0)
  cfg <- list(num_threads = 1L, num_shards = 4L)
  # An intermediate inner solve can stop on the trust radius before the
  # final evaluation succeeds. The curve comparison uses the final fit.
  fit_m <- suppressWarnings(adlaplace(
    y ~ iwp(x, p = 2, knots = knots, ref_value = 0, init = 0.5),
    data = dat,
    config = cfg,
    control = ctrl,
    hessian = FALSE,
    verbose = FALSE
  ))
  scale <- 10
  knots_s <- knots / scale
  dat_s <- dat
  dat_s$x <- dat$x / scale
  fit_s <- suppressWarnings(adlaplace(
    y ~ iwp(x, p = 2, knots = knots_s, ref_value = 0, init = 0.5),
    data = dat_s,
    config = cfg,
    control = ctrl,
    hessian = FALSE,
    verbose = FALSE
  ))
  expect_identical(fit_m$details$inner_opt$status, "Success")
  expect_identical(fit_s$details$inner_opt$status, "Success")

  linpred <- function(fit) {
    md <- fit$model_data
    n_beta <- nrow(md$term_data$info$beta)
    beta <- fit$details$outer_opt$par[seq_len(n_beta)]
    as.numeric(md$term_data$X %*% beta + md$term_data$A %*% fit$gamma)
  }
  expect_equal(linpred(fit_m), linpred(fit_s), tolerance = 1e-3)
})

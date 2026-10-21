test_that("sim_iid works for all-IID model and matches solve(H_inner)", {
  set.seed(21)
  n <- 120L
  n_re <- 5L
  g <- factor(sample(n_re, n, replace = TRUE))
  re <- rnorm(n_re, sd = 0.25)
  x <- rbinom(n, 1, 0.5)
  eta <- 1 + 0.4 * x + re[g]
  nb_sd <- 0.3
  z <- rgamma(n, nb_sd^(-2), nb_sd^(-2))
  dat <- data.frame(
    y = rpois(n, exp(eta) * z),
    x = x,
    g = g
  )
  fit <- adlaplace(
    nbinom(y, lower = 1e-9, init = 0.15) ~ x + iid(g, init = 0.3),
    data = dat,
    config = list(num_shards = 4L),
    control = list(maxit = 50L),
    verbose = FALSE
  )

  set.seed(22)
  out <- sim_iid(fit, n = 20L, probs = c(0.1, 0.5, 0.9))
  expect_s3_class(out, "sim_iid")
  expect_true("g" %in% names(out))
  expect_equal(length(out$g$x), n_re)
  expect_equal(nrow(out$g$sim), n_re)
  expect_equal(ncol(out$g$sim), 20L)
  expect_equal(nrow(out$g$pointwise), n_re)
  expect_equal(ncol(out$g$pointwise), 3L)
  expect_true(all(is.finite(out$g$sim)))

  var_iid <- attr(out, "var_iid")
  H_inner <- fit$details$hessian$inner
  expect_equal(dim(var_iid), dim(H_inner))
  expect_equal(
    unname(var_iid),
    unname(as.matrix(solve(H_inner))),
    tolerance = 1e-8
  )
})

test_that("sim_iid Woodbury path works with iid + iwp", {
  skip_if_not_installed("WoodburyMatrix")
  set.seed(23)
  n <- 100L
  n_re <- 4L
  g <- factor(sample(n_re, n, replace = TRUE))
  re <- rnorm(n_re, sd = 0.2)
  xx <- seq(0, 1, length.out = n)
  eta <- 0.5 + re[g] + 0.3 * sin(2 * pi * xx)
  dat <- data.frame(
    y = rnorm(n, eta, 0.25),
    x = xx,
    g = g
  )
  fit <- adlaplace(
    y ~ 0 + iwp(x, p = 2, knots = seq(0, 1, by = 0.25), init = 0.2) +
      iid(g, init = 0.3),
    data = dat,
    config = list(num_shards = 4L),
    control = list(maxit = 50L),
    verbose = FALSE
  )

  set.seed(24)
  out <- sim_iid(fit, n = 15L)
  n_iid <- sum(fit$model_data$term_data$info$gamma$model == "iid")
  expect_equal(nrow(attr(out, "var_iid")), n_iid)
  expect_equal(ncol(attr(out, "var_iid")), n_iid)
  expect_equal(length(attr(out, "iid_labels")), n_iid)
  expect_true("g" %in% names(out))
  expect_equal(ncol(out$g$sim), 15L)
  expect_true(all(is.finite(out$g$sim)))
  expect_true(all(is.finite(attr(out, "var_iid"))))
})

test_that("sim_iid returns GET global envelope when available", {
  skip_if_not_installed("GET")
  set.seed(25)
  n <- 80L
  n_re <- 4L
  g <- factor(sample(n_re, n, replace = TRUE))
  dat <- data.frame(
    y = rpois(n, 2),
    x = rbinom(n, 1, 0.5),
    g = g
  )
  fit <- adlaplace(
    nbinom(y, lower = 1e-9, init = 0.15) ~ x + iid(g, init = 0.3),
    data = dat,
    config = list(num_shards = 3L),
    control = list(maxit = 40L),
    verbose = FALSE
  )
  set.seed(26)
  out <- sim_iid(fit, n = 30L, coverage = 0.8)
  expect_true(!is.null(out$g$global))
  expect_true(all(c("lo", "central", "hi") %in% colnames(out$g$global)))
  expect_equal(nrow(out$g$global), length(out$g$x))
})

test_that("sim_iid errors when fit has no IID terms", {
  set.seed(27)
  n <- 60L
  x <- seq(0, 1, length.out = n)
  dat <- data.frame(
    y = rnorm(n, sin(2 * pi * x), 0.2),
    x = x
  )
  fit <- adlaplace(
    y ~ 0 + iwp(x, p = 2, knots = seq(0, 1, by = 0.25), init = 0.2),
    data = dat,
    config = list(num_shards = 3L),
    control = list(maxit = 40L),
    verbose = FALSE
  )
  expect_error(sim_iid(fit), "no IID")
})

test_that("sim_random works for iwp + rpoly terms", {
  set.seed(11)
  n <- 120L
  x <- seq(0, 1, length.out = n)
  dat <- data.frame(
    y = rnorm(n, sin(2 * pi * x), 0.2),
    x = x
  )
  fit <- adlaplace(
    y ~ 0 + iwp(x, p = 2, knots = seq(0, 1, by = 0.2), init = 0.2),
    data = dat,
    config = list(num_shards = 4L),
    control = list(maxit = 60L),
    verbose = FALSE
  )
  set.seed(12)
  out <- sim_random("x", fit = fit, num_sim = 15L, num_grid = 21L)
  expect_true(is.list(out))
  expect_true(is.numeric(out$x))
  expect_equal(nrow(out$sim), length(out$x))
  expect_equal(ncol(out$sim), 15L)
  expect_true(all(is.finite(out$sim)))
  expect_equal(nrow(out$pointwise), length(out$x))
  expect_equal(ncol(out$pointwise), 3L)

  gamma_sims <- rmvnldl(n = 8L, fit = fit)
  out2 <- sim_random("x", fit = fit, gamma_sims = gamma_sims, num_grid = 11L)
  expect_equal(ncol(out2$sim), 8L)
})

test_that("sim_random works for iid terms", {
  set.seed(13)
  n <- 150L
  n_re <- 6L
  g <- factor(sample(n_re, n, replace = TRUE))
  re <- rnorm(n_re, sd = 0.3)
  x <- rbinom(n, 1, 0.5)
  eta <- 1 + 0.5 * x + re[g]
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
    control = list(maxit = 60L),
    verbose = FALSE
  )
  set.seed(14)
  out <- sim_random("g", fit = fit, num_sim = 10L)
  expect_equal(length(out$x), nlevels(dat$g))
  expect_equal(nrow(out$sim), length(out$x))
  expect_equal(ncol(out$sim), 10L)
  expect_true(all(is.finite(out$sim)))
})

test_that("sim_random transform=exp and GET global when available", {
  skip_if_not_installed("GET")
  set.seed(16)
  n <- 80L
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
  set.seed(17)
  out_lp <- sim_random("x", fit = fit, num_sim = 20L, num_grid = 15L)
  set.seed(17)
  out_rr <- sim_random(
    "x",
    fit = fit, num_sim = 20L, num_grid = 15L, transform = exp
  )
  expect_equal(out_rr$sim, exp(out_lp$sim), tolerance = 1e-10)
  expect_true(!is.null(out_rr$global))
  expect_true(all(c("lo", "central", "hi") %in% colnames(out_rr$global)))
  expect_equal(nrow(out_rr$global), length(out_rr$x))
})

test_that("sim_random errors for unknown variable", {
  set.seed(15)
  n <- 80L
  g <- factor(sample(4L, n, replace = TRUE))
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
  expect_error(sim_random("nope", fit = fit), "no random terms")
})

simulate_fit_data <- function(n = 400L, n_re = 8L, seed = 1L) {
  set.seed(seed)
  g <- sample(n_re, n, replace = TRUE)
  re <- rnorm(n_re, sd = 0.3)
  x <- rbinom(n, 1, 0.5)
  eta <- 1 + 0.5 * x + re[g]
  nb_sd <- 0.3
  z <- rgamma(n, nb_sd^(-2), nb_sd^(-2))
  data.frame(
    y = rpois(n, exp(eta) * z),
    x = x,
    g = factor(g)
  )
}

test_that("sim_fit returns expected structure on small GLMM", {
  dat <- simulate_fit_data(n = 200L, seed = 3L)
  fit <- adlaplace(
    nbinom(y, lower = 1e-9, init = 0.15) ~ x + iid(g, init = 0.3),
    data = dat,
    config = list(num_shards = 5L),
    control = list(maxit = 80L),
    verbose = FALSE
  )
  newx <- data.frame(
    x = c(0, 1),
    g = factor(c(1L, 2L), levels = levels(dat$g))
  )
  set.seed(7)
  sims <- sim_fit(newx, data = fit$model_data, fit = fit$details, n = 12L)

  expect_true(inherits(sims, "Matrix"))
  expect_equal(dim(sims), c(2L, 12L))
  expect_true(all(is.finite(sims)))
  expect_true(!is.null(fit$details$hessian$chol_inner))
})

test_that("sim_fit rejects non-model_data input", {
  dat <- simulate_fit_data(n = 80L, seed = 9L)
  fit <- adlaplace(
    nbinom(y, lower = 1e-9, init = 0.15) ~ x + iid(g, init = 0.3),
    data = dat,
    config = list(num_shards = 4L),
    control = list(maxit = 50L),
    verbose = FALSE
  )
  expect_error(
    sim_fit(
      data.frame(x = 1, g = factor(1, levels = levels(dat$g))),
      data = list(foo = 1),
      fit = fit$details
    ),
    "model_data"
  )
})

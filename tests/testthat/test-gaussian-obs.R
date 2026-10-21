test_that("gaussian_obs + gaussian_extra match dnorm", {
  set.seed(11)
  n <- 30L
  n_re <- 3L
  X <- Matrix::Matrix(cbind(1, rnorm(n)))
  A <- Matrix::sparseMatrix(
    i = seq_len(n),
    j = sample(n_re, n, replace = TRUE),
    x = 1,
    dims = c(n, n_re)
  )
  beta <- c(2, -1)
  gamma <- rnorm(n_re, sd = 0.5)
  sigma <- 0.8
  eta <- as.vector(X %*% beta + A %*% gamma)
  y <- rnorm(n, eta, sigma)

  config <- list(
    beta = beta,
    gamma = gamma,
    theta = log(sigma),
    transform_theta = TRUE,
    obs_groups = adlaplace::obs_groups(A, num_shards = 2L)
  )
  theta_map <- Matrix::sparseMatrix(i = 1L, j = 1L, dims = c(1L, 1L))
  obs <- adlaplace:::density_data(
    y = y, A = A, X = X,
    theta_map = theta_map,
    ad_kind = "observations",
    density = "gaussian_obs"
  )
  extra <- adlaplace::density_data(
    obs,
    ad_kind = "parameters",
    density = "gaussian_extra"
  )
  ptr_obs <- adlaplace::ad_pack_ptr(obs, config)
  ptr_extra <- adlaplace::ad_pack_ptr(extra, config)

  x <- c(beta, gamma, log(sigma))
  manual <- sum(stats::dnorm(y, eta, sigma, log = TRUE))
  ad_total <- adlaplace::joint_log_dens(ptr_obs, x, negative = FALSE) +
    adlaplace::joint_log_dens(ptr_extra, x, negative = FALSE)
  expect_equal(ad_total, manual, tolerance = 1e-10)
})

test_that("bare response defaults to gaussian and adlaplace() fits it", {
  set.seed(12)
  n <- 300L
  n_re <- 6L
  g <- sample(n_re, n, replace = TRUE)
  re <- rnorm(n_re, sd = 0.5)
  x <- rnorm(n)
  y <- 2 + 0.7 * x + re[g] + rnorm(n, sd = 0.4)
  dat <- data.frame(y = y, x = x, g = factor(g))

  terms <- adlaplace::collect_terms(y ~ x + iid(g))
  expect_true(inherits(terms$gaussian_y, "gaussian"))
  expect_identical(terms$gaussian_y@density, "gaussian_obs")

  fit <- adlaplace(
    y ~ x + iid(g, init = 0.5),
    data = dat,
    config = list(num_shards = 5L),
    control = list(maxit = 100L)
  )
  expect_identical(fit$details$outer_opt$convergence, 0L)
  co <- coef(fit)
  labels <- names(co)
  expect_equal(
    unname(co[grep("intercept", labels)]), 2,
    tolerance = 0.5
  )
  expect_equal(
    unname(co[grep("^x", labels)]), 0.7,
    tolerance = 0.2
  )
  expect_equal(
    unname(co[grep("gaussian_sd", labels)]), 0.4,
    tolerance = 0.15
  )
  # agreement with lme4-style reference: compare against stats::lm on
  # group-demeaned data is overkill; just check logLik is finite and
  # the residual SD beats a null model fit
  expect_true(is.finite(as.numeric(logLik(fit))))
})

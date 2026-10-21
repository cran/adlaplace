test_that("known-sd intercept/linear/rpoly create random shards without theta", {
  set.seed(11)
  dat <- data.frame(
    y = rbinom(40, 1, 0.6),
    x = runif(40),
    g = factor(rep(1:5, each = 8)),
    trt = factor(rep(c("a", "b", "c"), length.out = 40))
  )

  md <- adlaplace::model_data(
    adlaplace::binom(y) ~
      adlaplace::intercept(sd = 5) +
      adlaplace::linear(trt, sd = 5) +
      adlaplace::rpoly(x, p = 1, sd = 5) +
      adlaplace::iid(g, init = 0.5),
    data = dat
  )

  expect_equal(nrow(md$term_data$info$beta), 0L)
  expect_equal(nrow(md$term_data$info$theta), 1L)
  expect_equal(md$term_data$info$theta$label, "g_iid")
  expect_true("intercept" %in% names(md$random))
  expect_true("trt_linear" %in% names(md$random))
  expect_true("x_rpoly" %in% names(md$random))
  expect_true("g_iid" %in% names(md$random))
  expect_equal(ncol(md$random$intercept@theta_map), 0L)
  expect_equal(ncol(md$random$trt_linear@theta_map), 0L)
  expect_equal(ncol(md$random$x_rpoly@theta_map), 0L)
  expect_equal(ncol(md$random$g_iid@theta_map), 1L)
})

test_that("known-sd random_diagonal evaluates without theta", {
  nr <- 3L
  model <- adlaplace::density_data(
    gamma_map = Matrix::sparseMatrix(
      i = seq_len(nr), j = seq_len(nr), x = 1, dims = c(nr, nr)
    ),
    theta_map = Matrix::Matrix(nrow = 0L, ncol = 0L),
    ad_kind = "random",
    density = "random_diagonal",
    precision = rep(1 / 25, nr)
  )
  config <- list(
    beta = numeric(0),
    gamma = c(0.2, -0.1, 0.3),
    theta = numeric(0),
    transform_theta = TRUE
  )
  ptr <- adlaplace::ad_pack_ptr(model, config)
  x <- config$gamma
  ll <- adlaplace::joint_log_dens(ptr, x, negative = FALSE)
  # N(0, 5^2): sum(dnorm(gamma, 0, 5, log = TRUE))
  expect_equal(
    ll,
    sum(stats::dnorm(x, 0, 5, log = TRUE)),
    tolerance = 1e-10
  )
})

test_that("prior maps 0-based theta id and builds parameters shard", {
  set.seed(12)
  dat <- data.frame(
    y = rbinom(30, 1, 0.5),
    g = factor(rep(1:5, each = 6))
  )
  md <- adlaplace::model_data(
    adlaplace::binom(y) ~ adlaplace::intercept(sd = 10) +
      adlaplace::iid(g, init = 1) +
      adlaplace::prior(theta = 0, dist = "exp", median = 1),
    data = dat
  )
  expect_true("prior_theta0" %in% names(md$parameters))
  shard <- md$parameters$prior_theta0
  expect_identical(shard@density, "exp_prior")
  expect_identical(shard@ad_kind, "parameters")
  expect_equal(as.numeric(shard@y), log(2), tolerance = 1e-12)
  expect_equal(ncol(shard@theta_map), 1L)

  expect_error(
    adlaplace::model_data(
      adlaplace::binom(y) ~ adlaplace::iid(g) +
        adlaplace::prior(theta = 3, dist = "exp", median = 1),
      data = dat
    ),
    "not found"
  )
  expect_error(
    adlaplace::model_data(
      adlaplace::binom(y) ~ adlaplace::iid(g) +
        adlaplace::prior(theta = "nope", dist = "exp", median = 1),
      data = dat
    ),
    "not found"
  )
})

test_that("prior accepts theta label", {
  set.seed(13)
  dat <- data.frame(
    y = rbinom(20, 1, 0.5),
    g = factor(rep(1:4, each = 5))
  )
  md <- adlaplace::model_data(
    adlaplace::binom(y) ~ adlaplace::intercept(sd = 5) +
      adlaplace::iid(g, init = 1) +
      adlaplace::prior(theta = "g_iid", dist = "exp", median = 2),
    data = dat
  )
  expect_true("prior_g_iid" %in% names(md$parameters))
  expect_equal(
    as.numeric(md$parameters$prior_g_iid@y),
    log(2) / 2,
    tolerance = 1e-12
  )
})

test_that("exp_prior matches dexp plus Jacobian", {
  model <- adlaplace::density_data(
    y = log(2),
    theta_map = c(1L, 1L),
    beta_map = 0L,
    gamma_map = 0L,
    ad_kind = "parameters",
    density = "exp_prior"
  )
  u <- log(1.5)
  config <- list(
    beta = numeric(0),
    gamma = numeric(0),
    theta = u,
    transform_theta = TRUE
  )
  ptr <- adlaplace::ad_pack_ptr(model, config)
  ll <- adlaplace::joint_log_dens(ptr, u, negative = FALSE)
  rate <- log(2)
  expect_equal(
    ll,
    stats::dexp(exp(u), rate = rate, log = TRUE) + u,
    tolerance = 1e-10
  )
})

test_that("bacteria Bayesian formula has one outer parameter and aghq runs", {
  skip_if_not_installed("MASS")
  skip_if_not_installed("aghq")
  skip_if_not_installed("numDeriv")

  data(bacteria, package = "MASS")
  bacteria$present <- as.integer(bacteria$y == "y")

  fit <- adlaplace::adlaplace(
    adlaplace::binom(present) ~
      adlaplace::intercept(sd = 10) +
      adlaplace::linear(trt, sd = 10) +
      adlaplace::linear(week, sd = 10) +
      adlaplace::iid(ID, init = 1) +
      adlaplace::prior(theta = 0, dist = "exp", median = 1),
    data = bacteria,
    config = list(num_shards = 20L),
    control = list(maxit = 100L),
    verbose = FALSE
  )
  expect_equal(nrow(fit$par_info), 1L)
  expect_identical(fit$par_info$label, "ID_iid")
  expect_identical(fit$details$outer_opt$convergence, 0L)

  ff <- adlaplace::aghq_ff(fit)
  quad <- aghq::aghq(
    ff = ff,
    k = 3,
    startingvalue = fit$par_info$mle_internal,
    control = aghq::default_control(negate = TRUE)
  )
  expect_true(is.finite(quad$normalized_posterior$lognormconst))
  naw <- quad$normalized_posterior$nodesandweights
  expect_true(all(is.finite(naw$logpost_normalized)))
  expect_true(is.finite(as.numeric(quad$optresults$mode)))
})

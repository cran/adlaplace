test_that("binomial_obs matches dbinom and has correct gradient", {
  set.seed(7)
  n <- 40L
  n_re <- 4L
  X <- Matrix::Matrix(cbind(1, rnorm(n)))
  A <- Matrix::sparseMatrix(
    i = seq_len(n), j = sample(n_re, n, replace = TRUE), x = 1,
    dims = c(n, n_re)
  )
  beta <- c(0.5, -0.3)
  gamma <- rnorm(n_re, sd = 0.4)
  eta <- as.vector(X %*% beta + A %*% gamma)
  prob <- 1 / (1 + exp(-eta))
  y <- rbinom(n, 1, prob)

  config <- list(
    beta = beta, gamma = gamma, theta = numeric(0),
    transform_theta = TRUE,
    obs_groups = adlaplace::obs_groups(A, num_shards = 3L)
  )
  model <- adlaplace:::density_data(
    y = y, A = A, X = X,
    theta_map = Matrix::Matrix(nrow = 0L, ncol = 0L),
    ad_kind = "observations",
    density = "binomial_obs"
  )
  ptr <- adlaplace::ad_pack_ptr(model, config)

  x <- c(beta, gamma)
  manual <- sum(stats::dbinom(y, 1, prob, log = TRUE))
  expect_equal(
    adlaplace::joint_log_dens(ptr, x, negative = FALSE),
    manual,
    tolerance = 1e-10
  )

  g <- as.numeric(adlaplace::grad(ptr, x, inner = FALSE, negative = FALSE))
  eps <- 1e-6
  g_fd <- vapply(seq_along(x), function(i) {
    xp <- x; xm <- x; xp[i] <- xp[i] + eps; xm[i] <- xm[i] - eps
    (adlaplace::joint_log_dens(ptr, xp, negative = FALSE) -
       adlaplace::joint_log_dens(ptr, xm, negative = FALSE)) / (2 * eps)
  }, numeric(1))
  expect_equal(g, g_fd, tolerance = 1e-5)
})

test_that("binomial term carries the expected slots and NULL theta_info", {
  term <- adlaplace::binom("yn")
  expect_s4_class(term, "binomial")
  expect_identical(term@density, "binomial_obs")
  expect_identical(term@ad_kind, "observations")
  expect_null(adlaplace::theta_info(term))
})

test_that("adlaplace() fits binomial GLMM on bacteria data", {
  skip_if_not_installed("MASS")

  data(bacteria, package = "MASS")
  bacteria$yn <- as.integer(bacteria$y == "y")
  bacteria$wk <- as.integer(bacteria$week > 2)

  fit <- adlaplace::adlaplace(
    binom(yn) ~ trt + wk + iid(ID, init = 1),
    data = bacteria,
    config = list(num_shards = 20L),
    control = list(maxit = 300L)
  )
  expect_identical(fit$details$outer_opt$convergence, 0L)

  est <- coef(fit)
  expect_gt(est[["intercept"]], 1)
  expect_lt(est[["wk_linear"]], 0)

  pql <- MASS::glmmPQL(
    y ~ trt + I(week > 2), random = ~ 1 | ID,
    family = binomial, data = bacteria, verbose = FALSE
  )
  fe <- nlme::fixef(pql)
  expect_equal(est[["intercept"]], unname(fe[["(Intercept)"]]), tolerance = 0.3)
  expect_equal(est[["wk_linear"]], unname(fe[["I(week > 2)TRUE"]]), tolerance = 0.3)
})

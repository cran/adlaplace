test_that("poisson term carries the expected slots and NULL theta_info", {
  term <- adlaplace::pois("Y")
  expect_s4_class(term, "poisson")
  expect_identical(term@density, "poisson_obs")
  expect_identical(term@ad_kind, "observations")
  expect_null(adlaplace::theta_info(term))
})

test_that("collect_terms parses pois(y) on the LHS", {
  terms <- adlaplace::collect_terms(adlaplace::pois(y) ~ x)
  pois_idx <- which(vapply(terms, function(x) {
    methods::is(x, "poisson")
  }, logical(1L)))
  expect_length(pois_idx, 1L)
  expect_identical(terms[[pois_idx]]@density, "poisson_obs")
  expect_identical(terms[[pois_idx]]@name, "y")
})

test_that("poisson_obs matches dpois with an offset", {
  set.seed(7)
  n <- 40L
  n_re <- 4L
  X <- Matrix::Matrix(cbind(1, rnorm(n)))
  A <- Matrix::sparseMatrix(
    i = seq_len(n),
    j = sample(n_re, n, replace = TRUE),
    x = 1,
    dims = c(n, n_re)
  )
  beta <- c(0.5, -0.3)
  gamma <- rnorm(n_re, sd = 0.4)
  offset <- log(runif(n, 0.5, 3))
  eta <- as.vector(X %*% beta + A %*% gamma)
  y <- rpois(n, exp(eta + offset))

  config <- list(
    beta = beta,
    gamma = gamma,
    theta = numeric(0),
    transform_theta = TRUE,
    offset = offset,
    obs_groups = adlaplace::obs_groups(A, num_shards = 3L)
  )
  model <- adlaplace:::density_data(
    y = y,
    A = A,
    X = X,
    theta_map = Matrix::sparseMatrix(
      i = integer(0), j = integer(0), dims = c(0L, 1L)
    ),
    ad_kind = "observations",
    density = "poisson_obs"
  )
  ptr <- adlaplace::ad_pack_ptr(model, config)

  x <- c(beta, gamma)
  manual <- sum(stats::dpois(y, exp(eta + offset), log = TRUE))
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

test_that("poisson_obs without offset and without shards", {
  set.seed(8)
  n <- 15L
  X <- Matrix::Matrix(cbind(1, rnorm(n)))
  A <- Matrix::sparseMatrix(
    i = seq_len(n), j = rep(1L, n), x = 1, dims = c(n, 1L)
  )
  beta <- c(0.2, 0.1)
  gamma <- 0.05
  eta <- as.vector(X %*% beta + A %*% gamma)
  y <- rpois(n, exp(eta))

  config <- list(
    beta = beta, gamma = gamma, theta = numeric(0),
    transform_theta = TRUE
  )
  model <- adlaplace:::density_data(
    y = y,
    A = A,
    X = X,
    theta_map = Matrix::sparseMatrix(
      i = integer(0), j = integer(0), dims = c(0L, 1L)
    ),
    ad_kind = "observations",
    density = "poisson_obs"
  )
  ptr <- adlaplace::ad_pack_ptr(model, config)
  expect_equal(
    adlaplace::joint_log_dens(ptr, c(beta, gamma), negative = FALSE),
    sum(stats::dpois(y, exp(eta), log = TRUE)),
    tolerance = 1e-10
  )
})

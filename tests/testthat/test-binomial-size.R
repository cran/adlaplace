test_that("binomial_obs with size matches softplus kernel", {
  set.seed(11)
  n <- 30L
  X <- Matrix::Matrix(cbind(1, rnorm(n)))
  beta <- c(-0.2, 0.4)
  eta <- as.vector(X %*% beta)
  N <- sample(5:20, n, replace = TRUE)
  prob <- 1 / (1 + exp(-eta))
  y <- rbinom(n, N, prob)

  config <- list(
    beta = beta, gamma = numeric(0), theta = numeric(0),
    transform_theta = TRUE
  )
  model <- adlaplace:::density_data(
    y = y, X = X,
    theta_map = Matrix::Matrix(nrow = 0L, ncol = 0L),
    ad_kind = "observations",
    density = "binomial_obs",
    weights = as.numeric(N)
  )
  ptr <- adlaplace::ad_pack_ptr(model, config)
  x <- beta
  # dbinom includes choose(N,y); AD density omits that constant
  manual_kernel <- sum(y * eta - N * log1p(exp(eta)))
  expect_equal(
    adlaplace::joint_log_dens(ptr, x, negative = FALSE),
    manual_kernel,
    tolerance = 1e-10
  )
})

test_that("binom(y, size = N) wires weights through model_data", {
  dat <- data.frame(
    y = c(1L, 2L, 0L),
    N = c(5L, 8L, 3L),
    x1 = c(0.1, -0.2, 0.3)
  )
  md <- adlaplace::model_data(
    binom(y, size = N) ~ x1,
    data = dat
  )
  expect_equal(md$observations$y@weights, as.numeric(dat$N))
  term <- md$terms[[grep("binomial", names(md$terms))[1L]]]
  expect_s4_class(term, "binomial")
  expect_identical(term@size, "N")
})

test_that("collect_terms evaluates in formula env and coerces size", {
  f <- binom(y, size = N) ~ x1
  terms <- adlaplace::collect_terms(f)
  bin <- terms[[grep("binomial", names(terms))[1L]]]
  expect_s4_class(bin, "binomial")
  expect_identical(bin@name, "y")
  expect_identical(bin@size, "N")
})

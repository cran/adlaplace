test_that("ad_pack and derivatives run on small GLMM data", {
  set.seed(0)
  Nobs <- 80L
  Nrandom1 <- 4L
  Nrandom2 <- 5L
  X <- Matrix::Matrix(cbind(1, rbinom(Nobs, 1, prob = 0.5)))
  AmatList <- list(
    Matrix::sparseMatrix(i = seq_len(Nobs), j = sample(Nrandom1, Nobs, replace = TRUE), x = 1),
    Matrix::sparseMatrix(i = seq_len(Nobs), j = sample(Nrandom2, Nobs, replace = TRUE), x = 1)
  )
  Amat <- do.call(cbind, AmatList)
  config <- list(
    beta = rep(0, ncol(X)),
    theta = c(-1, -1, -1),
    transform_theta = TRUE,
    gamma = rep(0, ncol(Amat)),
    obs_groups = adlaplace::obs_groups(Amat, num_shards = 20L),
    num_threads = 1L,
    verbose = FALSE,
    package = "adlaplace"
  )
  n_beta <- length(config$beta)
  n_gamma <- length(config$gamma)
  model <- test_ad_data(
    y = rpois(Nobs, 2),
    A = Amat,
    X = X,
    config = config,
    theta_local_row = length(config$theta) - 1L
  )
  random_shard <- test_random_shard(
    data = model,
    config = config,
    gamma_ids = seq.int(0L, length.out = ncol(Amat)),
    theta_id = 0L,
    Q = rep(1, ncol(Amat))
  )
  ad_ptr <- do.call(c, list(
    adlaplace::ad_pack_ptr(as_shard(model, "observations", "nbinom_obs"), config),
    adlaplace::ad_pack_ptr(random_shard, config),
    adlaplace::ad_pack_ptr(as_shard(model, "parameters", "nbinom_extra"), config)
  ))
  ad_pack <- adlaplace::ad_pack(ad_ptr)
  x <- c(config$beta, config$gamma, config$theta)

  dens <- adlaplace::joint_log_dens(ad_pack, x)
  expect_true(is.finite(dens))

  g <- adlaplace::grad(ad_pack, x)
  expect_equal(length(g), length(x))

  H <- adlaplace::hessian(ad_pack, x)
  expect_true(inherits(H, "Matrix"))

  inner_res <- adlaplace::inner_opt(
    parameters = c(config$beta, config$theta),
    gamma = config$gamma,
    ad_pack = ad_pack,
    control = list(maxit = 5L, report.level = 0, report.freq = 0),
    deriv = FALSE
  )
  expect_true(is.finite(inner_res$log_lik))
  expect_equal(inner_res$neg_log_lik, -inner_res$log_lik)
  expect_equal(
    inner_res$parameters,
    c(config$beta, config$theta)
  )
  expect_equal(
    length(inner_res$parameters) + length(config$gamma),
    length(inner_res$full_parameters)
  )

  config_min <- list(verbose = FALSE)
  ll_res <- adlaplace::log_lik_laplace(
    x = c(config$beta, config$theta),
    config = config_min,
    gamma = config$gamma,
    ad_pack = ad_pack,
    control = list(maxit = 5L, report.level = 0, report.freq = 0),
    deriv = FALSE
  )
  expect_true(is.finite(ll_res$log_lik))

  ll_deriv <- adlaplace::log_lik_laplace(
    x = c(config$beta, config$theta),
    config = config_min,
    gamma = config$gamma,
    ad_pack = ad_pack,
    control = list(maxit = 5L, report.level = 0, report.freq = 0),
    deriv = TRUE
  )
  expect_true(is.finite(ll_deriv$log_lik))
  expect_equal(length(ll_deriv$deriv$d_neg_log_lik), length(config$beta) + length(config$theta))
})

test_that("model_data builds data for iwp formula", {
  skip_if_not_installed("mgcv")
  set.seed(1)
  n <- 30L
  df <- data.frame(
    y = rnorm(n),
    x = runif(n),
    id = rep(1:10, each = 3L)
  )
  # bare response defaults to normal(y)
  md <- adlaplace::model_data(
    data = df,
    formula = y ~ intercept() + linear(x),
    verbose = FALSE
  )
  expect_true(is.list(md$term_data))
  expect_true(length(md$observations) >= 1L)
  expect_true(nrow(md$observations[[1]]@XTp) >= 1L)
  expect_true(length(md$term_data$info$beta$init) >= 1L)
})

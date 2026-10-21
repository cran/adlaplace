test_that("importance weights with flat log-ratio recover MVN mean", {
  skip_if_not_installed("mvQuad")
  ngamma <- 2L
  mean <- c(0.1, -0.2)
  cov <- diag(ngamma) * 0.5
  grid <- mvQuad::createNIGrid(
    dim = ngamma,
    type = "nHN",
    level = 3L,
    ndConstruction = "sparse"
  )
  mvQuad::rescale(grid, m = mean, C = cov, dec.type = 2L)
  nodes <- mvQuad::getNodes(grid)
  weights <- as.numeric(mvQuad::getWeights(grid))
  rw <- weights
  est <- colSums(nodes * rw) / sum(rw)
  expect_equal(as.numeric(est), mean, tolerance = 1e-10)
})

test_that("mean_mvquad smoke on small Laplace fit", {
  skip_if_not_installed("mvQuad")
  set.seed(11)
  Nobs <- 30L
  X <- Matrix::Matrix(cbind(1, rbinom(Nobs, 1, prob = 0.5)))
  Amat <- Matrix::sparseMatrix(
    i = seq_len(Nobs),
    j = sample(4L, Nobs, replace = TRUE),
    x = 1
  )
  config <- list(
    beta = rep(0, 2),
    theta = -1,
    transform_theta = TRUE,
    gamma = rep(0, ncol(Amat)),
    obs_groups = adlaplace::obs_groups(Amat, num_shards = 4L),
    num_threads = 1L,
    verbose = FALSE
  )
  model <- test_ad_data(
    y = rpois(Nobs, 2),
    A = Amat,
    X = X,
    config = config
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
  af <- adlaplace::ad_pack(ad_ptr, num_threads = 1L)
  x_outer <- c(config$beta, config$theta)
  ll <- adlaplace::log_lik_laplace(
    x = x_outer,
    config = config,
    gamma = config$gamma,
    ad_pack = af,
    control = list(maxit = 5L, report.level = 0, report.freq = 0),
    deriv = TRUE
  )

  est <- adlaplace::mean_mvquad(
    parameters = x_outer,
    mode = ll$inner_opt$solution,
    cov = ll$hessian$H_inv,
    ad_pack = af,
    n = 3L
  )
  expect_equal(length(est), ncol(Amat))
  expect_true(all(is.finite(est)))

  expect_error(
    adlaplace::mean_mvquad(
      parameters = x_outer,
      mode = ll$inner_opt$solution,
      ad_pack = af,
      n = 3L
    ),
    "cov is required"
  )
})

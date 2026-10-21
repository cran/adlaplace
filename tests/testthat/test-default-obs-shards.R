test_that("missing config$obs_groups uses one observation shard", {
  set.seed(2)
  Nobs <- 40L
  X <- Matrix::Matrix(cbind(1, rbinom(Nobs, 1, prob = 0.5)))
  Amat <- Matrix::sparseMatrix(
    i = seq_len(Nobs),
    j = sample(4L, Nobs, replace = TRUE),
    x = 1
  )
  config <- list(
    beta = rep(0, ncol(X)),
    theta = c(-1, -1, -1),
    transform_theta = TRUE,
    gamma = rep(0, ncol(Amat)),
    num_threads = 1L,
    verbose = FALSE
  )
  model <- test_ad_data(
    y = rpois(Nobs, 2),
    A = Amat,
    X = X,
    config = config,
    theta_local_row = length(config$theta) - 1L
  )
  obs_ptr <- adlaplace::ad_pack_ptr(
    as_shard(model, "observations", "nbinom_obs"),
    config
  )
  expect_equal(adlaplace:::n_groups(obs_ptr), 1L)
})

test_that("ad_pack_ptr normalizes missing config$gamma for observation shard", {
  set.seed(2)
  Nobs <- 40L
  X <- Matrix::Matrix(cbind(1, rnorm(Nobs)))
  Amat <- Matrix::sparseMatrix(
    i = seq_len(Nobs),
    j = sample(5L, Nobs, replace = TRUE),
    x = 1
  )
  config <- list(
    beta = rep(0, ncol(X)),
    theta = c(-1, -1),
    transform_theta = TRUE,
    obs_groups = adlaplace::obs_groups(Amat, num_shards = 5L),
    verbose = FALSE
  )
  model <- test_ad_data(
    y = rpois(Nobs, 2),
    A = Amat,
    X = X,
    config = config,
    theta_local_row = length(config$theta) - 1L,
    ad_kind = "observations",
    density = "nbinom_obs"
  )
  expect_silent(ptr <- adlaplace::ad_pack_ptr(model, config))
  expect_true(inherits(ptr, "ad_pack_ptr"))
})

test_that("validate_config_layout allows omitted config$gamma", {
  A <- Matrix::sparseMatrix(
    i = c(0L, 1L),
    j = c(0L, 1L),
    x = 1,
    dims = c(2L, 2L),
    index1 = FALSE,
    repr = "C"
  )
  model <- adlaplace:::density_data(y = 1:2, A = A, X = matrix(1, 2, 2))
  config <- list(beta = rep(0, 2))
  expect_silent(adlaplace:::validate_config_layout(model, config))
})

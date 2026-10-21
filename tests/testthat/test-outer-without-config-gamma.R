test_that("outer_fn and outer_gr work without config$gamma on minimal GLMM", {
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
  config_full <- list(
    beta = rep(0, ncol(X)),
    theta = c(-1, -1, -1),
    transform_theta = TRUE,
    gamma = rep(0, ncol(Amat)),
    obs_groups = adlaplace::obs_groups(Amat, num_shards = 20L),
    num_threads = 1L,
    verbose = FALSE,
    package = "adlaplace"
  )
  model <- test_ad_data(
    y = rpois(Nobs, 2),
    A = Amat,
    X = X,
    config = config_full,
    theta_local_row = length(config_full$theta) - 1L
  )
  random_shard <- test_random_shard(
    data = model,
    config = config_full,
    gamma_ids = seq.int(0L, length.out = ncol(Amat)),
    theta_id = 0L,
    Q = rep(1, ncol(Amat))
  )
  ad_ptr <- do.call(c, list(
    adlaplace::ad_pack_ptr(as_shard(model, "observations", "nbinom_obs"), config_full),
    adlaplace::ad_pack_ptr(random_shard, config_full),
    adlaplace::ad_pack_ptr(as_shard(model, "parameters", "nbinom_extra"), config_full)
  ))
  ad_pack <- adlaplace::ad_pack(ad_ptr)
  config_min <- list(verbose = FALSE)
  x0 <- c(config_full$beta, config_full$theta)
  cache <- new.env(parent = emptyenv())
  cache$gamma <- rep(0, ad_pack@sizes["gamma"])

  val <- adlaplace::outer_fn(x0, config_min, cache, ad_pack)
  gr <- adlaplace::outer_gr(x0, config_min, cache, ad_pack)
  expect_true(is.finite(val))
  expect_equal(length(gr), length(x0))
  expect_identical(cache$fg_evals, 1L)

  # Second gr at the same x should reuse the cached Laplace evaluation.
  gr2 <- adlaplace::outer_gr(x0, config_min, cache, ad_pack)
  expect_equal(gr2, gr)
  expect_identical(cache$fg_evals, 1L)

  fit <- stats::optim(
    par = x0,
    fn = adlaplace::outer_fn,
    gr = adlaplace::outer_gr,
    method = "L-BFGS-B",
    control = list(maxit = 3L, trace = 0L),
    config = config_min,
    ad_pack = ad_pack,
    cache = cache
  )
  expect_true(is.finite(fit$value))
  # L-BFGS-B pairs fn+gr; with caching, fg_evals should be close to fn counts.
  expect_lte(cache$fg_evals, fit$counts[["function"]] + 1L)
})

test_that("model_data ad_pack and outer wrappers work without config$gamma", {
  skip_if_not_installed("mgcv")
  set.seed(1)
  n <- 40L
  df <- data.frame(
    y = rpois(n, 2),
    x1 = rnorm(n),
    x2 = runif(n),
    fac = rep(1:10, each = 4L)
  )
  formula <- adlaplace::nbinom(y, init = 0.15) ~
    x1 +
    adlaplace::iwp(x2, ref_value = 0.5, p = 2, knots = seq(0, 1, len = 11)) +
    adlaplace::iid(fac, init = 0.25)
  md <- adlaplace::model_data(data = df, formula = formula, verbose = FALSE)
  config <- list(
    transform_theta = TRUE,
    obs_groups = adlaplace::obs_groups(md$term_data$A, num_shards = 10L),
    verbose = FALSE
  )
  ad_pack <- adlaplace::ad_pack(md, config)
  n_gamma <- as.integer(ad_pack@sizes["gamma"])
  skip_if(n_gamma < 1L)

  to_keep <- c("init", "lower", "upper", "parscale")
  theta_for_opt <- adlaplace::apply_theta_log(md$term_data$info$theta, cols = to_keep)
  config$opt <- as.list(rbind(md$term_data$info$beta[, to_keep], theta_for_opt[, to_keep]))
  x0 <- config$opt$init

  cache <- new.env(parent = emptyenv())
  cache$gamma <- rep(0, n_gamma)

  expect_true(is.finite(adlaplace::outer_fn(x0, config, cache, ad_pack)))
  expect_equal(
    length(adlaplace::outer_gr(x0, config, cache, ad_pack)),
    length(x0)
  )
})

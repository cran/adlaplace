test_that("grad_obs_units matches one-stratum ELGM packs", {
  set.seed(11)
  n_date <- 6L
  n_per <- 4L
  n <- n_date * n_per
  data <- data.frame(
    count = rpois(n, lambda = 2),
    hum = rnorm(n),
    region = rep(1:2, length.out = n),
    date = rep(seq_len(n_date), each = n_per),
    year = 2002L
  )
  formula <- dirichlet_multinom(
    count,
    by = c("year", "region", "date"),
    init = 0.1
  ) ~ hum + iid(date)
  md <- model_data(formula = formula, data = data, verbose = FALSE)
  obs <- md$observations$count
  n_s <- ncol(md$term_data$elgm_matrix)
  expect_gt(n_s, 0L)

  n_beta <- nrow(obs@beta_map)
  n_gamma <- nrow(obs@gamma_map)
  n_theta <- nrow(obs@theta_map)
  x <- c(
    rep(0.1, n_beta),
    rnorm(n_gamma, sd = 0.05),
    rep(log(0.1), n_theta)
  )

  G <- grad_obs_units(obs, x, batch_size = 2L)
  expect_s4_class(G, "dgCMatrix")
  expect_equal(dim(G), c(n_gamma, n_s))

  # Default transform_theta is TRUE (matches ad_pack / adlaplace).
  expect_equal(
    as.matrix(G),
    as.matrix(grad_obs_units(obs, x, batch_size = 2L, config = list(transform_theta = TRUE))),
    tolerance = 1e-10
  )

  og <- Matrix::sparseMatrix(
    i = seq.int(0L, n_s - 1L),
    j = seq.int(0L, n_s - 1L),
    x = 1,
    dims = c(n_s, n_s),
    index1 = FALSE
  )
  ptr <- ad_pack_ptr(
    obs,
    list(
      beta = x[seq_len(n_beta)],
      gamma = x[seq.int(n_beta + 1L, length.out = n_gamma)],
      theta = x[seq.int(n_beta + n_gamma + 1L, length.out = n_theta)],
      transform_theta = TRUE,
      obs_groups = og,
      compact_tape = TRUE,
      verbose = FALSE
    )
  )
  gamma_idx <- seq.int(n_beta + 1L, length.out = n_gamma)
  for (s in seq.int(0L, n_s - 1L)) {
    g <- as.numeric(grad(ptr, x, ad_shards = s, inner = TRUE, negative = TRUE))
    expect_equal(
      as.numeric(G[, s + 1L]),
      g[gamma_idx],
      tolerance = 1e-6,
      info = paste("stratum", s)
    )
  }

  G1 <- grad_obs_units(obs, x, batch_size = 1L)
  expect_equal(as.matrix(G), as.matrix(G1), tolerance = 1e-10)
})

test_that("grad_obs_units matches one-obs packs without ELGM", {
  set.seed(19)
  n <- 12L
  n_re <- 4L
  X <- Matrix::Matrix(cbind(1, rnorm(n)))
  A <- Matrix::sparseMatrix(
    i = seq_len(n),
    j = sample(n_re, n, replace = TRUE),
    x = 1,
    dims = c(n, n_re)
  )
  beta <- c(0.4, -0.2)
  gamma <- rnorm(n_re, sd = 0.3)
  y <- rpois(n, exp(as.vector(X %*% beta + A %*% gamma)))
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
  x <- c(beta, gamma)

  G <- grad_obs_units(model, x, batch_size = 3L)
  expect_equal(dim(G), c(n_re, n))

  og <- Matrix::sparseMatrix(
    i = seq.int(0L, n - 1L),
    j = seq.int(0L, n - 1L),
    x = 1,
    dims = c(n, n),
    index1 = FALSE
  )
  ptr <- ad_pack_ptr(
    model,
    list(
      beta = beta,
      gamma = gamma,
      theta = numeric(0),
      transform_theta = TRUE,
      obs_groups = og,
      compact_tape = TRUE,
      verbose = FALSE
    )
  )
  gamma_idx <- seq.int(length(beta) + 1L, length.out = n_re)
  for (s in seq.int(0L, n - 1L)) {
    g <- as.numeric(grad(ptr, x, ad_shards = s, inner = TRUE, negative = TRUE))
    expect_equal(
      as.numeric(G[, s + 1L]),
      g[gamma_idx],
      tolerance = 1e-6,
      info = paste("obs", s)
    )
  }

  G1 <- grad_obs_units(model, x, batch_size = 1L)
  expect_equal(as.matrix(G), as.matrix(G1), tolerance = 1e-10)
})

test_that("obs_groups_units identity matches grad_obs_units / obs_units pack", {
  set.seed(11)
  n_date <- 6L
  n_per <- 4L
  n <- n_date * n_per
  data <- data.frame(
    count = rpois(n, lambda = 2),
    hum = rnorm(n),
    region = rep(1:2, length.out = n),
    date = rep(seq_len(n_date), each = n_per),
    year = 2002L
  )
  formula <- dirichlet_multinom(
    count,
    by = c("year", "region", "date"),
    init = 0.1
  ) ~ hum + iid(date)
  md <- model_data(formula = formula, data = data, verbose = FALSE)
  obs <- md$observations$count
  n_s <- n_obs_units(obs)
  expect_equal(n_s, ncol(md$term_data$elgm_matrix))

  n_beta <- nrow(obs@beta_map)
  n_gamma <- nrow(obs@gamma_map)
  n_theta <- nrow(obs@theta_map)
  x <- c(
    rep(0.1, n_beta),
    rnorm(n_gamma, sd = 0.05),
    rep(log(0.1), n_theta)
  )

  G <- grad_obs_units(obs, x, batch_size = 2L)
  og_id <- obs_groups_units(n_s, grouping = "identity")
  expect_equal(dim(og_id), c(n_s, n_s))
  expect_equal(Matrix::nnzero(og_id), n_s)

  # Single-unit pack via obs_groups_units matches grad_obs_units column.
  s <- 2L
  ptr_one <- ad_pack_ptr(
    obs,
    list(
      beta = x[seq_len(n_beta)],
      gamma = x[seq.int(n_beta + 1L, length.out = n_gamma)],
      theta = x[seq.int(n_beta + n_gamma + 1L, length.out = n_theta)],
      transform_theta = TRUE,
      obs_groups = obs_groups_units(n_s, units = s, grouping = "identity"),
      compact_tape = TRUE,
      hessian_sparsity = FALSE,
      verbose = FALSE
    )
  )
  gamma_idx <- seq.int(n_beta + 1L, length.out = n_gamma)
  g <- as.numeric(grad(ptr_one, x, ad_shards = 0L, inner = TRUE, negative = TRUE))
  expect_equal(
    as.numeric(G[, s + 1L]),
    g[gamma_idx],
    tolerance = 1e-6
  )

  # config$obs_units is honored when building obs_groups for density_data.
  cfg_u <- list(obs_units = s)
  cfg_u <- adlaplace:::ensure_config_obs_groups(
    cfg_u,
    A = NULL,
    elgm_matrix = md$term_data$elgm_matrix,
    n_domain = n_s
  )
  expect_equal(
    as.matrix(cfg_u$obs_groups),
    as.matrix(obs_groups_units(n_s, units = s, grouping = "identity"))
  )

  # filter grouping drops empty shards and keeps selected units.
  og_coarse <- obs_groups(
    md$term_data$A,
    elgm_matrix = md$term_data$elgm_matrix,
    num_shards = 3L
  )
  keep <- c(0L, 2L, 4L)
  og_f <- obs_groups_units(
    n_s,
    units = keep,
    grouping = "filter",
    obs_groups = og_coarse
  )
  expect_equal(nrow(og_f), n_s)
  expect_lte(ncol(og_f), ncol(og_coarse))
  rows_hit <- sort(unique(methods::as(og_f, "TsparseMatrix")@i))
  expect_equal(rows_hit, keep)
})

test_that("hessian_sparsity = FALSE skips Hessian patterns", {
  set.seed(5)
  n <- 20L
  n_re <- 5L
  X <- Matrix::Matrix(cbind(1, rnorm(n)))
  A <- Matrix::sparseMatrix(
    i = seq_len(n),
    j = sample(n_re, n, replace = TRUE),
    x = 1,
    dims = c(n, n_re)
  )
  beta <- c(0.2, 0.1)
  gamma <- rnorm(n_re, sd = 0.2)
  y <- rpois(n, exp(as.vector(X %*% beta + A %*% gamma)))
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
  og <- adlaplace::obs_groups(A, num_shards = 2L)
  cfg <- list(
    beta = beta,
    gamma = gamma,
    theta = numeric(0),
    transform_theta = TRUE,
    obs_groups = og,
    compact_tape = TRUE,
    hessian_sparsity = FALSE,
    verbose = FALSE
  )
  ptr <- ad_pack_ptr(model, cfg)
  sz <- adlaplace:::get_sizes(ptr, 0L)
  expect_equal(as.integer(sz$nnz_hes_inner), 0L)
  expect_equal(as.integer(sz$nnz_hes_outer), 0L)
  expect_gt(as.integer(sz$nnz_grad_inner), 0L)

  x <- c(beta, gamma)
  g <- as.numeric(grad(ptr, x, inner = TRUE, negative = TRUE))
  expect_length(g, length(x))
  expect_true(any(abs(g[seq.int(length(beta) + 1L, length.out = n_re)]) > 0))
})

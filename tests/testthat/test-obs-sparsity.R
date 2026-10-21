pattern_to_dense <- function(rows, cols, n) {
  P <- matrix(0, n, n)
  if (!length(rows)) {
    return(P)
  }
  idx <- cbind(rows + 1L, cols + 1L)
  P[idx] <- 1
  P[idx[, c(2L, 1L)]] <- 1
  P
}

expect_pattern_covers <- function(H, rows, cols, tol = 1e-8) {
  n <- nrow(H)
  P <- pattern_to_dense(rows, cols, n)
  off <- abs(H[P == 0])
  expect_true(length(off) == 0L || max(off) < tol)
}

union_pattern <- function(ptr, n_g, inner = FALSE) {
  rows <- integer(0)
  cols <- integer(0)
  for (g in seq_len(n_g) - 1L) {
    pa <- adlaplace:::get_sparse_pattern(ptr, g)
    if (inner) {
      rows <- c(rows, pa$row_hess_inner)
      cols <- c(cols, pa$col_hess_inner)
    } else {
      rows <- c(rows, pa$row_hess)
      cols <- c(cols, pa$col_hess)
    }
  }
  list(row = rows, col = cols)
}

check_obs_sparsity <- function(model, config, x, expect_inner_diag = FALSE) {
  cfg_a <- config
  cfg_a$obs_hessian <- "analytic"
  cfg_d <- config
  cfg_d$obs_hessian <- "discover"
  ptr_a <- adlaplace::ad_pack_ptr(model, cfg_a)
  ptr_d <- adlaplace::ad_pack_ptr(model, cfg_d)
  n_g <- adlaplace:::n_groups(ptr_a)
  expect_equal(n_g, adlaplace:::n_groups(ptr_d))

  Ha <- as.matrix(hessian(ptr_a, x, inner = FALSE, negative = FALSE))
  Hd <- as.matrix(hessian(ptr_d, x, inner = FALSE, negative = FALSE))
  expect_equal(Ha, Hd, tolerance = 1e-8)
  Hai <- as.matrix(hessian(ptr_a, x, inner = TRUE, negative = FALSE))
  Hdi <- as.matrix(hessian(ptr_d, x, inner = TRUE, negative = FALSE))
  expect_equal(Hai, Hdi, tolerance = 1e-8)

  pu <- union_pattern(ptr_a, n_g, inner = FALSE)
  pui <- union_pattern(ptr_a, n_g, inner = TRUE)
  expect_pattern_covers(Hd, pu$row, pu$col)
  expect_pattern_covers(Hdi, pui$row, pui$col)

  for (g in seq_len(n_g) - 1L) {
    pa <- adlaplace:::get_sparse_pattern(ptr_a, g)
    pd <- adlaplace:::get_sparse_pattern(ptr_d, g)
    expect_lte(length(pa$row_hess), length(pd$row_hess))
    expect_lte(length(pa$row_hess_inner), length(pd$row_hess_inner))
    Hg <- as.matrix(hessian(
      ptr_d, x, ad_shards = g, inner = FALSE, negative = FALSE
    ))
    Hgi <- as.matrix(hessian(
      ptr_d, x, ad_shards = g, inner = TRUE, negative = FALSE
    ))
    expect_pattern_covers(Hg, pa$row_hess, pa$col_hess)
    expect_pattern_covers(Hgi, pa$row_hess_inner, pa$col_hess_inner)
    if (isTRUE(expect_inner_diag)) {
      inner <- cbind(pa$row_hess_inner, pa$col_hess_inner)
      off <- inner[inner[, 1L] != inner[, 2L], , drop = FALSE]
      expect_equal(nrow(off), 0L)
    }
  }
  list(ptr_a = ptr_a, ptr_d = ptr_d)
}

glm_obs_setup <- function(family, compact_tape = TRUE, n = 36L, n_re = 8L,
                          n_shards = 3L, with_weights = FALSE) {
  X <- Matrix::Matrix(cbind(1, rnorm(n), rbinom(n, 1, 0.4)))
  A <- Matrix::sparseMatrix(
    i = seq_len(n),
    j = sample(n_re, n, replace = TRUE),
    x = 1,
    dims = c(n, n_re)
  )
  beta <- c(0.3, -0.2, 0.1)
  gamma <- rnorm(n_re, sd = 0.25)
  eta <- as.vector(X %*% beta + A %*% gamma)
  weights <- if (with_weights) sample(1:4, n, replace = TRUE) else numeric(0)

  if (identical(family, "poisson_obs")) {
    y <- rpois(n, lambda = exp(pmin(eta, 4)))
    theta <- numeric(0)
    theta_map <- Matrix::sparseMatrix(
      i = integer(0), j = integer(0), dims = c(0L, 1L)
    )
  } else if (identical(family, "binomial_obs")) {
    ntrials <- if (with_weights) weights else rep(1, n)
    y <- rbinom(n, ntrials, 1 / (1 + exp(-eta)))
    theta <- numeric(0)
    theta_map <- Matrix::Matrix(nrow = 0L, ncol = 0L)
  } else if (identical(family, "gaussian_obs")) {
    sigma <- 0.8
    y <- rnorm(n, eta, sigma)
    theta <- log(sigma)
    theta_map <- Matrix::sparseMatrix(i = 1L, j = 1L, dims = c(1L, 1L))
  } else if (identical(family, "nbinom_obs")) {
    y <- rnbinom(n, size = 5, mu = exp(pmin(eta, 3)))
    theta <- log(1 / sqrt(5))
    theta_map <- Matrix::sparseMatrix(i = 1L, j = 1L, dims = c(1L, 1L))
  } else {
    stop("unknown family: ", family)
  }

  config <- list(
    beta = beta,
    gamma = gamma,
    theta = theta,
    transform_theta = TRUE,
    obs_groups = adlaplace::obs_groups(A, num_shards = n_shards),
    compact_tape = compact_tape,
    verbose = FALSE
  )
  model <- adlaplace:::density_data(
    y = y,
    A = A,
    X = X,
    theta_map = theta_map,
    ad_kind = "observations",
    density = family,
    weights = weights
  )
  x <- c(beta, gamma, theta)
  list(model = model, config = config, x = x)
}

test_that("poisson_obs analytic Hessian sparsity covers numeric Hessian", {
  set.seed(11)
  for (compact in c(TRUE, FALSE)) {
    fx <- glm_obs_setup("poisson_obs", compact_tape = compact)
    check_obs_sparsity(fx$model, fx$config, fx$x, expect_inner_diag = TRUE)
  }
})

test_that("binomial_obs analytic Hessian sparsity (optional weights)", {
  set.seed(12)
  fx <- glm_obs_setup("binomial_obs", compact_tape = TRUE, with_weights = TRUE)
  check_obs_sparsity(fx$model, fx$config, fx$x, expect_inner_diag = TRUE)
})

test_that("gaussian_obs analytic Hessian sparsity includes theta crosses", {
  set.seed(13)
  fx <- glm_obs_setup("gaussian_obs", compact_tape = TRUE)
  res <- check_obs_sparsity(
    fx$model, fx$config, fx$x, expect_inner_diag = TRUE
  )
  pa <- adlaplace:::get_sparse_pattern(res$ptr_a, 0L)
  n_beta <- length(fx$config$beta)
  n_gamma <- length(fx$config$gamma)
  theta_index <- n_beta + n_gamma
  outer <- cbind(row = pa$row_hess, col = pa$col_hess)
  has_theta_diag <- any(outer[, 1L] == theta_index & outer[, 2L] == theta_index)
  expect_true(has_theta_diag)
  has_cross <- any(
    (outer[, 1L] == theta_index & outer[, 2L] != theta_index) |
      (outer[, 2L] == theta_index & outer[, 1L] != theta_index)
  )
  expect_true(has_cross)
})

test_that("nbinom_obs analytic Hessian sparsity includes theta crosses", {
  set.seed(14)
  fx <- glm_obs_setup("nbinom_obs", compact_tape = TRUE)
  check_obs_sparsity(fx$model, fx$config, fx$x, expect_inner_diag = TRUE)
})

test_that("dirichlet_multinomial analytic Hessian uses per-stratum cliques", {
  set.seed(7)
  n_strata <- 12L
  n_per <- 5L
  n <- n_strata * n_per
  data <- data.frame(
    count = rpois(n, lambda = 2),
    hum = rnorm(n),
    region = rep(1:3, length.out = n),
    date = rep(seq_len(n_strata), each = n_per),
    year = 2002L
  )
  formula <- dirichlet_multinom(
    count,
    by = c("year", "region", "date"),
    init = 0.1
  ) ~ hum + iid(date)
  md <- model_data(formula = formula, data = data, verbose = FALSE)
  obs <- md$observations$count
  og <- obs_groups(
    md$term_data$A,
    elgm_matrix = md$term_data$elgm_matrix,
    num_shards = 4L
  )
  expect_equal(ncol(og), 4L)

  make_cfg <- function(compact) {
    list(
      beta = rep(0.1, nrow(obs@beta_map)),
      gamma = rnorm(nrow(obs@gamma_map), sd = 0.05),
      theta = rep(log(0.1), nrow(obs@theta_map)),
      transform_theta = TRUE,
      obs_groups = og,
      compact_tape = compact,
      verbose = FALSE
    )
  }
  for (compact in c(TRUE, FALSE)) {
    cfg <- make_cfg(compact)
    x <- c(cfg$beta, cfg$gamma, cfg$theta)
    check_obs_sparsity(obs, cfg, x, expect_inner_diag = FALSE)
  }
})

test_that("hessian_sparsity = FALSE still yields empty observation Hessians", {
  set.seed(15)
  fx <- glm_obs_setup("poisson_obs", compact_tape = TRUE)
  fx$config$hessian_sparsity <- FALSE
  ptr <- adlaplace::ad_pack_ptr(fx$model, fx$config)
  sz <- adlaplace:::get_sizes(ptr, 0L)
  expect_equal(as.integer(sz$nnz_hes_inner), 0L)
  expect_equal(as.integer(sz$nnz_hes_outer), 0L)
  expect_gt(as.integer(sz$nnz_grad_inner), 0L)
})

test_that("obs_hessian analytic is the default verbose path", {
  set.seed(16)
  fx <- glm_obs_setup("poisson_obs", n = 20L, n_re = 4L, n_shards = 2L)
  fx$config$verbose <- TRUE
  out_a <- paste(capture.output(
    adlaplace::ad_pack_ptr(fx$model, fx$config)
  ), collapse = "\n")
  expect_match(out_a, "using external Hessian sparsity pattern")
  expect_false(grepl("discovering Hessian sparsity pattern", out_a))

  fx$config$obs_hessian <- "discover"
  out_d <- paste(capture.output(
    adlaplace::ad_pack_ptr(fx$model, fx$config)
  ), collapse = "\n")
  expect_match(out_d, "discovering Hessian sparsity pattern")
})

test_that("obs_hessian rejects unknown values", {
  set.seed(17)
  fx <- glm_obs_setup("poisson_obs", n = 12L, n_re = 3L, n_shards = 2L)
  fx$config$obs_hessian <- "nope"
  expect_error(
    adlaplace::ad_pack_ptr(fx$model, fx$config),
    "obs_hessian"
  )
})

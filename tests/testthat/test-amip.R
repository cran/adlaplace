test_that("amip_set finds a flip set or reports impossibility", {
  psi <- c(0.5, 0.3, 0.1, -0.2)
  S <- amip_set(psi, phi = 0.7)
  expect_equal(S$idx, c(1L, 2L))
  expect_equal(S$approx_change, -(0.5 + 0.3), tolerance = 1e-12)
  expect_equal(S$alpha, 2 / 4)

  psi_neg <- c(0.5, 0.3, 0.1, -0.2, -0.15)
  S_neg <- amip_set(psi_neg, phi = -0.3)
  expect_equal(sort(S_neg$idx), c(4L, 5L))
  expect_equal(S_neg$approx_change, -(-0.2 + -0.15), tolerance = 1e-12)

  S_fail <- amip_set(psi, phi = 10)
  expect_equal(S_fail$idx, integer(0))
  expect_true(is.na(S_fail$alpha))

  S0 <- amip_set(psi, phi = 0)
  expect_equal(S0$idx, integer(0))
})

test_that("phi_gamma / influence_scores / ad_pack_drop work on ELGM toy", {
  set.seed(21)
  n_date <- 8L
  n_per <- 3L
  n <- n_date * n_per
  data <- data.frame(
    count = rpois(n, lambda = 3),
    hum = rnorm(n),
    region = rep(1:2, length.out = n),
    date = rep(seq_len(n_date), each = n_per),
    year = 2002L
  )
  formula <- dirichlet_multinom(
    count,
    by = c("year", "region", "date"),
    init = 0.1
  ) ~ hum + iid(date, init = 0.2)
  fit <- adlaplace(
    formula,
    data = data,
    config = list(num_threads = 1L, num_shards = 4L),
    control = list(maxit = 40L, trace = 0),
    verbose = FALSE,
    hessian = FALSE
  )
  expect_equal(n_obs_units(fit), ncol(fit$model_data$term_data$elgm_matrix))

  # coef path: unit vector on first gamma label
  lab <- names(fit$gamma)[[1L]]
  a_coef <- phi_gamma(fit, coef = lab)
  expect_equal(unname(a_coef[[lab]]), 1)
  expect_equal(sum(abs(a_coef)), 1)

  inf <- influence_scores(fit, a_coef, batch_size = 2L)
  expect_equal(length(inf$psi), inf$n_units)
  expect_equal(inf$phi, sum(a_coef * inf$gamma_hat), tolerance = 1e-10)
  expect_true(is.finite(inf$phi))

  S <- amip_set(inf$psi, inf$phi)
  # Drop-pack joint density equals full minus dropped units (obs only).
  if (length(S$idx)) {
    drop_idx <- S$idx
  } else {
    # Force a small drop for the likelihood check.
    drop_idx <- which.max(abs(inf$psi))
  }
  af_drop <- ad_pack_drop(fit, drop_idx)
  x_full <- inf$x_full
  ll_full_obs <- joint_log_dens(
    fit$ad_pack,
    x_full,
    negative = FALSE
  )
  # Observation contribution via sum of unit grads is awkward; compare
  # observation-only packs: full identity vs keep.
  obs <- fit$model_data$observations[[1L]]
  n_u <- n_obs_units(obs)
  n_beta <- nrow(obs@beta_map)
  n_gamma <- nrow(obs@gamma_map)
  n_theta <- nrow(obs@theta_map)
  cfg_base <- list(
    beta = x_full[seq_len(n_beta)],
    gamma = x_full[seq.int(n_beta + 1L, length.out = n_gamma)],
    theta = x_full[seq.int(n_beta + n_gamma + 1L, length.out = n_theta)],
    transform_theta = TRUE,
    compact_tape = TRUE,
    hessian_sparsity = FALSE,
    verbose = FALSE
  )
  ptr_full <- ad_pack_ptr(
    obs,
    modifyList(cfg_base, list(
      obs_groups = obs_groups_units(n_u, grouping = "together")
    ))
  )
  keep0 <- setdiff(seq.int(0L, n_u - 1L), drop_idx - 1L)
  ptr_keep <- ad_pack_ptr(
    obs,
    modifyList(cfg_base, list(
      obs_groups = obs_groups_units(n_u, units = keep0, grouping = "together")
    ))
  )
  ptr_drop_only <- ad_pack_ptr(
    obs,
    modifyList(cfg_base, list(
      obs_groups = obs_groups_units(
        n_u,
        units = drop_idx - 1L,
        grouping = "together"
      )
    ))
  )
  ll_full <- joint_log_dens(ptr_full, x_full, negative = FALSE)
  ll_keep <- joint_log_dens(ptr_keep, x_full, negative = FALSE)
  ll_drop <- joint_log_dens(ptr_drop_only, x_full, negative = FALSE)
  expect_equal(ll_full, ll_keep + ll_drop, tolerance = 1e-8)

  # Inner confirmation: dropped pack should optimize.
  io_drop <- inner_opt(
    parameters = as.numeric(fit$par_info$mle_internal),
    gamma = inf$gamma_hat,
    ad_pack = af_drop,
    control = list(report.level = 0, report.freq = 0),
    deriv = FALSE,
    return_hessians = FALSE,
    verbose = FALSE
  )
  expect_true(is.numeric(io_drop$inner_opt$solution))
  expect_length(io_drop$inner_opt$solution, length(inf$gamma_hat))
})

test_that("phi_gamma and influence_scores work on binomial without ELGM", {
  skip_if_not_installed("MASS")
  data(bacteria, package = "MASS")
  bacteria$present <- as.integer(bacteria$y == "y")
  fit <- adlaplace(
    binom(present) ~
      intercept(sd = 10) +
      linear(trt, sd = 10) +
      linear(week, sd = 10) +
      iid(ID, init = 1) +
      prior(theta = 0, dist = "exp", median = 1),
    data = bacteria,
    config = list(num_threads = 1L, num_shards = 8L),
    control = list(maxit = 60L, trace = 0),
    verbose = FALSE,
    hessian = FALSE
  )
  expect_equal(n_obs_units(fit), nrow(bacteria))

  gamma_lab <- fit$model_data$term_data$info$gamma$gamma_label
  trt_labs <- grep("^trt_linear_", gamma_lab, value = TRUE)
  drug_lab <- trt_labs[endsWith(trt_labs, "drug")]
  if (!length(drug_lab)) {
    drug_lab <- trt_labs[[1L]]
  }
  a <- phi_gamma(fit, coef = drug_lab)
  expect_equal(unname(a[[drug_lab]]), 1)

  inf <- influence_scores(fit, a, batch_size = 16L)
  expect_equal(length(inf$psi), nrow(bacteria))
  S <- amip_set(inf$psi, inf$phi)
  expect_true(is.list(S))
  expect_true("idx" %in% names(S))

  if (length(S$idx)) {
    af_drop <- ad_pack_drop(fit, S$idx)
    io_drop <- inner_opt(
      parameters = as.numeric(fit$par_info$mle_internal),
      gamma = inf$gamma_hat,
      ad_pack = af_drop,
      control = list(report.level = 0, report.freq = 0),
      deriv = FALSE,
      return_hessians = FALSE,
      verbose = FALSE
    )
    phi_drop <- sum(a * io_drop$inner_opt$solution)
    # Linear approx should have the correct sign of change.
    expect_equal(
      sign(phi_drop - inf$phi),
      sign(S$approx_change),
      tolerance = 0
    )
  }
})

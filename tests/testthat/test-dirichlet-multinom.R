test_that("model_data attaches elgm_matrix for dirichlet_multinom", {
  set.seed(0)
  n_strata <- 6L
  n_per <- 4L
  n <- n_strata * n_per
  data <- data.frame(
    count = rpois(n, lambda = 2),
    hum = rnorm(n),
    region = rep(1:2, length.out = n),
    date = rep(seq_len(n_strata), each = n_per),
    year = 2002L
  )
  formula <- dirichlet_multinom(
    count,
    by = c("year", "region", "date")
  ) ~ hum + iid(date)

  md <- model_data(
    formula = formula,
    data = data,
    na_omit = TRUE,
    verbose = FALSE
  )
  expect_gt(ncol(md$term_data$elgm_matrix), 0L)
  expect_gt(ncol(md$observations$count@elgm_matrix), 0L)
  expect_identical(md$parameters$count_extra@density, "dirichlet_multinomial_extra")
  expect_identical(md$observations$count@package, "adlaplace")
})

test_that("dirichlet compact_tape matches full-domain with elgm obs_groups", {
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
  elgm <- md$term_data$elgm_matrix
  og <- obs_groups(
    md$term_data$A,
    elgm_matrix = elgm,
    num_shards = 4L
  )
  expect_lt(max(og@i), ncol(elgm))

  build_pack <- function(compact) {
    ad_pack(
      md,
      config = list(
        transform_theta = TRUE,
        obs_groups = og,
        compact_tape = compact,
        verbose = FALSE
      ),
      num_threads = 1L,
      reorder_shards = "none"
    )
  }

  pack_c <- build_pack(TRUE)
  pack_f <- build_pack(FALSE)
  x <- c(
    as.numeric(md$term_data$info$beta$init),
    rep(0, nrow(md$term_data$info$gamma)),
    as.numeric(md$term_data$info$theta$init)
  )
  # theta may be on log scale in info; use pack sizes / config from dens eval
  n_beta <- as.integer(pack_c@sizes[["beta"]])
  n_gamma <- as.integer(pack_c@sizes[["gamma"]])
  n_theta <- as.integer(pack_c@sizes[["theta"]])
  x <- c(rep(0.1, n_beta), rnorm(n_gamma, sd = 0.05), rep(log(0.1), n_theta))

  expect_equal(
    joint_log_dens(pack_c@ptr, x, negative = FALSE),
    joint_log_dens(pack_f@ptr, x, negative = FALSE),
    tolerance = 1e-8
  )
  expect_equal(
    as.numeric(grad(pack_c@ptr, x, negative = FALSE)),
    as.numeric(grad(pack_f@ptr, x, negative = FALSE)),
    tolerance = 1e-6
  )
})

test_that("ad_pack builds elgm obs_groups from num_shards when missing", {
  set.seed(3)
  n_strata <- 8L
  n_per <- 4L
  n <- n_strata * n_per
  data <- data.frame(
    count = rpois(n, lambda = 2),
    hum = rnorm(n),
    region = rep(1:2, length.out = n),
    date = rep(seq_len(n_strata), each = n_per),
    year = 2002L
  )
  formula <- dirichlet_multinom(
    count,
    by = c("year", "region", "date"),
    init = 0.1
  ) ~ hum + iid(date)
  md <- model_data(formula = formula, data = data, verbose = FALSE)
  n_strata_keep <- ncol(md$term_data$elgm_matrix)

  pack <- ad_pack(
    md,
    config = list(
      transform_theta = TRUE,
      num_shards = 3L,
      verbose = FALSE
    ),
    num_threads = 1L,
    reorder_shards = "none"
  )
  expect_gt(adlaplace:::n_groups(pack@ptr), 1L)

  # Shard without ensure_config_obs_groups must not use length(y) as strata.
  obs <- md$observations$count
  cfg <- list(
    beta = rep(0, nrow(obs@beta_map)),
    gamma = rep(0, nrow(obs@gamma_map)),
    theta = rep(log(0.1), nrow(obs@theta_map)),
    transform_theta = TRUE,
    verbose = FALSE
  )
  ptr <- ad_pack_ptr(obs, cfg)
  expect_equal(adlaplace:::n_groups(ptr), 1L)
  # One-group default over strata: dens must finish without OOB.
  x <- c(cfg$beta, cfg$gamma, cfg$theta)
  expect_true(is.finite(joint_log_dens(ptr, x, negative = FALSE)))
  expect_equal(ncol(obs@elgm_matrix), n_strata_keep)
})

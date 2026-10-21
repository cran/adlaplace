test_that("term_data_setup returns empty beta info when no fixed terms", {
  dat <- data.frame(
    y = rpois(20L, lambda = 1),
    f = factor(rep(1:4, each = 5L))
  )
  terms <- adlaplace::collect_terms(
    stats::update.formula(adlaplace::nbinom(y) ~ adlaplace::iid(f), . ~ . - 1)
  )
  setup <- adlaplace::term_data_setup(terms, dat)

  expect_s3_class(setup$info$beta, "data.frame")
  expect_equal(nrow(setup$info$beta), 0L)
  expect_equal(length(setup$info$beta$init), 0L)
})

test_that("ad_pack(model_data) works with zero betas", {
  dat <- data.frame(
    y = rpois(20L, lambda = 1),
    f = factor(rep(1:4, each = 5L))
  )
  md <- adlaplace::model_data(
    stats::update.formula(adlaplace::nbinom(y) ~ adlaplace::iid(f), . ~ . - 1),
    data = dat
  )

  expect_equal(nrow(md$term_data$info$beta), 0L)
  expect_equal(length(md$term_data$info$beta$init), 0L)

  config <- list(
    transform_theta = TRUE,
    obs_groups = adlaplace::obs_groups(md$term_data$A, num_shards = 2L),
    verbose = FALSE
  )
  af <- adlaplace::ad_pack(md, config, num_threads = 1L)

  expect_equal(as.integer(af@sizes["beta"]), 0L)
  n_gamma <- as.integer(af@sizes["gamma"])
  n_theta <- as.integer(af@sizes["theta"])
  expect_gt(n_gamma, 0L)
  expect_gt(n_theta, 0L)

  x <- c(
    rep(0, n_gamma),
    md$term_data$info$theta$init
  )
  expect_equal(length(x), n_gamma + n_theta)

  ld <- adlaplace::joint_log_dens(af, x, negative = FALSE)
  expect_true(is.finite(ld))

  g <- adlaplace::grad(af, x)
  expect_equal(length(g), length(x))
})

test_that("ad_pack_ptr accepts omitted config$beta when n_beta is zero", {
  nr <- 4L
  model <- adlaplace:::density_data(
    gamma_map = Matrix::sparseMatrix(
      i = seq_len(nr),
      j = seq_len(nr),
      x = 1,
      dims = c(6L, nr)
    ),
    theta_map = c(1L, 1L),
    ad_kind = "random",
    density = "random_diagonal",
    precision = rep(2, nr)
  )
  config <- list(
    gamma = rep(0, 6),
    theta = 0.1,
    transform_theta = FALSE
  )
  ptr <- adlaplace::ad_pack_ptr(model, config)
  expect_true(is(ptr, "ad_pack_ptr"))
})

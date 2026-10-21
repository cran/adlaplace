test_that("Dirichlet-multinomial log density matches extraDistr ddirmnom", {
  skip_if_not_installed("extraDistr")
  ddirmnom <- utils::getFromNamespace("ddirmnom", "extraDistr")

  set.seed(1)
  K <- 6
  N <- 100 * K
  mu <- 2
  tau <- 0.2
  mu_vec <- rep(mu, K)

  U <- stats::rgamma(N, shape = tau^(-2), rate = tau^(-2))
  y <- stats::rpois(N, mu * U)

  cc_matrix <- Matrix::sparseMatrix(
    i = seq_len(N),
    j = rep(seq_len(N / K), each = K),
    x = 1
  )

  alpha_norm <- mu_vec / sum(mu_vec)
  alpha <- alpha_norm / tau^2

  for (D in seq_len(ncol(cc_matrix))) {
    y_here <- y[which(cc_matrix[, D] == 1)]

    y_double_sum <- mapply(
      seq,
      len = y_here,
      from = alpha_norm,
      MoreArgs = list(by = tau^2)
    )

    log_dens_simplified <-
      lfactorial(sum(y_here)) -
      sum(lfactorial(y_here)) -
      sum(log(seq(from = 1, by = tau^2, len = sum(y_here)))) +
      sum(log(unlist(y_double_sum)))

    log_dens_with_gammas <-
      lgamma(sum(alpha)) + lgamma(sum(y_here) + 1) -
      lgamma(sum(y_here) + sum(alpha)) +
      sum(lgamma(y_here + alpha)) -
      sum(lgamma(alpha)) - sum(lgamma(y_here + 1))

    log_dens_package <- ddirmnom(
      x = matrix(y_here, nrow = 1, ncol = length(y_here)),
      size = sum(y_here),
      alpha = matrix(alpha, nrow = 1, ncol = length(y_here)),
      log = TRUE
    )

    expect_equal(log_dens_simplified, log_dens_with_gammas, tolerance = 1e-8)
    expect_equal(log_dens_simplified, as.numeric(log_dens_package), tolerance = 1e-8)
  }
})

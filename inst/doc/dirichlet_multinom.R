## ----setup, include=FALSE-----------------------------------------------------
if (file.exists("vignette_full.R")) {
  source("vignette_full.R", local = TRUE)
} else {
  vignette_full <- TRUE
  vignette_num_threads <- 2L
}

## ----testDirMn----------------------------------------------------------------
K <- 6
N <- 100 * K
mu <- 2
tau <- 0.2
mu_vec <- rep(mu, K)
alpha <- mu_vec / sum(mu_vec)

U <- rgamma(N, shape = tau^(-2), rate = tau^(-2))
c(mean(U), sd(U))
lambda <- mu * U
y <- rpois(N, lambda)

cc_matrix <- Matrix::sparseMatrix(
  i = 1:N, j = rep(seq(1, N / K), each = K), x = 1
)

mu_vec <- rep(mu, K)
alpha_norm <- mu_vec / sum(mu_vec)
alpha <- alpha_norm / tau^2

log_dens_with_gammas <- log_dens_simplified <- rep(NA, ncol(cc_matrix))

for (D in 1:ncol(cc_matrix)) {
  # Extract the indices where the column has non-zero values
  indices_here <- which(cc_matrix[, D] == 1)
  y_here <- y[indices_here]

  # Simplified log density using the sum formula
  # Handle zeros correctly by filtering them out for the sum
  y_double_sum <- mapply(seq,
    len = y_here,
    from = alpha_norm,
    MoreArgs = list(by = tau^2)
  )

  log_dens_simplified[D] <-
    lfactorial(sum(y_here)) -
    sum(lfactorial(y_here)) -
    sum(log(seq(from = 1, by = tau^2, len = sum(y_here)))) +
    sum(log(unlist(y_double_sum)))

  # Log density using Gamma functions
  # lgamma is defined for all non-negative integers, so no special handling needed for zeros
  log_dens_with_gammas[D] <-
    lgamma(sum(alpha)) + lgamma(sum(y_here) + 1) -
    lgamma(sum(y_here) + sum(alpha)) +
    sum(lgamma(y_here + alpha)) -
    sum(lgamma(alpha)) - sum(lgamma(y_here + 1))
}

## ----plot---------------------------------------------------------------------
plot(log_dens_simplified, log_dens_with_gammas,
  xlab = "simplified", ylab = "gammas"
)
abline(0, 1)


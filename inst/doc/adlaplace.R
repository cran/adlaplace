## ----setup, include=FALSE-----------------------------------------------------
library("Matrix")
library("adlaplace")
if (file.exists("vignette_full.R")) {
  source("vignette_full.R", local = TRUE)
} else {
  vignette_full <- TRUE
  vignette_num_threads <- 2L
}

## ----theData------------------------------------------------------------------
Nobs <- if (vignette_full) 5000 else 400
Nrandom1 <- 10
Nrandom2 <- 25

set.seed(0)

X <- Matrix::Matrix(cbind(1, rbinom(Nobs, 1, prob = 0.5)))

Adf <- data.frame(
  r1 = sample(Nrandom1, Nobs, replace = TRUE),
  r2 = sample(Nrandom2, Nobs, replace = TRUE)
)
AmatList <- list(
  r1 = Matrix::sparseMatrix(
    i = seq_len(Nobs),
    j = Adf$r1
  ),
  r2 = Matrix::sparseMatrix(
    i = seq_len(Nobs),
    j = Adf$r2
  )
)
Amat <- do.call(cbind, AmatList)

beta <- rep(1, ncol(X))
thetaOrig <- c(0.1, 0.1, 0.2) # SD for each random group, then overdispersion
theta <- log(thetaOrig)

gamma <- rnorm(
  ncol(Amat),
  sd = rep(thetaOrig[1:2], c(Nrandom1, Nrandom2))
)
eta_true <- as.vector(X %*% beta + Amat %*% gamma)
mu_true <- exp(eta_true)
Z <- rgamma(length(mu_true), thetaOrig[3]^(-2), thetaOrig[3]^(-2))
y <- rpois(length(mu_true), mu_true * Z)

config <- list(
  beta = beta,
  theta = log(thetaOrig),
  transform_theta = TRUE,
  # Warm-start near the data-generating values (close to the MLE).
  gamma = gamma,
  obs_groups = adlaplace::obs_groups(
    Amat,
    num_shards = if (vignette_full) 50L else 8L
  ),
  verbose = FALSE,
  package = "adlaplace"
)

n_beta <- length(config$beta)
n_gamma <- length(config$gamma)
n_theta <- length(config$theta)

## ----adShards_obs-------------------------------------------------------------
data_obs <- adlaplace:::density_data(
  y = y,
  A = Amat,
  X = X,
  theta_map = Matrix::sparseMatrix(
    i = n_theta,
    j = 1L,
    dims = c(n_theta, 1L)
  ),
  ad_kind = "observations",
  density = "nbinom_obs"
)

data_extra <- adlaplace::density_data(
  data_obs,
  ad_kind = "parameters",
  density = "nbinom_extra"
)

ad_fun_obs <- adlaplace::ad_pack_ptr(data = data_obs, config = config)
ad_fun_extra <- adlaplace::ad_pack_ptr(data = data_extra, config = config)

## ----adShards_random----------------------------------------------------------
model_r1 <- adlaplace:::density_data(
  beta_map = n_beta,
  gamma_map = Matrix::sparseMatrix(
    i = seq_len(Nrandom1),
    j = seq_len(Nrandom1),
    dims = c(n_gamma, Nrandom1)
  ),
  theta_map = c(1L, n_theta),
  ad_kind = "random",
  density = "random_diagonal",
  precision = rep(1, Nrandom1)
)
ad_fun_r1 <- adlaplace::ad_pack_ptr(data = model_r1, config = config)

model_r2 <- adlaplace:::density_data(
  beta_map = n_beta,
  gamma_map = Matrix::sparseMatrix(
    i = seq.int(Nrandom1 + 1L, length.out = Nrandom2),
    j = seq_len(Nrandom2),
    dims = c(n_gamma, Nrandom2)
  ),
  theta_map = c(2L, n_theta),
  ad_kind = "random",
  density = "random_diagonal",
  precision = rep(1, Nrandom2)
)
ad_fun_r2 <- adlaplace::ad_pack_ptr(data = model_r2, config = config)

## ----obsLogDens---------------------------------------------------------------
xx <- c(beta, gamma, config$theta)
log_theta_true <- config$theta[n_theta]

nb_log_dens <- function(log_theta) {
  r <- exp(-2 * log_theta)
  sum(stats::dnbinom(y, mu = mu_true, size = r, log = TRUE))
}

ad_nb_log_dens <- function(log_theta) {
  xx_scan <- xx
  xx_scan[length(xx_scan)] <- log_theta
  adlaplace::joint_log_dens(ad_fun_obs, xx_scan, negative = FALSE) +
    adlaplace::joint_log_dens(ad_fun_extra, xx_scan, negative = FALSE)
}

seq_log_theta <- log_theta_true + seq(-0.5, 0.5, length.out = 5)
cmp <- data.frame(
  theta = exp(seq_log_theta),
  stats = vapply(seq_log_theta, nb_log_dens, numeric(1)),
  ad = vapply(seq_log_theta, ad_nb_log_dens, numeric(1))
)
cmp$diff <- cmp$ad - cmp$stats
cmp

## ----adShards_combined--------------------------------------------------------
ad_fun_plain <- c(
  ad_fun_obs, ad_fun_extra,
  ad_fun_r1, ad_fun_r2
)
ad_pack <- adlaplace::ad_pack(
  adlaplace::clone_ad_pack_ptr(ad_fun_plain),
  num_threads = vignette_num_threads
)

## ----checkByShard-------------------------------------------------------------
x_full <- c(config$beta, config$gamma, config$theta)
shards <- seq.int(from = 0L, length.out = adlaplace:::n_groups(ad_fun_plain))

by_shard <- vapply(
  shards,
  function(s) {
    adlaplace::joint_log_dens(
      ad_fun_plain, x_full,
      ad_shards = s, negative = TRUE
    )
  },
  numeric(1)
)

res_fdf <- fun_obj_fdfh(
  ad_pack,
  c(config$beta, config$theta), config$gamma,
  inner = FALSE, verbose = TRUE
)

by_shard[seq(1, min(c(5, length(by_shard))))]
c(
  sum(by_shard),
  adlaplace::joint_log_dens(ad_fun_plain, x_full, negative = TRUE),
  res_fdf$f
)

## ----testLogDens--------------------------------------------------------------
log_all <- adlaplace::joint_log_dens(ad_fun_plain, x_full, negative = TRUE)
log_sub <- adlaplace::joint_log_dens(
  ad_fun_plain, x_full,
  ad_shards = 0:3, negative = TRUE
)

grad_all <- adlaplace::grad(ad_fun_plain, x_full, negative = TRUE)
grad_sub <- adlaplace::grad(
  ad_fun_plain, x_full,
  ad_shards = 0:3, negative = TRUE
)

hess_all <- adlaplace::hessian(ad_fun_plain, x_full, negative = TRUE)
hess_sub <- adlaplace::hessian(
  ad_fun_plain, x_full,
  ad_shards = 0:3, negative = TRUE
)

c(
  log_diff = log_all - res_fdf$f,
  max_abs_grad_diff = max(abs(grad_all - res_fdf$grad)),
  max_abs_hess_diff = max(abs(
    as.matrix(hess_all) - as.matrix(res_fdf$hessian)
  ))
)

## ----testInnerOpt-------------------------------------------------------------
x_outer <- c(config$beta, config$theta)
x_full <- c(config$beta, config$gamma, config$theta)


inner_res <- adlaplace::inner_opt(
  parameters = x_outer,
  gamma = config$gamma,
  ad_pack = ad_pack,
  control = list(
    maxit = 20L,
    report.level = 0,
    report.freq = 0,
    grad.tol = 1e-4,
    cg.tol = 1e-4,
    restart.gamma.clamp = Inf
  ),
  deriv = TRUE,
  verbose = FALSE
)
adlaplace::joint_log_dens(ad_fun_plain, x_full, negative = TRUE)


inner_res$inner_opt$fval

plot(
  inner_res$inner_opt$solution, gamma,
  xlab = "estimated", ylab = "true"
)
abline(0, 1, col = "red", lty = 2)

## ----testLogLik---------------------------------------------------------------
res_noderiv <- adlaplace::log_lik_laplace(
  x = x_outer,
  ad_pack = ad_pack,
  config = config,
  deriv = FALSE
)

res_deriv <- adlaplace::log_lik_laplace(
  x = x_outer,
  ad_pack = ad_pack,
  config = config,
  deriv = TRUE
)

data.frame(
  log_lik = c(res_noderiv$log_lik, res_deriv$log_lik),
  neg_log_lik = c(res_noderiv$neg_log_lik, res_deriv$neg_log_lik),
  inner_fval = c(res_deriv$inner_opt$fval, res_deriv$inner_opt$fval),
  row.names = c("deriv = FALSE", "deriv = TRUE")
)

summary(res_deriv$gradient$outer$trace3)
res_deriv$deriv$d_neg_log_lik

## ----trustOptimOuterWrappers--------------------------------------------------
cache <- new.env(parent = emptyenv())
cache$gamma <- config$gamma

control_inner <- list(
  maxit = 30L,
  report.level = 0,
  report.freq = 0,
  restart.gamma.clamp = Inf
)

# Start near the truth (approx. MLE); few outer steps suffice.
x_start <- x_outer + c(0.05, -0.05, 0.1, -0.1, 0.05)
optim_control <- list(
  maxit = if (vignette_full) 25L else 5L,
  trace = 0, factr = 1e9, pgtol = 1e-3
)
bounds <- list(lower = rep(-5, length(x_outer)), upper = rep(5, length(x_outer)))

common_optim_args <- c(
  list(
    par = x_start,
    fn = adlaplace::outer_fn,
    method = "L-BFGS-B",
    control = optim_control,
    config = config,
    ad_pack = ad_pack,
    cache = cache,
    control_inner = control_inner
  ),
  bounds
)

outer_fit <- do.call(stats::optim, c(common_optim_args, list(gr = adlaplace::outer_gr)))

fit_cmp <- rbind(
  optim = outer_fit$par,
  start = x_start,
  true = c(beta, log(thetaOrig))
)
colnames(fit_cmp) <- c(
  paste0("beta[", seq_len(n_beta), "]"),
  paste0("log_theta[", seq_len(n_theta), "]")
)
fit_cmp
exp(outer_fit$par[n_beta + seq_len(n_theta)])
outer_fit$value

## ----testLogLgrad, fig.height=7, eval = vignette_full-------------------------
config$verbose <- FALSE
oldpar <- par(no.readonly = TRUE)
par(mfrow = c(3, 2), mar = c(2, 2, 2, 0), mgp = c(1, 0.5, 0))

x1 <- outer_fit$par
Dpar <- length(x1)
scan_ll <- deriv_scan_laplace(
  ad_pack,
  x = x1,
  index = Dpar,
  width = 0.2,
  n = 5L,
  config = config,
  gamma = cache$gamma,
  control = control_inner
)
res_mid <- scan_ll$mid_result

plot(scan_ll$grid, scan_ll$value, type = "l",
     xlab = expression(theta[3]), ylab = "-log lik")
plot(scan_ll$grid, scan_ll$grad[, Dpar], type = "l",
     xlab = expression(theta[3]), ylab = "grad")
points(scan_ll$mid, scan_ll$fd_value, pch = 16)
legend("topright", legend = c("AD", "finite diff"),
       lty = c(1, NA), pch = c(NA, 16), bty = "n")

Du <- 2L
plot(scan_ll$grid, scan_ll$u_hat[, Du], type = "l",
     xlab = expression(theta[3]), ylab = expression(u[Du]))
plot(scan_ll$grid, scan_ll$dU_index[, Du], type = "l",
     xlab = expression(theta[3]),
     ylab = expression(d * u[Du] / d * theta[3]))
points(scan_ll$mid, scan_ll$fd_u[, Du], pch = 16)

plot(scan_ll$grid, scan_ll$half_log_det, type = "l",
     xlab = expression(theta[3]), ylab = "half log det")
plot(scan_ll$grid, scan_ll$d_det[, Dpar], type = "l",
     xlab = expression(theta[3]),
     ylab = expression(d / d * theta[3] ~ log ~ det))
points(scan_ll$mid, scan_ll$fd_det, pch = 16)

hessian_outer <- Matrix::forceSymmetric(res_mid$hessian$outer)
hessian_plain <- adlaplace::hessian(
  ad_fun_plain,
  res_mid$full_parameters,
  inner = FALSE,
  negative = TRUE
)
max_abs_hess_diff <- max(abs((hessian_outer - hessian_plain)@x), na.rm = TRUE)
max_abs_hess_diff
par(oldpar)

## ----checkDu, eval = vignette_full--------------------------------------------
seq_gamma1 <- seq.int(
  ad_pack@sizes["beta"] + 1L,
  length.out = ad_pack@sizes["gamma"]
)
dU_check <- -solve(
  res_mid$hessian$inner,
  res_mid$hessian$outer[seq_gamma1, -seq_gamma1]
)
dU_check2 <- -
  res_mid$hessian$H_inv %*%
  res_mid$hessian$outer[seq_gamma1, -seq_gamma1]

quantile(abs(dU_check - res_mid$gradient$outer$dU))
quantile(abs(dU_check - dU_check2))
quantile(abs(dU_check2 - res_mid$gradient$outer$dU))

oldpar <- par(no.readonly = TRUE)
par(mfrow = c(1, 1))
plot(
  scan_ll$grid, scan_ll$dU_index[, Du],
  type = "l",
  xlab = expression(theta[3]),
  ylab = expression(d * u[Du] / d * theta[3])
)
points(scan_ll$mid, scan_ll$fd_u[, Du], pch = 16)
points(
  rep(res_mid$parameters[Dpar], 2),
  c(dU_check[Du, Dpar], dU_check2[Du, Dpar]),
  col = c("red", "blue"), cex = c(5, 3)
)
par(oldpar)

## ----derivJointDens, fig.height=5, eval = vignette_full-----------------------
oldpar <- par(no.readonly = TRUE)
par(mfrow = c(4, 2), mar = c(2, 2, 2, 0), mgp = c(1, 0.5, 0))

x_here <- res_mid$full_parameters
Dpar_dens <- length(x_here) - 1L
shards <- seq.int(from = 0L, length.out = adlaplace:::n_groups(ad_fun_plain))

scan_j <- deriv_scan_joint(
  ad_fun_plain,
  x = x_here,
  index = Dpar_dens,
  width = 0.1,
  n = 5L,
  ad_shards = shards,
  negative = FALSE
)

plot(scan_j$grid, scan_j$grad[, Dpar_dens], type = "l", ylab = "AD gradient")
points(scan_j$mid, scan_j$fd_value, pch = 16)
legend("topright", legend = c("AD", "finite diff"),
       lty = c(1, NA), pch = c(NA, 16), bty = "n")

s_par2 <- sort(unique(c(Dpar_dens, 1:3, seq(to = length(x_here), length.out = 5))))
s_par2 <- s_par2[seq(1, min(c(prod(par("mfrow")) - 1L, length(s_par2))))]
for (Dpar2 in s_par2) {
  plot(
    scan_j$grid, scan_j$hess_row[, Dpar2],
    type = "l",
    ylab = paste0("H[", Dpar_dens, ",", Dpar2, "]"),
    ylim = range(c(scan_j$hess_row[, Dpar2], scan_j$fd_grad[, Dpar2]), na.rm = TRUE)
  )
  points(scan_j$mid, scan_j$fd_grad[, Dpar2], pch = 16)
}
par(oldpar)

## ----checkNicerInterface, eval = vignette_full--------------------------------
dat <- cbind(
  data.frame(y = y),
  Adf,
  x = X[, 2]
)
md <- model_data(
  data = dat,
  formula =
    nbinom(y, lower = 1e-9, init = 0.2) ~
      x + iid(r1, init = 0.1) + iid(r2, init = 0.1)
)
config2 <- list(
  transform_theta = TRUE,
  obs_groups = obs_groups(
    md$term_data$A,
    num_shards = 50L
  )
)
ad_fun2 <- ad_pack(md, config2, num_threads = vignette_num_threads)
# Serial dens for mean_mvquad: clone clears multi-thread affinity, then
# ad_pack(..., num_threads = 1) rebuilds templates on a dens-safe handle.
ad_fun2_dens <- ad_pack(clone_ad_pack_ptr(ad_fun2@ptr), num_threads = 1L)

md$observations$y@XTp[, 1:5]
data_obs@XTp[, 1:5]

md$term_data$info$parameters[, c("term", "model", "label", "init", "log")]
data_obs@theta_map

param_remap <- c(2, 1, 5, 3, 4)
log_lik2 <- log_lik_laplace(
  x = res_deriv$parameters[param_remap],
  config = config2,
  ad_pack = ad_fun2,
  deriv = TRUE
)
c(log_lik2$log_lik, res_deriv$log_lik)
rbind(log_lik2$deriv$d_neg_log_lik, res_deriv$deriv$d_neg_log_lik[param_remap])

## ----meanEstimation, eval = vignette_full-------------------------------------
library(mvQuad)
mean_estimate <- mean_mvquad(
  parameters = log_lik2$parameters,
  mode = log_lik2$inner_opt$solution,
  cov = log_lik2$hessian$H_inv,
  ad_pack = ad_fun2_dens,
  n = 3
)
quantile(mean_estimate - log_lik2$inner_opt$solution)
plot(log_lik2$inner_opt$solution, mean_estimate)
abline(0, 1, col = "red", lty = 2)


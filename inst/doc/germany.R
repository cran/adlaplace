## ----setup, include=FALSE-----------------------------------------------------
library("Matrix")
library("adlaplace")
if (file.exists("vignette_full.R")) {
  source("vignette_full.R", local = TRUE)
} else {
  vignette_full <- TRUE
  vignette_num_threads <- 2L
}
data("germany")
# District polygons live in inst/extdata (not in data/germany.rda) so loading
# the dataset does not require terra.
if (requireNamespace("terra", quietly = TRUE)) {
  germany$map <- terra::vect(
    system.file("extdata", "germany_map.gpkg", package = "adlaplace")
  )
}

## ----shards-------------------------------------------------------------------
n <- length(germany$Y)
n_theta <- 2L
n_gamma <- 2 * n

A_obs <- cbind(
  Matrix::sparseMatrix(i = seq_len(n), j = seq_len(n), x = 1, dims = c(n, n)),
  Matrix::sparseMatrix(i = seq_len(n), j = seq_len(n), x = 1, dims = c(n, n))
)
X_obs <- Matrix::Matrix(0, n, 0)

obs_term <- pois("Y")

config <- list(
  transform_theta = TRUE,
  gamma = rep(0, n_gamma),
  theta = log(c(1e-2, 0.1)),
  offset = log(germany$E),
  obs_groups = adlaplace::obs_groups(A_obs, num_shards = 2L),
  verbose = FALSE
)

data_obs <- adlaplace::density_data(
  y = germany$Y,
  A = A_obs,
  X = X_obs,
  theta_map = n_theta,
  ad_kind = obs_term@ad_kind,
  density = obs_term@density
)

model_struct <- adlaplace::density_data(
  gamma_map = Matrix::sparseMatrix(
    i = seq_len(n), j = seq_len(n), x = 1,
    dims = c(n_gamma, n)
  ),
  theta_map = c(1L, n_theta),
  ad_kind = "random",
  density = "random_mult",
  precision = germany$prec
)


model_iid <- adlaplace::density_data(
  gamma_map = Matrix::sparseMatrix(
    i = seq.int(n + 1L, length.out = n), j = seq_len(n), x = 1,
    dims = c(n_gamma, n)
  ),
  theta_map = c(2L, n_theta),
  ad_kind = "random",
  density = "random_diagonal",
  precision = rep(1, n)
)

## ----ad_pack------------------------------------------------------------------
ptr_obs <- adlaplace::ad_pack_ptr(data = data_obs, config = config)
ptr_iid <- adlaplace::ad_pack_ptr(data = model_iid, config = config)
ptr_struct <- adlaplace::ad_pack_ptr(data = model_struct, config = config)

ad_fun_plain <- c(ptr_obs, ptr_struct, ptr_iid)
ad_pack <- adlaplace::ad_pack(
  adlaplace::clone_ad_pack_ptr(ad_fun_plain),
  num_threads = vignette_num_threads
)

## ----checkByShard-------------------------------------------------------------
x_full <- c(config$gamma, config$theta)
shards <- seq.int(from = 0L, length.out = adlaplace::n_groups(ad_fun_plain))

by_shard <- vapply(
  shards,
  function(s) {
    adlaplace::joint_log_dens(ad_fun_plain, x_full, ad_shards = s, negative = TRUE)
  },
  numeric(1)
)

res_fdf <- adlaplace::fun_obj_fdfh(
  ad_pack,
  config$theta, config$gamma,
  inner = FALSE, verbose = FALSE
)

c(
  sum(by_shard),
  adlaplace::joint_log_dens(ad_fun_plain, x_full, negative = TRUE),
  res_fdf$f
)

## ----testLogDens--------------------------------------------------------------
log_all <- adlaplace::joint_log_dens(ad_fun_plain, x_full, negative = TRUE)
log_sub <- adlaplace::joint_log_dens(
  ad_fun_plain, x_full,
  ad_shards = 0:2, negative = TRUE
)

grad_all <- adlaplace::grad(ad_fun_plain, x_full, negative = TRUE)
grad_sub <- adlaplace::grad(ad_fun_plain, x_full, ad_shards = 0:2, negative = TRUE)

hess_all <- adlaplace::hessian(ad_fun_plain, x_full, negative = TRUE)
hess_sub <- adlaplace::hessian(ad_fun_plain, x_full, ad_shards = 0:2, negative = TRUE)

rbind(
  log_diff = log_all - res_fdf$f,
  max_abs_grad_diff = max(abs(grad_all - res_fdf$grad)),
  max_abs_hess_diff = max(abs(as.matrix(hess_all) - as.matrix(res_fdf$hessian))),
  log_sub_diff = log_all - log_sub,
  max_abs_grad_sub_diff = max(abs(grad_all - grad_sub)),
  max_abs_hess_sub_diff = max(abs(as.matrix(hess_all) - as.matrix(hess_sub)))
)

## ----testInnerOpt-------------------------------------------------------------
x_outer <- config$theta

inner_res <- adlaplace::inner_opt(
  parameters = log(c(1e-1, 1e-1)),
  gamma = config$gamma,
  ad_pack = ad_pack,
  control = list(
    maxit = if (vignette_full) 1000L else 40L,
    report.level = if (vignette_full) 4 else 0,
    report.freq = if (vignette_full) 5 else 0,
    grad.tol = 1e-8,
    cg.tol = 1e-6
  ),
  deriv = vignette_full,
  verbose = FALSE
)

inner_res$inner_opt$fval

if (vignette_full) {
  summary(inner_res$hessian$trace3)
}

## ----derivJointDens, fig.height=5, eval = vignette_full-----------------------
# Finite-difference points should fall on the AD lines.
oldpar <- par(no.readonly = TRUE)
par(mfrow = c(4, 2), mar = c(2, 2, 2, 0), mgp = c(1, 0.5, 0))

x_here <- inner_res$full_parameters
Dpar_dens <- 96L
shards <- seq.int(from = 0L, length.out = adlaplace::n_groups(ad_fun_plain))

scan_j <- deriv_scan_joint(
  ad_fun_plain,
  x = x_here,
  index = Dpar_dens,
  width = 0.1,
  n = 11L,
  ad_shards = shards,
  negative = FALSE
)

plot(scan_j$grid, scan_j$grad[, Dpar_dens], type = "l", ylab = "AD gradient")
points(scan_j$mid, scan_j$fd_value, pch = 16)
legend("topright", legend = c("AD", "finite diff"),
       lty = c(1, NA), pch = c(NA, 16), bty = "n")

s_par2 <- sort(unique(c(
  Dpar_dens + 0:5,
  Dpar_dens + length(germany$Y),
  seq(to = length(x_here), length.out = 5)
)))
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

res_deriv$deriv$d_neg_log_lik

## ----optim--------------------------------------------------------------------
cache <- new.env(parent = emptyenv())
cache$gamma <- rep(0, n_gamma)

control_inner <- list(
  maxit = 100L,
  report.level = 0,
  report.freq = 0
)

opt <- optim(
  par = config$theta,
  fn = adlaplace::outer_fn,
  gr = adlaplace::outer_gr,
  method = "L-BFGS-B",
  lower = rep(-15, n_theta),
  control = list(
    maxit = if (vignette_full) 200L else 8L,
    factr = 1e3,
    trace = if (vignette_full) 1 else 0,
    REPORT = 1
  ),
  config = config,
  ad_pack = ad_pack,
  cache = cache,
  control_inner = control_inner
)

res <- adlaplace::log_lik_laplace(
  x = opt$par,
  gamma = cache$gamma,
  ad_pack = ad_pack,
  config = config,
  control = control_inner,
  deriv = FALSE
)

u_hat <- res$inner_opt$solution
eta_hat <- u_hat[seq_len(n)] + u_hat[seq.int(n + 1L, length.out = n)]
sd_hat <- exp(opt$par)
data.frame(
  component = c("SD structured (ICAR)", "SD unstructured (iid)"),
  estimate = sd_hat
)

## ----maps, fig.height=4-------------------------------------------------------
if (!is.null(germany$map)) {
  germany$map$eta_hat <- eta_hat
  terra::plot(
    germany$map, "eta_hat",
    main = expression(hat(eta) ~ "(adlaplace mode)")
  )
} else {
  oldpar <- par(no.readonly = TRUE)
  par(mfrow = c(1, 1), mar = c(4, 4, 2, 1), mgp = c(2.2, 0.7, 0))
  plot(
    seq_len(n), eta_hat,
    type = "h", xlab = "district", ylab = expression(hat(eta)),
    main = "Total spatial effect (adlaplace mode; install terra for map)"
  )
  par(oldpar)
}


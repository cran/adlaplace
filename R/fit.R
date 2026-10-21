#' @include log_lik.R model_data.R obs_groups.R
NULL

#' Fit a hierarchical model by Laplace-approximate maximum likelihood
#'
#' One-call interface to the \pkg{adlaplace} pipeline: parse the formula with
#' \code{\link{model_data}()}, partition observations into AD shards with
#' \code{\link{obs_groups}()}, build the AD handle with \code{\link{ad_pack}()},
#' maximize the Laplace-approximate marginal likelihood with
#' \code{\link[stats]{optim}()} using the exact profile gradient
#' (\code{\link{outer_fn}} / \code{\link{outer_gr}}), and return a fitted-model
#' object with standard accessor methods.
#'
#' @param formula Model formula built from model terms, e.g.
#'   \code{nbinom(y) ~ x + iid(g)}, a list of term objects from
#'   \code{\link{collect_terms}}, or a precomputed \code{\link{model_data}}
#'   result (in which case \code{data} is ignored).
#' @param data Data frame containing the variables referenced in \code{formula}.
#'   Not needed when \code{formula} is already a \code{model_data} result.
#' @param config List of backend options merged over the defaults
#'   \code{list(transform_theta = TRUE, verbose = verbose, num_threads = 1L,
#'   num_shards = 100L)}. When \code{config$obs_groups} is missing it is filled
#'   by \code{obs_groups(A, num_shards = config$num_shards)}, or with
#'   \code{elgm_matrix} when the model data includes one (e.g. case-crossover).
#'   \code{config$num_threads} is the OpenMP thread count for the AD handle;
#'   \code{config$num_shards} is the target number of observation shards for
#'   \code{\link{obs_groups}()}.
#' @param control List of \code{\link[stats]{optim}} control options, merged
#'   over the defaults \code{list(maxit = 200L, parscale = <from terms>)}.
#' @param control_inner List of control options for the inner (random effects)
#'   trust-region optimizer.
#' @param method \code{\link[stats]{optim}} method; bounds are used for
#'   \code{"L-BFGS-B"} and \code{"Brent"} only.
#' @param hessian Logical. When \code{NULL} (default), set to \code{TRUE} if
#'   there are at most five outer parameters and \code{FALSE} otherwise.
#'   When \code{TRUE}, \code{\link[stats]{optimHess}} numerically differentiates
#'   the exact AD profile gradient at the optimum, giving the observed
#'   information used by \code{\link{vcov.adlaplace_fit}},
#'   \code{\link{summary.adlaplace_fit}}, and \code{\link{confint.adlaplace_fit}}.
#' @param start Optional numeric vector of outer starting values
#'   \code{c(beta, theta)} on the \emph{internal} (log-theta) scale, length
#'   \code{nrow(par_info)}. When \code{NULL} (default), term \code{init} values
#'   are used.
#' @param verbose Logical, print progress.
#' @param na_omit Passed to \code{\link{model_data}()}.
#'
#' @return An object of class \code{"adlaplace_fit"}: a list with components
#' \describe{
#'   \item{call, formula}{The matched call and model formula.}
#'   \item{par_info}{Parameter table with \code{mle}, \code{se},
#'     \code{mle_internal}, \code{se_internal}, bounds, and \code{log}.}
#'   \item{gamma}{Named random-effect modes at the optimum.}
#'   \item{nobs}{Number of observations.}
#'   \item{details}{Laplace evaluation at the optimum, including
#'     \code{outer_opt}, \code{inner_opt}, \code{vcov}, \code{log_lik},
#'     \code{gradient}, and \code{hessian}.}
#'   \item{model_data, ad_pack, config, cache}{Objects needed by the accessor
#'     methods and for warm restarts.}
#' }
#' If \code{optim} throws (for example L-BFGS-B when \code{fn} is not finite),
#' the same class is returned with \code{par_info} estimates set to \code{NA},
#' \code{details$outer_opt = NULL}, and \code{details$error} set to the
#' condition message. \code{ad_pack} and \code{cache} are still present so the
#' Laplace objective can be inspected or re-run without rebuilding tapes:
#' \preformatted{
#' stats::optim(
#'   fit$cache$last_par_fn,
#'   fn = adlaplace::outer_fn, gr = adlaplace::outer_gr,
#'   method = fit$method, control = fit$control,
#'   lower = fit$par_info$lower, upper = fit$par_info$upper,
#'   config = fit$config, ad_pack = fit$ad_pack,
#'   cache = fit$cache, control_inner = fit$control_inner
#' )
#' }
#'
#' @seealso \code{\link{summary.adlaplace_fit}}, \code{\link{predict.adlaplace_fit}},
#'   \code{\link{log_lik_laplace}}, \code{\link{model_data}}, \code{\link{fit_outer}}
#'
#' @examples
#' set.seed(1)
#' n <- 80
#' g <- factor(sample(4, n, replace = TRUE))
#' x <- rbinom(n, 1, 0.5)
#' eta <- 0.5 + 0.4 * x + rnorm(4, sd = 0.3)[g]
#' nb_sd <- 0.3
#' z <- rgamma(n, nb_sd^(-2), nb_sd^(-2))
#' dat <- data.frame(y = rpois(n, exp(eta) * z), x = x, g = g)
#' md <- model_data(
#'   nbinom(y, lower = 1e-9, init = 0.15) ~ x + iid(g, init = 0.3),
#'   data = dat
#' )
#' config <- list(
#'   transform_theta = TRUE,
#'   obs_groups = obs_groups(md$term_data$A, num_shards = 5),
#'   verbose = FALSE
#' )
#' ap <- ad_pack(md, config)
#' cache <- new.env(parent = emptyenv())
#' x0 <- md$term_data$info$parameters$init
#' outer_fn(x = x0, config = config, cache = cache, ad_pack = ap)
#' outer_gr(x = x0, config = config, cache = cache, ad_pack = ap)
#' \donttest{
#' fit <- adlaplace(md, config = list(num_shards = 5L, verbose = FALSE))
#' summary(fit)
#' confint(fit)
#' }
#' @export
adlaplace <- function(
  formula,
  data,
  config = list(),
  control = list(),
  control_inner = list(maxit = 100L, report.level = 0, report.freq = 0),
  method = "L-BFGS-B",
  hessian = NULL,
  start = NULL,
  verbose = FALSE,
  na_omit = TRUE
) {
  cl <- match.call()
  if (is.list(formula) &&
    all(c("terms", "term_data", "observations") %in% names(formula))) {
    md <- formula
    formula <- NULL
  } else {
    md <- model_data(
      formula = formula, data = data,
      verbose = verbose, na_omit = na_omit
    )
  }

  config <- utils::modifyList(
    list(
      transform_theta = TRUE,
      verbose = verbose,
      num_threads = 1L,
      num_shards = 100L
    ),
    config
  )
  config$num_threads <- as.integer(config$num_threads)[1L]
  config$num_shards <- as.integer(config$num_shards)[1L]
  config <- ensure_config_obs_groups(
    config,
    A = md$term_data$A,
    elgm_matrix = md$term_data$elgm_matrix,
    num_shards = config$num_shards,
    num_threads = config$num_threads
  )

  af <- ad_pack(md, config, num_threads = config$num_threads)

  par_meta <- md$term_data$info$parameters
  if (is.null(hessian)) {
    hessian <- length(par_meta$init) <= 5L
  }
  hessian <- isTRUE(hessian)

  control_use <- utils::modifyList(
    list(maxit = 200L, parscale = par_meta$parscale),
    control
  )

  cache <- new.env(parent = emptyenv())
  cache$gamma <- if (!is.null(config$gamma) &&
    length(config$gamma) == nrow(md$term_data$info$gamma)) {
    as.numeric(config$gamma)
  } else {
    rep(0, nrow(md$term_data$info$gamma))
  }

  outer <- fit_outer(
    fn = outer_fn,
    gr = outer_gr,
    par_meta = par_meta,
    method = method,
    control = control_use,
    hessian = hessian,
    start = start,
    config = config,
    ad_pack = af,
    cache = cache,
    control_inner = control_inner
  )

  if (!is.null(outer$error)) {
    return(.adlaplace_fit_optim_failed(
      err = simpleError(outer$error),
      cl = cl,
      formula = formula,
      md = md,
      af = af,
      config = config,
      control_use = control_use,
      control_inner = control_inner,
      method = method,
      par_meta = par_meta,
      cache = cache
    ))
  }

  # Full Laplace evaluation at the MLE for fit$details
  details <- log_lik_laplace(
    x = outer$opt$par,
    gamma = cache$gamma,
    ad_pack = af,
    config = config,
    control = control_inner,
    deriv = TRUE,
    return_hessians = TRUE
  )
  cache$gamma <- details$inner_opt$solution
  cache$fg_x <- as.numeric(outer$opt$par)
  cache$neg_log_lik <- details$neg_log_lik
  cache$d_neg_log_lik <- as.numeric(details$deriv$d_neg_log_lik)
  if (exists("fg_result", envir = cache, inherits = FALSE)) {
    rm("fg_result", envir = cache)
  }

  details$outer_opt <- outer$opt
  details$vcov <- outer$vcov

  out <- list(
    call = cl,
    formula = formula,
    model_data = md,
    ad_pack = af,
    config = config,
    control = control,
    control_inner = control_inner,
    method = method,
    details = details,
    par_info = outer$par_info,
    gamma = stats::setNames(
      details$inner_opt$solution,
      md$term_data$info$gamma$gamma_label
    ),
    nobs = length(md$term_data$y),
    cache = cache
  )
  class(out) <- "adlaplace_fit"
  out
}

#' Outer optimization scaffolding for Laplace profile likelihoods
#'
#' Runs \code{\link[stats]{optim}} on a user-supplied outer objective / gradient
#' (typically \code{\link{outer_fn}} / \code{\link{outer_gr}}, or a modular
#' average of several packs) and returns the standard \code{par_info} /
#' \code{vcov} / \code{opt} structure used by \code{\link{adlaplace}}.
#'
#' @param fn,gr Objective and gradient functions with signature
#'   \code{function(par, ...)}.
#' @param par_meta Data frame of outer-parameter metadata with columns
#'   \code{label}, \code{init}, \code{lower}, \code{upper}, \code{parscale},
#'   \code{log} (as produced by \code{model_data()$term_data$info$parameters}).
#' @param method \code{\link[stats]{optim}} method.
#' @param control Control list for \code{optim}.
#' @param hessian Logical; compute a numerical outer Hessian via
#'   \code{\link[stats]{optimHess}} when \code{TRUE}.
#' @param start Optional starting values on the internal scale; defaults to
#'   \code{par_meta$init}.
#' @param ... Extra arguments forwarded to \code{fn} / \code{gr} / \code{optimHess}
#'   (e.g. \code{config}, \code{ad_pack}, \code{cache}, \code{control_inner}).
#'
#' @return A list with components \code{opt}, \code{par_info}, \code{vcov}, and
#'   \code{error} (character message when \code{optim} fails, otherwise
#'   \code{NULL}).
#'
#' @seealso \code{\link{adlaplace}}, \code{\link{outer_fn}}
#' @export
fit_outer <- function(
  fn,
  gr,
  par_meta,
  method = "L-BFGS-B",
  control = list(),
  hessian = FALSE,
  start = NULL,
  ...
) {
  labels <- par_meta$label
  n_par <- length(par_meta$init)
  if (is.null(start)) {
    start <- as.numeric(par_meta$init)
  } else {
    start <- as.numeric(start)
    if (length(start) != n_par) {
      stop(
        "length(start) (", length(start),
        ") must equal number of outer parameters (", n_par, ")",
        call. = FALSE
      )
    }
    if (any(!is.finite(start))) {
      stop("start must be finite", call. = FALSE)
    }
  }

  control_use <- utils::modifyList(
    list(maxit = 200L, parscale = par_meta$parscale),
    control
  )

  optim_args <- list(
    par = start,
    fn = fn,
    gr = gr,
    method = method,
    control = control_use,
    hessian = FALSE,
    ...
  )
  if (method %in% c("L-BFGS-B", "Brent")) {
    optim_args$lower <- par_meta$lower
    optim_args$upper <- par_meta$upper
  }

  opt <- tryCatch(
    do.call(stats::optim, optim_args),
    error = function(e) e
  )
  if (inherits(opt, "error")) {
    return(list(
      opt = NULL,
      par_info = .par_info_na(par_meta),
      vcov = NULL,
      error = conditionMessage(opt)
    ))
  }

  vc <- NULL
  if (isTRUE(hessian)) {
    hess <- tryCatch(
      stats::optimHess(
        par = opt$par,
        fn = fn,
        gr = gr,
        control = control_use,
        ...
      ),
      error = function(e) e
    )
    if (!inherits(hess, "error") && all(is.finite(hess))) {
      opt$hessian <- hess
      vc <- tryCatch(
        solve(hess),
        error = function(e) {
          warning(
            "outer Hessian is not invertible; vcov unavailable: ",
            conditionMessage(e),
            call. = FALSE
          )
          NULL
        }
      )
      if (!is.null(vc)) {
        dimnames(vc) <- list(labels, labels)
      }
    } else if (inherits(hess, "error")) {
      warning(
        "outer Hessian failed; vcov unavailable: ",
        conditionMessage(hess),
        call. = FALSE
      )
    }
  }

  mle_internal <- stats::setNames(opt$par, labels)
  se_internal <- if (!is.null(vc)) {
    sqrt(pmax(0, diag(vc)))
  } else {
    rep(NA_real_, length(labels))
  }
  names(se_internal) <- labels

  log_idx <- par_meta$log %in% TRUE
  mle <- mle_internal
  mle[log_idx] <- exp(mle_internal[log_idx])
  se <- se_internal
  se[log_idx] <- NA_real_

  par_info <- data.frame(
    label = par_meta$label,
    mle = unname(mle),
    se = unname(se),
    mle_internal = unname(mle_internal),
    se_internal = unname(se_internal),
    init = par_meta$init,
    lower = par_meta$lower,
    upper = par_meta$upper,
    parscale = par_meta$parscale,
    log = par_meta$log,
    stringsAsFactors = FALSE
  )

  list(
    opt = opt,
    par_info = par_info,
    vcov = vc,
    error = NULL
  )
}

#' @noRd
.par_info_na <- function(par_meta) {
  n_par <- length(par_meta$label)
  data.frame(
    label = par_meta$label,
    mle = rep(NA_real_, n_par),
    se = rep(NA_real_, n_par),
    mle_internal = rep(NA_real_, n_par),
    se_internal = rep(NA_real_, n_par),
    init = par_meta$init,
    lower = par_meta$lower,
    upper = par_meta$upper,
    parscale = par_meta$parscale,
    log = par_meta$log,
    stringsAsFactors = FALSE
  )
}

#' Build an adlaplace_fit after outer optim throws
#'
#' @noRd
.adlaplace_fit_optim_failed <- function(
    err,
    cl,
    formula,
    md,
    af,
    config,
    control_use,
    control_inner,
    method,
    par_meta,
    cache) {
  err_msg <- conditionMessage(err)
  labels <- par_meta$label
  last_x <- if (!is.null(cache$last_par_fn)) {
    as.numeric(cache$last_par_fn)
  } else {
    as.numeric(par_meta$init)
  }
  names(last_x) <- labels
  log_idx <- par_meta$log %in% TRUE
  last_nat <- last_x
  last_nat[log_idx] <- exp(last_x[log_idx])
  g <- cache$gamma

  message("outer optim failed: ", err_msg)
  message(
    "last outer parameters (internal):\n",
    paste(sprintf("  %s = %s", labels, format(last_x, digits = 6L)), collapse = "\n")
  )
  message(
    "last outer parameters (natural scale for log slots):\n",
    paste(sprintf("  %s = %s", labels, format(last_nat, digits = 6L)), collapse = "\n")
  )
  if (!is.null(cache$last_par_gr)) {
    gr <- as.numeric(cache$last_par_gr)
    names(gr) <- labels
    message(
      "last outer gradient:\n",
      paste(sprintf("  %s = %s", labels, format(gr, digits = 6L)), collapse = "\n")
    )
  }
  message(
    "gamma length: ", length(g),
    "; any non-finite gamma: ", any(!is.finite(g)),
    "; any non-finite last x: ", any(!is.finite(last_x))
  )

  n_par <- length(labels)
  par_info <- data.frame(
    label = par_meta$label,
    mle = rep(NA_real_, n_par),
    se = rep(NA_real_, n_par),
    mle_internal = rep(NA_real_, n_par),
    se_internal = rep(NA_real_, n_par),
    init = par_meta$init,
    lower = par_meta$lower,
    upper = par_meta$upper,
    parscale = par_meta$parscale,
    log = par_meta$log,
    stringsAsFactors = FALSE
  )
  gamma_lab <- md$term_data$info$gamma$gamma_label
  if (is.null(g)) {
    g <- numeric()
    gamma_lab <- character()
  }
  out <- list(
    call = cl,
    formula = formula,
    model_data = md,
    ad_pack = af,
    config = config,
    control = control_use,
    control_inner = control_inner,
    method = method,
    details = list(outer_opt = NULL, vcov = NULL, error = err_msg),
    par_info = par_info,
    gamma = stats::setNames(as.numeric(g), gamma_lab),
    nobs = length(md$term_data$y),
    cache = cache
  )
  class(out) <- "adlaplace_fit"
  out
}

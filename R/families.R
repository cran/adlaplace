#' Binomial observation model term
#'
#' @description Model term for the observation-level binomial log density with
#' a logit link, registered as \code{binomial_obs}. For each observation with
#' linear predictor \code{eta = X beta + A gamma}, trial count \code{N}, and
#' success count \code{y}, the log density contribution is
#' \code{y * eta - N * log(1 + exp(eta))} (up to a data-only constant). When
#' \code{size} is omitted, \code{N = 1} (Bernoulli). There are no
#' observation-level hyperparameters.
#' @name binomial-class
#' @aliases binom
#' @docType class
#' @title Binomial observation term
#' @exportClass binomial
#'
#' @section Slots (inherited from \code{model}):
#' \describe{
#'   \item{\code{ad_pack}}{Character scalar \code{"binomial_obs"}.}
#'   \item{\code{ad_kind}}{Character scalar \code{"observations"}.}
#' }
#' @section Slots:
#' \describe{
#'   \item{\code{size}}{Optional column name for per-observation trial counts.}
#' }
NULL

setClass(
  "binomial",
  slots = c(size = "character"),
  contains = "model_term",
  prototype = prototype(
    knots = numeric(0),
    ref_value = numeric(0),
    p.order = as.integer(1),
    init = numeric(0),
    lower = numeric(0),
    upper = numeric(0),
    parscale = numeric(0),
    model_role = factor("response", levels = .model_role_levels),
    density = "binomial_obs",
    ad_kind = "observations",
    size = character(0)
  )
)

#' Binomial observation term constructor
#'
#' @description Creates a response term wired to the \code{binomial_obs} AD
#' shard. A Bernoulli likelihood (\code{size} omitted) has no observation-level
#' hyperparameters. With \code{size}, trial counts are read from that column
#' of \code{data} and stored as observation weights.
#'
#' @rdname binomial-class
#' @param x Outcome variable name (counts of successes; 0/1 when Bernoulli).
#' @param size Optional column name for the number of trials per observation.
#' @return A \code{binomial} object.
#' @export
binom <- function(x, size = NULL) {
  x <- strip_term_name(as.character(x))
  size_col <- if (is.null(size)) {
    character(0)
  } else {
    strip_term_name(as.character(size))
  }
  methods::new(
    "binomial",
    name = x,
    label = paste(x, "binomial", sep = "_"),
    formula = stats::as.formula(paste(x, "~."), env = new.env()),
    size = size_col
  )
}

#' Poisson observation model term
#'
#' @description Model term for the observation-level Poisson log density with
#' a log link, registered as \code{poisson_obs}. For each observation with
#' linear predictor \code{eta = X beta + A gamma} and optional offset
#' \code{o} (via \code{config$offset}), the mean is \code{exp(eta + o)} and
#' the log density contribution is
#' \code{y * eta - exp(eta + o) + y * o - lgamma(y + 1)}. There are no
#' observation-level hyperparameters.
#' @name poisson-class
#' @aliases pois
#' @docType class
#' @title Poisson observation term
#' @exportClass poisson
#'
#' @section Slots (inherited from \code{model}):
#' \describe{
#'   \item{\code{ad_pack}}{Character scalar \code{"poisson_obs"}.}
#'   \item{\code{ad_kind}}{Character scalar \code{"observations"}.}
#' }
NULL

setClass(
  "poisson",
  contains = "model_term",
  prototype = prototype(
    knots = numeric(0),
    ref_value = numeric(0),
    p.order = as.integer(1),
    init = numeric(0),
    lower = numeric(0),
    upper = numeric(0),
    parscale = numeric(0),
    model_role = factor("response", levels = .model_role_levels),
    density = "poisson_obs",
    ad_kind = "observations"
  )
)

#' Poisson observation term constructor
#'
#' @description Creates a response term wired to the \code{poisson_obs} AD
#' shard. There are no observation-level hyperparameters. An exposure or
#' expected-count offset is supplied through \code{config$offset} when
#' building AD shards (for example \code{log(E)} in disease mapping).
#'
#' @rdname poisson-class
#' @param x Outcome variable name (non-negative counts).
#' @return A \code{poisson} object.
#' @export
pois <- function(x) {
  x <- strip_term_name(as.character(x))
  methods::new(
    "poisson",
    name = x,
    label = paste(x, "poisson", sep = "_"),
    formula = stats::as.formula(paste(x, "~."), env = new.env())
  )
}

#' Gaussian observation model term
#'
#' @description Model term for the observation-level Gaussian log density
#' registered as \code{gaussian_obs}. This is the default observation term:
#' a bare response symbol on the left-hand side of a formula (e.g.
#' \code{y ~ x}) is coerced to \code{normal(y)} by
#' \code{\link{collect_terms}}.
#' @name gaussian-class
#' @aliases normal
#' @docType class
#' @title Gaussian observation term
#' @exportClass gaussian
#'
#' @section Slots (inherited from \code{model}):
#' \describe{
#'   \item{\code{ad_pack}}{Character scalar \code{"gaussian_obs"}.}
#'   \item{\code{ad_kind}}{Character scalar \code{"observations"}.}
#' }
NULL

setClass(
  "gaussian",
  contains = "model_term",
  prototype = prototype(
    knots = numeric(0),
    ref_value = numeric(0),
    p.order = as.integer(1),
    init = numeric(0),
    lower = numeric(0),
    upper = numeric(0),
    parscale = numeric(0),
    model_role = factor("response", levels = .model_role_levels),
    density = "gaussian_obs",
    ad_kind = "observations"
  )
)

#' Gaussian observation term constructor
#'
#' @description Creates a response term wired to the \code{gaussian_obs} AD
#' shard, with a single residual standard deviation parameter (log scale
#' during optimization).
#'
#' @rdname gaussian-class
#' @param x Outcome variable name.
#' @param init Initial value for the residual standard deviation.
#' @param lower Lower bound for the residual standard deviation.
#' @param upper Upper bound for the residual standard deviation.
#' @param parscale Parameter scale for optimization.
#' @param log Whether theta is optimized on the log scale.
#' @param term A model term object (S4 methods).
#' @return A \code{gaussian} object.
#' @export
normal <- function(x,
                   init = 1,
                   lower = .my_theta_lower,
                   upper = .my_theta_upper,
                   parscale = .my_theta_parscale,
                   log = TRUE) {
  x <- strip_term_name(as.character(x))
  methods::new(
    "gaussian",
    name = x,
    label = paste(x, "gaussian_sd", sep = "_"),
    formula = stats::as.formula(paste(x, "~."), env = new.env()),
    init = init,
    lower = lower,
    upper = upper,
    parscale = parscale,
    log = log
  )
}

#' @describeIn gaussian-class Theta info for the residual standard deviation.
#' @export
setMethod("theta_info", "gaussian", function(term) {
  data.frame(
    term = term@name,
    model = "gaussian",
    label = term@label,
    init = term@init,
    lower = term@lower,
    upper = term@upper,
    parscale = term@parscale,
    model_role = term@model_role,
    log = term@log,
    stringsAsFactors = FALSE
  )
})

#' Negative binomial observation model term
#'
#' @description Model term for the observation-level negative binomial log density
#' registered as \code{nbinom_obs}.
#' @name nbinom-class
#' @aliases nbinom
#' @docType class
#' @title Negative binomial observation term
#' @exportClass nbinom
#'
#' @section Slots (inherited from \code{model}):
#' \describe{
#'   \item{\code{ad_pack}}{Character scalar \code{"nbinom_obs"}.}
#'   \item{\code{ad_kind}}{Character scalar \code{"observations"}.}
#' }
NULL

setClass(
  "nbinom",
  contains = "model_term",
  prototype = prototype(
    knots = numeric(0),
    ref_value = numeric(0),
    p.order = as.integer(1),
    init = numeric(0),
    lower = numeric(0),
    upper = numeric(0),
    parscale = numeric(0),
    model_role = factor("response", levels = .model_role_levels),
    density = "nbinom_obs",
    ad_kind = "observations"
  )
)

#' Negative binomial observation term constructor
#'
#' @description Creates a response term wired to the \code{nbinom_obs} AD shard.
#' @rdname nbinom-class
#' @param x Outcome variable name.
#' @param init Initial value for the overdispersion parameter (log-SD scale).
#' @param lower Lower bound for the overdispersion parameter.
#' @param upper Upper bound for the overdispersion parameter.
#' @param parscale Parameter scale for optimization.
#' @param log Whether theta is optimized on the log scale.
#' @param term A model term object (S4 methods).
#' @return A \code{nbinom} object.
#' @export
nbinom <- function(x,
                   init = .my_theta_init,
                   lower = .my_theta_lower,
                   upper = .my_theta_upper,
                   parscale = .my_theta_parscale,
                   log = TRUE) {
  x <- strip_term_name(as.character(x))
  methods::new(
    "nbinom",
    name = x,
    label = paste(x, "nbinom_sd", sep = "_"),
    formula = stats::as.formula(paste(x, "~."), env = new.env()),
    init = init,
    lower = lower,
    upper = upper,
    parscale = parscale,
    log = log
  )
}

#' @describeIn nbinom-class Theta info for the overdispersion parameter.
#' @export
setMethod("theta_info", "nbinom", function(term) {
  data.frame(
    term = term@name,
    model = "nbinom",
    label = term@label,
    init = term@init,
    lower = term@lower,
    upper = term@upper,
    parscale = term@parscale,
    model_role = term@model_role,
    log = term@log,
    stringsAsFactors = FALSE
  )
})


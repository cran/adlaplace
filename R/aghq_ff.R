#' Build \pkg{aghq} objective list from an adlaplace fit
#'
#' Returns \code{list(fn, gr, he)} suitable for \code{\link[aghq]{aghq}},
#' closing over the fitted AD handle and cache. The wrappers return the
#' \emph{negated} Laplace-approximate log posterior (same convention as
#' \code{\link{outer_fn}} / \code{\link{outer_gr}}), so call
#' \code{aghq::default_control(negate = TRUE)}.
#'
#' @param fit An \code{"adlaplace_fit"} from \code{\link{adlaplace}}.
#' @param control_inner Inner trust-region control list; defaults to
#'   \code{fit$control_inner} when present, otherwise a quiet short run.
#' @return A list with \code{fn}, \code{gr}, and \code{he}. The Hessian is
#'   a numerical Jacobian of \code{gr} (no AD outer Hessian is available).
#' @seealso \code{\link{outer_fn}}, \code{\link{outer_gr}}, \code{\link{adlaplace}}
#' @export
aghq_ff <- function(fit, control_inner = NULL) {
  if (!inherits(fit, "adlaplace_fit")) {
    stop("`fit` must be an adlaplace_fit object", call. = FALSE)
  }
  if (is.null(fit$ad_pack) || is.null(fit$config) || is.null(fit$cache)) {
    stop("`fit` must contain ad_pack, config, and cache", call. = FALSE)
  }
  if (is.null(control_inner)) {
    control_inner <- fit$control_inner
  }
  if (is.null(control_inner)) {
    control_inner <- list(maxit = 100L, report.level = 0, report.freq = 0)
  }

  ad_pack <- fit$ad_pack
  config <- fit$config
  cache <- fit$cache

  list(
    fn = function(theta, ...) {
      outer_fn(
        x = theta,
        config = config,
        cache = cache,
        ad_pack = ad_pack,
        control_inner = control_inner,
        ...
      )
    },
    gr = function(theta, ...) {
      outer_gr(
        x = theta,
        config = config,
        cache = cache,
        ad_pack = ad_pack,
        control_inner = control_inner,
        ...
      )
    },
    he = function(theta, ...) {
      numDeriv::jacobian(
        function(tt) {
          outer_gr(
            x = tt,
            config = config,
            cache = cache,
            ad_pack = ad_pack,
            control_inner = control_inner,
            ...
          )
        },
        theta
      )
    }
  )
}

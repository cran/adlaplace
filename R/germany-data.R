#' Germany oral cavity cancer (Besag-York-Mollie example)
#'
#' Disease-mapping data distributed with \pkg{INLA} (Besag, York, and Mollie 1991).
#' Counts \eqn{y_i} in 544 districts follow
#' \eqn{y_i \sim \mathrm{Poisson}(E_i e^{\eta_i})} with a BYM spatial random
#' effect. The bundled ICAR precision \code{Q_scaled} is the INLA
#' \code{scale.model} version with sum-to-zero constraint; \code{prec} is the
#' \code{random_mult} precision payload (\code{Q}, \code{log_det}, \code{rank}).
#'
#' District boundaries are not stored in this object (to avoid a hard
#' dependency on \pkg{terra}). They ship as
#' \code{system.file("extdata", "germany_map.gpkg", package = "adlaplace")}
#' and can be loaded with \code{terra::vect(...)} when \pkg{terra} is installed.
#'
#' @format A list with:
#' \describe{
#'   \item{Y}{Observed case counts (length 544).}
#'   \item{E}{Expected counts (length 544).}
#'   \item{region}{District index (length 544).}
#'   \item{adj}{District adjacency matrix (\code{dgCMatrix}, 544 x 544).}
#'   \item{Q_scaled}{Scaled ICAR precision (\code{dsCMatrix}, \code{uplo = "U"}).}
#'   \item{prec}{List passed to \code{density = "random_mult"}: \code{Q}
#'     (\code{dsCMatrix}, \code{uplo = "U"}), \code{log_det}, and \code{rank}.}
#' }
#' @source \pkg{INLA} \code{Germany} data and \code{germany.graph}.
#' @references
#' Besag, J., York, J., and Mollie, A. (1991). Bayesian image restoration,
#' with two applications in spatial statistics. \emph{Annals of the Institute
#' of Statistical Mathematics}, 43(1), 1--20.
#'
#' Rue, H., Martino, S., and Chopin, N. (2009). Approximate Bayesian inference
#' for latent Gaussian models by using integrated nested Laplace approximations.
#' \emph{Journal of the Royal Statistical Society: Series B}, 71(2), 319--392.
#' @keywords datasets
"germany"

% Generated manually
\docType{data}
\name{gamm}
\alias{gamm}
\alias{dat}
\title{Example GAMM simulation data}
\format{
A data frame \code{dat} with 500 rows and columns from
\code{mgcv::gamSim(6, n = 500)}, plus simulated negative-binomial responses
used in the GAMM vignette (\code{y}, \code{f2}, \code{mu}, \code{Z}, etc.).
}
\description{Simulation data for the GAMM vignette example.}
\source{
Generated with \code{mgcv::gamSim()} and a custom negative-binomial
response; see the regeneration comments in \code{vignettes/gamm.Rmd}.
}
\examples{
data("gamm", package = "adlaplace")
exists("dat")
nrow(dat)
}
\keyword{datasets}

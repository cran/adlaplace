#' London daily mortality and PM10 (case-crossover example)
#'
#' Daily all-cause mortality and air-quality covariates for London,
#' 2002--2006, from the case-crossover example of Tobias et al. (2024).
#'
#' @format A data frame with 1826 rows and columns:
#' \describe{
#'   \item{city}{City label (\code{"London"}).}
#'   \item{date}{Calendar date.}
#'   \item{year, month, day}{Integer calendar components.}
#'   \item{dow}{Day of week (integer encoding from the source file).}
#'   \item{all}{Daily all-cause death count.}
#'   \item{tmean}{Mean temperature (degrees Celsius).}
#'   \item{rh}{Relative humidity (percent).}
#'   \item{pm10}{PM10 concentration.}
#' }
#' @source
#' \url{https://github.com/aureliotobias/casecrossover}
#' @references
#' Tobias, A., et al. (2024). Case-crossover designs for air pollution
#' epidemiology (example data).
#' @keywords datasets
#' @examples
#' data("london", package = "adlaplace")
#' head(london)
"london"

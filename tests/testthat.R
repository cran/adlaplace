library(testthat)
library(adlaplace)

# CRAN policy: at most two cores at once. data.table OpenMP can otherwise
# run alongside package OpenMP teams and inflate CPU/elapsed time.
data.table::setDTthreads(1L)

test_check("adlaplace")

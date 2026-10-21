#!/usr/bin/env bash
# Install adlaplace with CppAD thread_alloc asserts enabled (-UNDEBUG) and run
# the parallel CppAD teardown / setup-transition tests.
#
# Usage (from repo root or this package directory):
#   ./adlaplace/inst/scripts/check-cppad-threads.sh
#
# Or:
#   PKG_CXXFLAGS="-DADLAPLACE_CPPAD_ASSERT" R CMD INSTALL --clean adlaplace
#   Rscript -e 'testthat::test_dir("adlaplace/tests/testthat", filter="cppad")'
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PKG="${ROOT}/adlaplace"

echo "check-cppad-threads: installing with -DADLAPLACE_CPPAD_ASSERT (-UNDEBUG)..."
PKG_CXXFLAGS="-DADLAPLACE_CPPAD_ASSERT" R CMD INSTALL --clean "${PKG}"

echo "check-cppad-threads: running CppAD parallel tests..."
Rscript -e "
  library(testthat)
  library(adlaplace)
  # Prefer package tests next to the installed sources when developing.
  test_paths <- c(
    file.path('${PKG}', 'tests', 'testthat'),
    system.file('tests', 'testthat', package = 'adlaplace')
  )
  test_dir <- test_paths[dir.exists(test_paths)][1]
  if (is.na(test_dir) || !nzchar(test_dir)) {
    stop('could not find tests/testthat')
  }
  # Load helpers used by test-cppad-*.R (test_ad_data, etc.).
  helper <- file.path(test_dir, 'helper-ad-data.R')
  if (file.exists(helper)) source(helper)
  test_files <- list.files(test_dir, pattern = '^test-cppad', full.names = TRUE)
  if (!length(test_files)) stop('no test-cppad*.R files in ', test_dir)
  for (f in test_files) {
    cat('==> ', basename(f), '\\n', sep = '')
    test_file(f, reporter = 'summary')
  }
"

echo "check-cppad-threads: OK"

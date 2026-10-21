# OpenMP detection for adlaplace and backends that source this script.
# CppAD comes from LinkingTo: RCppAD (no system CppAD / -lcppad_lib).

OPENMP_CPPFLAGS=""
OPENMP_CXXFLAGS=""
OPENMP_LIBS=""
VISIBILITY_CXXFLAGS=""
BREW_PREFIX=""

UNAME_S="$(uname -s 2>/dev/null || echo unknown)"

# R CMD INSTALL/check set R_HOME; bare `R` is rejected there (R Installation §1.6).
# Use ${R_HOME:-} so `set -u` configure wrappers do not fail when unset.
# configure.win sets R_ARCH_BIN (e.g. /x64) so we find R/Rscript under bin/x64.
if [ -z "${R_HOME:-}" ]; then
  R_HOME=`R RHOME`
fi
if [ -n "${R_ARCH_BIN:-}" ]; then
  R_EXE="${R_HOME}/bin${R_ARCH_BIN}/R"
  RSCRIPT="${R_HOME}/bin${R_ARCH_BIN}/Rscript"
else
  R_EXE="${R_HOME}/bin/R"
  RSCRIPT="${R_HOME}/bin/Rscript"
fi
CXX_CMD="$("${R_EXE}" CMD config CXX 2>/dev/null || echo c++)"
CXXFLAGS_CMD="$("${R_EXE}" CMD config CXXFLAGS 2>/dev/null || true)"
CPPFLAGS_CMD="$("${R_EXE}" CMD config CPPFLAGS 2>/dev/null || true)"

# Probe scratch dirs go under TMPDIR (mktemp). Relative config.$$ in the
# package tree is fragile during R CMD INSTALL / staged installs.
make_conftest_dir () {
  mktemp -d "${TMPDIR:-/tmp}/adlaplace-conftest.XXXXXX"
}

check_cxxflag () {
  flag="$1"
  tmpdir="$(make_conftest_dir)" || return 1
  cat > "$tmpdir/conftest.cpp" <<'EOF'
int main() { return 0; }
EOF
  # Capture stderr: Clang accepts unknown -Wno-* with exit 0 but still warns
  # (e.g. -Wno-maybe-uninitialized), which R CMD check treats as significant.
  log="$tmpdir/conftest.log"
  if eval "$CXX_CMD $CPPFLAGS_CMD $CXXFLAGS_CMD $flag -c $tmpdir/conftest.cpp -o $tmpdir/conftest.o" \
       >"$log" 2>&1 \
     && ! grep -Eqi 'unknown warning option|Wunknown-warning-option' "$log"; then
    rm -rf "$tmpdir"
    return 0
  fi
  echo "configure: C++ flag probe failed for '$flag':" >&2
  cat "$log" >&2 || true
  eval "$CXX_CMD $CPPFLAGS_CMD $CXXFLAGS_CMD $flag -c $tmpdir/conftest.cpp -o $tmpdir/conftest.o" >&2 || true
  rm -rf "$tmpdir"
  return 1
}

check_openmp_darwin () {
  tmpdir="$(make_conftest_dir)" || return 1
  cat > "$tmpdir/conftest.cpp" <<'EOF'
#include <omp.h>
int main() {
  omp_set_num_threads(1);
  return omp_get_max_threads() >= 1 ? 0 : 1;
}
EOF
  # Prefer dynamic libomp: static libomp.a + dyn.load (pkgload/load_all) can
  # segfault in OpenMP TLS on macOS under R's -undefined dynamic_lookup.
  if eval "$CXX_CMD $CPPFLAGS_CMD $CXXFLAGS_CMD -I$BREW_PREFIX/opt/libomp/include -Xpreprocessor -fopenmp \
      $tmpdir/conftest.cpp -L$BREW_PREFIX/opt/libomp/lib -lomp \
      -Wl,-rpath,$BREW_PREFIX/opt/libomp/lib -o $tmpdir/conftest" \
      >/dev/null 2>&1; then
    rm -rf "$tmpdir"
    return 0
  fi
  echo "configure: macOS OpenMP link probe failed:" >&2
  eval "$CXX_CMD $CPPFLAGS_CMD $CXXFLAGS_CMD -I$BREW_PREFIX/opt/libomp/include -Xpreprocessor -fopenmp \
      $tmpdir/conftest.cpp -L$BREW_PREFIX/opt/libomp/lib -lomp \
      -Wl,-rpath,$BREW_PREFIX/opt/libomp/lib -o $tmpdir/conftest" >&2 || true
  rm -rf "$tmpdir"
  return 1
}

check_openmp_linux () {
  tmpdir="$(make_conftest_dir)" || return 1
  cat > "$tmpdir/conftest.cpp" <<'EOF'
#include <omp.h>
int main() {
  omp_set_num_threads(1);
  return omp_get_max_threads() >= 1 ? 0 : 1;
}
EOF
  if eval "$CXX_CMD $CPPFLAGS_CMD $CXXFLAGS_CMD -fopenmp \
      $tmpdir/conftest.cpp -o $tmpdir/conftest" \
      >/dev/null 2>&1; then
    rm -rf "$tmpdir"
    return 0
  fi
  echo "configure: Linux OpenMP link probe failed:" >&2
  eval "$CXX_CMD $CPPFLAGS_CMD $CXXFLAGS_CMD -fopenmp \
      $tmpdir/conftest.cpp -o $tmpdir/conftest" >&2 || true
  rm -rf "$tmpdir"
  return 1
}

# ---- Brew discovery (macOS only; used for libomp paths, not -I for all of Homebrew) ----
# Do not add -I$BREW_PREFIX/include: that can put system CppAD ahead of LinkingTo: RCppAD.
if [ "$UNAME_S" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
  BREW_PREFIX="$(brew --prefix 2>/dev/null || true)"
fi

# ---- OpenMP (optional; SystemRequirements lists OpenMP as recommended) ----
# Soft-fail: leave OPENMP_* empty so the package installs serially.
#
# Emscripten / WebR wasm: no omp.h; skip probes entirely. uname is still Linux
# under emconfigure, and a failed em++ -fopenmp test can abort the install even
# when the shell logic intends soft-fail.
IS_WASM_TOOLCHAIN=0
if [ -n "${EMSCRIPTEN:-}" ] || [ -n "${EMSDK:-}" ] || [ "${EMCONFIGURE_JS:-}" = "1" ]; then
  IS_WASM_TOOLCHAIN=1
fi
case "$CXX_CMD" in
  *em++*|*emcc*) IS_WASM_TOOLCHAIN=1 ;;
esac

if [ "$IS_WASM_TOOLCHAIN" = "1" ]; then
  echo "configure: OpenMP not enabled (Emscripten/wasm toolchain; no libomp)" >&2
elif [ "$UNAME_S" = "Darwin" ]; then
  if [ -n "$BREW_PREFIX" ] \
     && [ -f "$BREW_PREFIX/opt/libomp/include/omp.h" ] \
     && [ -f "$BREW_PREFIX/opt/libomp/lib/libomp.dylib" ]; then
    OPENMP_CPPFLAGS="-I$BREW_PREFIX/opt/libomp/include"
    OPENMP_CXXFLAGS="-Xpreprocessor -fopenmp"
    OPENMP_LIBS="-L$BREW_PREFIX/opt/libomp/lib -lomp -Wl,-rpath,$BREW_PREFIX/opt/libomp/lib"
    if check_openmp_darwin; then
      echo "configure: OpenMP enabled on macOS using dynamic Homebrew libomp" >&2
    else
      # Headers/dylib present: still enable flags. A failed probe can be
      # transient (TMPDIR/noexec, PATH) and omitting -lomp breaks dyn.load.
      echo "configure: WARNING: OpenMP link probe failed; enabling Homebrew libomp flags anyway" >&2
    fi
  else
    echo "configure: OpenMP not enabled on macOS (Homebrew libomp not found)" >&2
    echo "configure: install with: brew install libomp (recommended; SystemRequirements: OpenMP)" >&2
  fi
else
  if check_cxxflag "-fopenmp" && check_openmp_linux; then
    OPENMP_CXXFLAGS="-fopenmp"
    OPENMP_LIBS="-fopenmp"
    echo "configure: OpenMP enabled using -fopenmp" >&2
  else
    echo "configure: OpenMP not enabled (compiler does not accept -fopenmp)" >&2
    echo "configure: install a toolchain with OpenMP for multi-threaded fits (SystemRequirements: OpenMP)" >&2
  fi
fi

# ---- Symbol visibility (backends; keep CppAD statics local to each DSO) ----
# Only emit -fvisibility=hidden after a compiler probe (CRAN non-portable flags).
if check_cxxflag "-fvisibility=hidden"; then
  VISIBILITY_CXXFLAGS="-fvisibility=hidden"
  echo "configure: using -fvisibility=hidden" >&2
else
  echo "configure: -fvisibility=hidden not supported; leaving default visibility" >&2
fi

# Do not pass -Wno-* (CRAN/winbuilder: non-portable compilation flag WARNING).
WARN_CXXFLAGS=""

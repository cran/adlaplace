#ifndef ADLAPLACE_OMP_COMPAT_HPP
#define ADLAPLACE_OMP_COMPAT_HPP

// OpenMP is optional at compile time. When _OPENMP is undefined, provide
// serial stubs so call sites can keep using omp_* without #ifdef clutter.
// #pragma omp is ignored by compilers when OpenMP is not enabled.

#ifdef _OPENMP
#include <omp.h>
#else
inline void omp_set_dynamic(int) {}
inline void omp_set_num_threads(int) {}
inline int omp_get_thread_num() { return 0; }
inline int omp_get_num_threads() { return 1; }
inline int omp_get_max_threads() { return 1; }
inline int omp_get_num_procs() { return 1; }
inline int omp_in_parallel() { return 0; }
inline int omp_get_dynamic() { return 0; }
#if !defined(omp_get_max_active_levels)
inline int omp_get_max_active_levels() { return 1; }
#endif
#endif

#endif

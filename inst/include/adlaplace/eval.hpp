#ifndef ADLAPLACE_EVAL_HPP
#define ADLAPLACE_EVAL_HPP

#include <cstddef>

#include "adlaplace/backend.hpp"

int pack_sparsity_sizes(
  AdTape& ad_pack,
  int* n_inner,
  int* n_outer,
  int* n_beta,
  int* n_theta,
  int* nnz_grad_inner,
  int* nnz_grad_outer,
  int* nnz_hes_inner,
  int* nnz_hes_outer);

int get_pattern(
  AdTape& ad_pack,
  int* pattern_grad_inner,
  int* pattern_grad_outer,
  int* pattern_hes_inner_row,
  int* pattern_hes_inner_col,
  int* pattern_hes_outer_row,
  int* pattern_hes_outer_col);

const char* trace_hinv_t_strerror(int rc);

// Defined in eval_impl.hpp (one .cpp per shared library).
class EvalShard;

#endif

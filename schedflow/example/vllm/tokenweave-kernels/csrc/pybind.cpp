#include "cuda_utils.h"
#include "ops.h"
#include "core/registration.h"

#include <torch/library.h>
#include <torch/version.h>

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // TokenWeave Kernels
  ops.def(
      "fused_rs_ln_ag_cta(Tensor! input, Tensor! residual, Tensor weight, "
      "int mcptr, int signal_pads, int rank, int world_size, int MAX_CTAS, "
      "float epsilon) -> ()");
  ops.impl("fused_rs_ln_ag_cta", torch::kCUDA,
             &fused_rs_ln_ag_cta);
}


REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

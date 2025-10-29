#pragma once

#include <optional>
#include <torch/all.h>

#include <vector>

void fused_rs_ln_ag_cta(torch::Tensor& input,     // [..., hidden_size]
    torch::Tensor& residual,  // [..., hidden_size]
    torch::Tensor& weight,    // [hidden_size]
    int64_t mcptr,      // [..., hidden_size] multimem_ptr
    int64_t signal_pads, // [..., hidden_size] signal pads
    int64_t rank,
    int64_t world_size,
    int64_t MAX_CTAS,
    double epsilon) ;

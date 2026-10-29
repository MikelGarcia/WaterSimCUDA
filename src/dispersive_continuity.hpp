#pragma once

#include "config.hpp"
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cstddef>

namespace water {

class DispersiveContinuity {
  public:
    // The borrowed stream must outlive this object.
    DispersiveContinuity(const Config& config, cudaStream_t stream);
    ~DispersiveContinuity();
    DispersiveContinuity(const DispersiveContinuity&) = delete;
    DispersiveContinuity& operator=(const DispersiveContinuity&) = delete;
    void set_depth(float depth);
    void advance(float* eta, const float* u, const float* v, float dt);

  private:
    friend struct ContinuityObservation;
    void release() noexcept;
    Grid grid_;
    cudaStream_t stream_;
    cufftHandle forward_ = 0, inverse_ = 0;
    float* real_ = nullptr;
    cufftComplex* spectrum_ = nullptr;
    float* weights_[2] = {nullptr, nullptr};
    void* workspace_ = nullptr;
    std::size_t workspace_bytes_ = 0;
    int active_ = 0;
    float depth_ = 0;
};

} // namespace water

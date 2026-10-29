#pragma once

#include "config.hpp"
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cstddef>

namespace water {

class Simulation;
struct FftResources;

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
    friend FftResources fft_resources(const Simulation&);
    friend void set_fft_timing_events(Simulation&, cudaEvent_t, cudaEvent_t);
    void release() noexcept;
    Grid grid_;
    cudaStream_t stream_;
    cufftHandle forward_ = 0, inverse_ = 0;
    float* real_ = nullptr;
    cufftComplex* spectrum_ = nullptr;
    float* weights_[2] = {nullptr, nullptr};
    void* workspace_ = nullptr;
    std::size_t workspace_bytes_ = 0;
    cudaEvent_t timing_begin_ = nullptr, timing_end_ = nullptr;
    int active_ = 0;
    float depth_ = 0;
};

} // namespace water

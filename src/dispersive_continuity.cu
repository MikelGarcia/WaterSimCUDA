#include "dispersive_continuity.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace water {
namespace {
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
void check_fft(cufftResult status, const char* operation) {
    if (status != CUFFT_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": cuFFT status " + std::to_string(int(status)));
}
void cleanup(cudaError_t status, const char* operation) noexcept {
    if (status != cudaSuccess)
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
}
void cleanup_fft(cufftResult status, const char* operation) noexcept {
    if (status != CUFFT_SUCCESS)
        std::fprintf(stderr, "%s: cuFFT status %d\n", operation, int(status));
}

__global__ void fill_weights(float* weights, Grid g, float depth) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * g.ny * (g.nx + 1))
        return;
    const int m = i % (g.nx + 1), q = i / (g.nx + 1);
    const int n = q < g.ny ? q : 2 * g.ny - q;
    float weight = 0;
    if ((m || n) && m != g.nx && n != g.ny) {
        constexpr double pi = 3.14159265358979323846;
        const double k = hypot(pi * m / g.lx, pi * n / g.lz);
        const double sx = 2.0 * g.nx / g.lx * sin(pi * m / (2 * g.nx));
        const double sz = 2.0 * g.ny / g.lz * sin(pi * n / (2 * g.ny));
        weight = float(k * tanh(k * double(depth)) / (sx * sx + sz * sz));
    }
    weights[i] = weight;
}

__global__ void mirror_divergence(float* real, const float* u, const float* v, Grid g) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 4 * g.nx * g.ny)
        return;
    const int a = i % (2 * g.nx), b = i / (2 * g.nx);
    const int x = a < g.nx ? a : 2 * g.nx - 1 - a;
    const int y = b < g.ny ? b : 2 * g.ny - 1 - b;
    real[i] = (u[y * (g.nx + 1) + x + 1] - u[y * (g.nx + 1) + x]) * (g.nx / g.lx) +
              (v[(y + 1) * g.nx + x] - v[y * g.nx + x]) * (g.ny / g.lz);
}
__global__ void multiply(cufftComplex* spectrum, const float* weights, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        spectrum[i].x *= weights[i];
        spectrum[i].y *= weights[i];
    }
}
__global__ void crop(float* eta, const float* real, Grid g, float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < g.nx * g.ny)
        eta[i] -= dt * real[(i / g.nx) * (2 * g.nx) + i % g.nx] / (4.0f * g.nx * g.ny);
}
} // namespace

DispersiveContinuity::DispersiveContinuity(const Config& config, cudaStream_t stream)
    : grid_(config.grid), stream_(stream) {
    // The dimension limit bounds byte counts and kernel indices independently of scene settings.
    if (grid_.nx < 8 || grid_.nx > 2048 || grid_.ny < 8 || grid_.ny > 2048 || !std::isfinite(grid_.lx) ||
        grid_.lx <= 0 || !std::isfinite(grid_.lz) || grid_.lz <= 0 || !std::isfinite(config.depth) ||
        config.depth <= 0 || !std::isfinite(config.gravity) || config.gravity <= 0)
        throw std::invalid_argument("invalid dispersive continuity dimensions, depth or gravity");
    try {
        check_fft(cufftCreate(&forward_), "create forward FFT plan");
        check_fft(cufftCreate(&inverse_), "create inverse FFT plan");
        check_fft(cufftSetAutoAllocation(forward_, 0), "disable forward FFT allocation");
        check_fft(cufftSetAutoAllocation(inverse_, 0), "disable inverse FFT allocation");
        std::size_t forward_bytes = 0, inverse_bytes = 0;
        check_fft(cufftMakePlan2d(forward_, 2 * grid_.ny, 2 * grid_.nx, CUFFT_R2C, &forward_bytes),
                  "plan mirrored R2C FFT");
        check_fft(cufftMakePlan2d(inverse_, 2 * grid_.ny, 2 * grid_.nx, CUFFT_C2R, &inverse_bytes),
                  "plan mirrored C2R FFT");
        workspace_bytes_ = std::max(forward_bytes, inverse_bytes);
        if (workspace_bytes_)
            check(cudaMalloc(&workspace_, workspace_bytes_), "allocate FFT workspace");
        check_fft(cufftSetWorkArea(forward_, workspace_), "bind forward FFT workspace");
        check_fft(cufftSetWorkArea(inverse_, workspace_), "bind inverse FFT workspace");
        check_fft(cufftSetStream(forward_, stream_), "bind forward FFT stream");
        check_fft(cufftSetStream(inverse_, stream_), "bind inverse FFT stream");
        const std::size_t real_count = 4 * std::size_t(grid_.nx) * grid_.ny;
        const std::size_t complex_count = 2 * std::size_t(grid_.ny) * (grid_.nx + 1);
        check(cudaMalloc(&real_, real_count * sizeof(float)), "allocate mirrored divergence");
        check(cudaMalloc(&spectrum_, complex_count * sizeof(cufftComplex)), "allocate divergence spectrum");
        for (auto& weights : weights_)
            check(cudaMalloc(&weights, complex_count * sizeof(float)), "allocate depth weights");
        set_depth(config.depth);
    } catch (...) {
        release();
        throw;
    }
}

DispersiveContinuity::~DispersiveContinuity() {
    release();
}

void DispersiveContinuity::release() noexcept {
    cleanup(cudaStreamSynchronize(stream_), "synchronize continuity cleanup");
    if (inverse_)
        cleanup_fft(cufftDestroy(inverse_), "destroy inverse FFT plan");
    if (forward_)
        cleanup_fft(cufftDestroy(forward_), "destroy forward FFT plan");
    if (workspace_)
        cleanup(cudaFree(workspace_), "free FFT workspace");
    if (spectrum_)
        cleanup(cudaFree(spectrum_), "free divergence spectrum");
    if (real_)
        cleanup(cudaFree(real_), "free mirrored divergence");
    for (auto weights : weights_)
        if (weights)
            cleanup(cudaFree(weights), "free depth weights");
}

void DispersiveContinuity::set_depth(float depth) {
    if (!std::isfinite(depth) || depth <= 0)
        throw std::invalid_argument("depth must be finite and positive");
    if (depth == depth_)
        return;
    const int next = 1 - active_, count = 2 * grid_.ny * (grid_.nx + 1);
    fill_weights<<<(count + 255) / 256, 256, 0, stream_>>>(weights_[next], grid_, depth);
    check(cudaGetLastError(), "refresh depth weights");
    active_ = next;
    depth_ = depth;
}

void DispersiveContinuity::advance(float* eta, const float* u, const float* v, float dt) {
    const int real_count = 4 * grid_.nx * grid_.ny;
    const int complex_count = 2 * grid_.ny * (grid_.nx + 1);
    mirror_divergence<<<(real_count + 255) / 256, 256, 0, stream_>>>(real_, u, v, grid_);
    check(cudaGetLastError(), "mirror staggered divergence");
    if (timing_begin_) check(cudaEventRecord(timing_begin_, stream_), "record FFT begin");
    check_fft(cufftExecR2C(forward_, real_, spectrum_), "execute divergence R2C FFT");
    multiply<<<(complex_count + 255) / 256, 256, 0, stream_>>>(spectrum_, weights_[active_], complex_count);
    check(cudaGetLastError(), "apply dispersive depth multiplier");
    check_fft(cufftExecC2R(inverse_, spectrum_, real_), "execute divergence C2R FFT");
    if (timing_end_) check(cudaEventRecord(timing_end_, stream_), "record FFT end");
    crop<<<(grid_.nx * grid_.ny + 255) / 256, 256, 0, stream_>>>(eta, real_, grid_, dt);
    check(cudaGetLastError(), "apply normalized continuity response");
}

} // namespace water

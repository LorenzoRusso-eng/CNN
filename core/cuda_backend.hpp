#pragma once

#include "core/layer.hpp"

#include <cstddef>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace cuda_backend {

enum class TransposeOp {
    NoTrans,
    Trans
};

template <typename T>
class CudaBatchTensor {
public:
    CudaBatchTensor() = default;
    ~CudaBatchTensor();

    CudaBatchTensor(const CudaBatchTensor &) = delete;
    CudaBatchTensor &operator=(const CudaBatchTensor &) = delete;

    CudaBatchTensor(CudaBatchTensor &&other) noexcept;
    CudaBatchTensor &operator=(CudaBatchTensor &&other) noexcept;

    void ensure_capacity(std::size_t count);
    void resize(int batch_size, int height, int width, int channels);
    void clear() noexcept;
    void release() noexcept;

    void copy_from_host(const T *src, int batch_size, int h, int w, int ch);
    void copy_from_device(const T *src, int batch_size, int h, int w, int ch);
    void copy_from_device(const CudaBatchTensor<T> &src);
    void copy_to_host(T *dst, std::size_t count) const;

    T *data() noexcept { return ptr_; }
    const T *data() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return size_; }
    std::size_t capacity() const noexcept { return capacity_; }
    int batch_size() const noexcept { return batch_size_; };
    int height() const noexcept { return height_; };
    int width() const noexcept { return width_; };
    int channels() const noexcept { return channels_; };
    bool empty() const noexcept { return size_ == 0; }

private:
    T *ptr_ = nullptr;
    std::size_t size_ = 0;
    std::size_t capacity_ = 0;
    int batch_size_ = 0;
    int height_ = 0;
    int width_ = 0;
    int channels_ = 0;
};

struct CublasContext {
    CublasContext();
    ~CublasContext();

    CublasContext(const CublasContext &) = delete;
    CublasContext &operator=(const CublasContext &) = delete;

    void *handle = nullptr;
};

struct CudaLayerState {
    CudaBatchTensor<float> a;
    CudaBatchTensor<float> y;
    CudaBatchTensor<float> delta;
    CudaBatchTensor<int> pooling_argmax;

    void clear();
};

struct CudaBatchLayerRuntime {
    CudaBatchTensor<float> a;
    CudaBatchTensor<float> y;
    CudaBatchTensor<float> delta;
    CudaBatchTensor<int> pooling_argmax;
    CudaBatchTensor<float> conv_im2col;
    CudaBatchTensor<float> backprop_cost_from_next;
    CudaBatchTensor<float> reduction_ones;
    CudaBatchTensor<float> loss_values;
    CudaBatchTensor<float> loss_sum;
    CudaBatchTensor<int> loss_reduce_temp;
    int batch_size = 0;
    int flat_size = 0;

    void clear();
};

using CudaRuntimeList = std::vector<CudaLayerState>;
using CudaBatchRuntimeList = std::vector<CudaBatchLayerRuntime>;


struct CudaParameterBuffer {
    std::vector<CudaBatchTensor<float>> dense_weights;
    std::vector<CudaBatchTensor<float>> dense_biases;
    std::vector<CudaBatchTensor<float>> conv_weights;
    std::vector<CudaBatchTensor<float>> conv_biases;

    void clear();
};


void check_cuda(cudaError_t status, const char *context);
void check_cuda_kernel(const char *context);
void check_cublas(cublasStatus_t status, const char *context);
bool is_cuda_enabled() noexcept;
void *current_cublas_handle();
void init_cuda_runtime_buffers(const LayerList &architecture, CudaRuntimeList &runtime);
void init_cuda_batch_runtime_buffers(const LayerList &architecture, CudaBatchRuntimeList &runtime, int batch_size);
void init_cuda_parameter_buffer(const LayerList &architecture, CudaParameterBuffer &buffer);
void init_or_load_velocity(const LayerList &architecture, CudaParameterBuffer &velocity, const TrainingRuntimeState *runtime_state);
void zero_cuda_parameter_buffer(const LayerList &architecture, CudaParameterBuffer &buffer);
void sync_cuda_parameters_from_cpu(const LayerList &architecture, CudaParameterBuffer &buffer);
void sync_cuda_parameters_to_cpu(const LayerList &architecture, const CudaParameterBuffer &buffer, ParameterBuffer &host_buffer);
void sync_cuda_parameters_to_architecture(LayerList &architecture, const CudaParameterBuffer &buffer);



} // namespace cuda_backend

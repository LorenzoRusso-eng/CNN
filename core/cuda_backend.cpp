#include "core/cuda_backend.hpp"

#include "core/layer_validation.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace cuda_backend {
namespace {

struct DeviceMirrorEntry {
    const float *device_ptr = nullptr;
    std::size_t size = 0;
};

std::unordered_map<const float *, DeviceMirrorEntry> &device_mirror_cache(){
    static std::unordered_map<const float *, DeviceMirrorEntry> cache;
    return cache;
}

void register_device_mirror(const float *host_ptr, const CudaBatchTensor<float> &device_buffer){
    if(host_ptr != nullptr && device_buffer.data() != nullptr){
        device_mirror_cache()[host_ptr] = DeviceMirrorEntry{device_buffer.data(), device_buffer.size()};
    }
}

void clear_device_mirror_cache(){
    device_mirror_cache().clear();
}

const float *lookup_device_mirror(const float *host_ptr, std::size_t required_size){
    const auto found = device_mirror_cache().find(host_ptr);
    if(found == device_mirror_cache().end()){
        return nullptr;
    }
    if(found->second.size < required_size){
        return nullptr;
    }
    return found->second.device_ptr;
}

void check_cublas(cublasStatus_t status, const char *context){
    if(status != CUBLAS_STATUS_SUCCESS){
        throw std::runtime_error(std::string(context) + ": cuBLAS error " + std::to_string(static_cast<int>(status)));
    }
}

cublasOperation_t to_cublas_op(TransposeOp op){
    return op == TransposeOp::NoTrans ? CUBLAS_OP_N : CUBLAS_OP_T;
}


struct DeviceOperand {
    CudaBatchTensor<float> temp;
    const float *device_ptr = nullptr;

    const float *prepare(const float *host_ptr, std::size_t size){
        device_ptr = lookup_device_mirror(host_ptr, size);
        if(device_ptr != nullptr){
            return device_ptr;
        }
        temp.copy_from_host(host_ptr, 1, 1, static_cast<int>(size), 1);
        device_ptr = temp.data();
        return device_ptr;
    }
};

float matrix_value_row_major(const float *matrix, int row, int col, int leading_dim){
    return matrix[static_cast<std::size_t>(row) * static_cast<std::size_t>(leading_dim) + static_cast<std::size_t>(col)];
}

std::size_t physical_matrix_size(TransposeOp trans, int physical_rows_if_no_trans, int physical_rows_if_trans, int leading_dim){
    const int physical_rows = (trans == TransposeOp::NoTrans) ? physical_rows_if_no_trans : physical_rows_if_trans;
    return static_cast<std::size_t>(std::max(0, physical_rows)) * static_cast<std::size_t>(std::max(0, leading_dim));
}

} // namespace

void check_cuda(cudaError_t status, const char *context){
    if(status != cudaSuccess){
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
CudaBatchTensor<T>::~CudaBatchTensor(){
    release();
}

template <typename T>
CudaBatchTensor<T>::CudaBatchTensor(CudaBatchTensor &&other) noexcept
    : ptr_(other.ptr_),
      size_(other.size_),
      capacity_(other.capacity_),
      batch_size_(other.batch_size_),
      height_(other.height_),
      width_(other.width_),
      channels_(other.channels_){
    other.ptr_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
    other.batch_size_ = 0;
    other.height_ = 0;
    other.width_ = 0;
    other.channels_ = 0;
}

template <typename T>
CudaBatchTensor<T> &CudaBatchTensor<T>::operator=(CudaBatchTensor &&other) noexcept{
    if(this != &other){
        release();
        ptr_ = other.ptr_;
        size_ = other.size_;
        capacity_ = other.capacity_;
        batch_size_ = other.batch_size_;
        height_ = other.height_;
        width_ = other.width_;
        channels_ = other.channels_;
        other.ptr_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
        other.batch_size_ = 0;
        other.height_ = 0;
        other.width_ = 0;
        other.channels_ = 0;
    }
    return *this;
}

template <typename T>
void CudaBatchTensor<T>::ensure_capacity(std::size_t count){
    if(count <= capacity_){
        return;
    }

    T *new_ptr = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&new_ptr), count * sizeof(T)), "cudaMalloc");
    if(ptr_ != nullptr){
        check_cuda(cudaFree(ptr_), "cudaFree");
    }
    ptr_ = new_ptr;
    capacity_ = count;
}

template <typename T>
void CudaBatchTensor<T>::resize(int batch_size, int height, int width, int channels){
    require_condition(batch_size >= 0 && height >= 0 && width >= 0 && channels >= 0,
                      "CudaBatchTensor::resize: dimensioni negative");
    const std::size_t count = static_cast<std::size_t>(batch_size) *
                              static_cast<std::size_t>(height) *
                              static_cast<std::size_t>(width) *
                              static_cast<std::size_t>(channels);
    ensure_capacity(count);
    size_ = count;
    batch_size_ = batch_size;
    height_ = height;
    width_ = width;
    channels_ = channels;
}

template <typename T>
void CudaBatchTensor<T>::clear() noexcept{
    size_ = 0;
    batch_size_ = 0;
    height_ = 0;
    width_ = 0;
    channels_ = 0;
}

template <typename T>
void CudaBatchTensor<T>::release() noexcept{
    if(ptr_ != nullptr){
        cudaFree(ptr_);
    }
    ptr_ = nullptr;
    size_ = 0;
    capacity_ = 0;
    batch_size_ = 0;
    height_ = 0;
    width_ = 0;
    channels_ = 0;
}

template <typename T>
void CudaBatchTensor<T>::copy_from_host(const T *src, int batch_size, int h, int w, int ch){
    resize(batch_size, h, w, ch);
    if(size_ > 0){
        check_cuda(cudaMemcpy(ptr_, src, size_ * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy HostToDevice");
    }

}

template <typename T>
void CudaBatchTensor<T>::copy_from_device(const T *src, int batch_size, int h, int w, int ch){
    resize(batch_size, h, w, ch);
    if(size_ > 0){
        check_cuda(cudaMemcpy(ptr_, src, size_ * sizeof(T), cudaMemcpyDeviceToDevice), "cudaMemcpy DeviceToDevice");
    }
}

template <typename T>
void CudaBatchTensor<T>::copy_from_device(const CudaBatchTensor<T> &src){
    copy_from_device(src.data(), src.batch_size(), src.height(), src.width(), src.channels());
}

template <typename T>
void CudaBatchTensor<T>::copy_to_host(T *dst, std::size_t count) const{
    require_condition(count <= size_, "CudaBatchTensor::copy_to_host: count maggiore della dimensione del buffer");
    if(count > 0){
        check_cuda(cudaMemcpy(dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy DeviceToHost");
    }
}

template class CudaBatchTensor<float>;
template class CudaBatchTensor<int>;

CublasContext::CublasContext(){
    cublasHandle_t raw_handle = nullptr;
    check_cublas(cublasCreate(&raw_handle), "cublasCreate");
    handle = raw_handle;
}

CublasContext::~CublasContext(){
    if(handle != nullptr){
        cublasDestroy(static_cast<cublasHandle_t>(handle));
    }
    handle = nullptr;
}

void CudaLayerState::clear(){
    a.clear();
    y.clear();
    delta.clear();
    pooling_argmax.clear();
}

void CudaBatchLayerRuntime::clear(){
    a.clear();
    y.clear();
    delta.clear();
    pooling_argmax.clear();
    conv_im2col.clear();
    backprop_cost_from_next.clear();
    reduction_ones.clear();
    batch_size = 0;
    flat_size = 0;
}

void CudaParameterBuffer::clear(){
    dense_weights.clear();
    dense_biases.clear();
    conv_weights.clear();
    conv_biases.clear();
}

bool is_cuda_enabled() noexcept{
    return true;
}

void *current_cublas_handle(){
    static CublasContext context;
    return context.handle;
}

void init_cuda_runtime_buffers(const LayerList &architecture, CudaRuntimeList &runtime){
    runtime.resize(architecture.size());
    for(std::size_t layer_index = 0; layer_index < architecture.size(); layer_index++){
        CudaLayerState &state = runtime[layer_index];
        state.clear();
        const Layer &layer = architecture[layer_index];
        const bool uses_cuda_compute = layer.type == Layer_type::Dense || layer.type == Layer_type::Conv;
        const bool previous_uses_cuda = layer_index > 0 && (architecture[layer_index - 1].type == Layer_type::Dense || architecture[layer_index - 1].type == Layer_type::Conv);
        const bool next_uses_cuda = layer_index + 1 < architecture.size() && (architecture[layer_index + 1].type == Layer_type::Dense || architecture[layer_index + 1].type == Layer_type::Conv);

        if(uses_cuda_compute){
            state.a.resize(1, layer.dim_layer[0], layer.dim_layer[1], layer.dim_layer[2]);
            state.delta.resize(1, layer.dim_layer[0], layer.dim_layer[1], layer.dim_layer[2]);
        }
        if(layer.type == Layer_type::Pooling && layer.pooling_type == Pooling_type::Max){
            state.pooling_argmax.resize(1, layer.dim_layer[0], layer.dim_layer[1], layer.dim_layer[2]);
        }
        if(layer.type == Layer_type::Input || next_uses_cuda || previous_uses_cuda || uses_cuda_compute){
            state.y.resize(1, layer.dim_layer[0], layer.dim_layer[1], layer.dim_layer[2]);
        }
    }
}

void init_cuda_batch_runtime_buffers(const LayerList &architecture, CudaBatchRuntimeList &runtime, int batch_size){
    runtime.resize(architecture.size());
    for(std::size_t layer_index = 0; layer_index < architecture.size(); layer_index++){
        CudaBatchLayerRuntime &state = runtime[layer_index];
        state.clear();
        const Layer &layer = architecture[layer_index];
        const int flat_size = layer.flat_output_size();
        const bool uses_activation = layer.type == Layer_type::Dense || layer.type == Layer_type::Conv;
        const bool previous_uses_cuda = layer_index > 0 && (architecture[layer_index - 1].type == Layer_type::Dense || architecture[layer_index - 1].type == Layer_type::Conv);
        const bool next_uses_cuda = layer_index + 1 < architecture.size() && (architecture[layer_index + 1].type == Layer_type::Dense || architecture[layer_index + 1].type == Layer_type::Conv);

        state.batch_size = batch_size;
        state.flat_size = flat_size;
        state.y.resize(batch_size, layer.dim_layer[0], layer.dim_layer[1], layer.dim_layer[2]);
        state.delta.resize(batch_size, layer.dim_layer[0], layer.dim_layer[1], layer.dim_layer[2]);
        state.backprop_cost_from_next.resize(batch_size, layer.dim_layer[0], layer.dim_layer[1], layer.dim_layer[2]);

        if(uses_activation){
            state.a.resize(batch_size, layer.dim_layer[0], layer.dim_layer[1], layer.dim_layer[2]);
            int ones_count = batch_size;
            if(layer.type == Layer_type::Conv){
                const int patch_size = layer.kernel_dim[0] * layer.kernel_dim[1] * layer.kernel_dim[2];
                const int patches_per_sample = layer.dim_layer[0] * layer.dim_layer[1];
                state.conv_im2col.resize(batch_size, patch_size, layer.input_dim[0] * layer.input_dim[1], 1);
                ones_count = batch_size * patches_per_sample;
            }
            std::vector<float> ones(static_cast<std::size_t>(ones_count), 1.0f);
            state.reduction_ones.copy_from_host(ones.data(), 1, ones_count, 1, 1);
        }

        if(layer.type == Layer_type::Pooling && layer.pooling_type == Pooling_type::Max){
            state.pooling_argmax.resize(batch_size, layer.dim_layer[0], layer.dim_layer[1], layer.dim_layer[2]);
        }

    }
}

void init_or_load_velocity(const LayerList &architecture, CudaParameterBuffer &velocity, const TrainingRuntimeState *runtime_state){
    init_cuda_parameter_buffer(architecture, velocity);
    if(runtime_state == nullptr){
        return;
    }
    if(runtime_state->velocity.dense_weights.size() == architecture.size() &&
       runtime_state->velocity.dense_biases.size() == architecture.size() &&
       runtime_state->velocity.conv_weights.size() == architecture.size() &&
       runtime_state->velocity.conv_biases.size() == architecture.size()){
        for(int l = 0; l < static_cast<int>(architecture.size()); l++){
            if(architecture[l].type == Layer_type::Dense){
                velocity.dense_weights[l].copy_from_host(runtime_state->velocity.dense_weights[l].data(), 1, architecture[l].dense_output_size, architecture[l].dense_input_size, 1);
                velocity.dense_biases[l].copy_from_host(runtime_state->velocity.dense_biases[l].data(), 1, architecture[l].dense_output_size, 1, 1);
            } else if(architecture[l].type == Layer_type::Conv){
                velocity.conv_weights[l].copy_from_host(runtime_state->velocity.conv_weights[l].data(), 1, architecture[l].dim_layer[2], architecture[l].kernel_dim[0] * architecture[l].kernel_dim[1] * architecture[l].kernel_dim[2], 1);
                velocity.conv_biases[l].copy_from_host(runtime_state->velocity.conv_biases[l].data(), 1, architecture[l].dim_layer[2], 1, 1);
            }
        }
         
    }
}

void init_cuda_parameter_buffer(const LayerList &architecture, CudaParameterBuffer &buffer){
    const std::size_t num_layers = architecture.size();
    buffer.dense_weights.resize(num_layers);
    buffer.dense_biases.resize(num_layers);
    buffer.conv_weights.resize(num_layers);
    buffer.conv_biases.resize(num_layers);
    
    for(int l = 0; l < static_cast<int>(num_layers); l++){
        const Layer &layer = architecture[l];
        if(layer.type == Layer_type::Dense){
            buffer.dense_weights[l].resize(1, layer.flat_output_size(), layer.dense_input_size, 1);
            buffer.dense_biases[l].resize(1, layer.flat_output_size(), 1, 1);
        } else {
            buffer.dense_weights[l].clear();
            buffer.dense_biases[l].clear();
        }

        if(layer.type == Layer_type::Conv){
            buffer.conv_weights[l].resize(1,layer.dim_layer[2], layer.kernel_dim[0] * layer.kernel_dim[1] * layer.kernel_dim[2], 1);
            buffer.conv_biases[l].resize(1, layer.dim_layer[2], 1, 1);
        } else {
            buffer.conv_weights[l].clear();
            buffer.conv_biases[l].clear();
        }
    }
}

void zero_cuda_parameter_buffer(const LayerList &architecture, CudaParameterBuffer &buffer){
    for(std::size_t layer_index = 0; layer_index < architecture.size(); layer_index++){
        const Layer &layer = architecture[layer_index];
        if(layer.type == Layer_type::Dense){
            check_cuda(cudaMemset(buffer.dense_weights[layer_index].data(), 0, buffer.dense_weights[layer_index].size() * sizeof(float)), "cudaMemset Dense Weights");
            check_cuda(cudaMemset(buffer.dense_biases[layer_index].data(), 0, buffer.dense_biases[layer_index].size() * sizeof(float)), "cudaMemset Dense Biases");
        } else if(layer.type == Layer_type::Conv){
            check_cuda(cudaMemset(buffer.conv_weights[layer_index].data(), 0, buffer.conv_weights[layer_index].size() * sizeof(float)), "cudaMemset Conv Weights");
            check_cuda(cudaMemset(buffer.conv_biases[layer_index].data(), 0, buffer.conv_biases[layer_index].size() * sizeof(float)), "cudaMemset Conv Biases");
        }
    }
}

void sync_cuda_parameters_from_cpu(const LayerList &architecture, CudaParameterBuffer &buffer){
    if(buffer.dense_weights.size() != architecture.size()){
        init_cuda_parameter_buffer(architecture, buffer);
    }

    for(std::size_t layer_index = 0; layer_index < architecture.size(); layer_index++){
        const Layer &layer = architecture[layer_index];
        if(layer.type == Layer_type::Dense){
            auto &weights = buffer.dense_weights[layer_index];
            auto &biases = buffer.dense_biases[layer_index];
            weights.copy_from_host(
                layer.dense_params.weights.data(),
                1,
                layer.dense_output_size,
                layer.dense_input_size,
                1
            );
            register_device_mirror(layer.dense_params.weights.data(), weights);
            biases.copy_from_host(
                layer.dense_params.bias.data(),
                1,
                1,
                layer.dense_output_size,
                1
            );
            register_device_mirror(layer.dense_params.bias.data(), biases);
        } else if(layer.type == Layer_type::Conv){
            auto &weights = buffer.conv_weights[layer_index];
            auto &biases = buffer.conv_biases[layer_index];
            weights.copy_from_host(
                layer.conv_params.filters.data(),
                1,
                layer.dim_layer[2],
                layer.kernel_dim[0] * layer.kernel_dim[1] * layer.kernel_dim[2],
                1
            );
            register_device_mirror(layer.conv_params.filters.data(), weights);
            biases.copy_from_host(
                layer.conv_params.bias.data(),
                1,
                1,
                layer.dim_layer[2],
                1
            );
            register_device_mirror(layer.conv_params.bias.data(), biases);
        }
    }
}

void sync_cuda_parameters_to_cpu(const LayerList &architecture, const CudaParameterBuffer &buffer, ParameterBuffer &host_buffer){
    host_buffer.dense_weights.resize(architecture.size());
    host_buffer.dense_biases.resize(architecture.size());
    host_buffer.conv_weights.resize(architecture.size());
    host_buffer.conv_biases.resize(architecture.size());

    for(std::size_t layer_index = 0; layer_index < architecture.size(); layer_index++){
        const Layer &layer = architecture[layer_index];
        if(layer.type == Layer_type::Dense){
            host_buffer.dense_weights[layer_index].resize(layer.dense_params.weights.size());
            host_buffer.dense_biases[layer_index].resize(layer.dense_params.bias.size());
            buffer.dense_weights[layer_index].copy_to_host(
                host_buffer.dense_weights[layer_index].data(),
                host_buffer.dense_weights[layer_index].size()
            );
            buffer.dense_biases[layer_index].copy_to_host(
                host_buffer.dense_biases[layer_index].data(),
                host_buffer.dense_biases[layer_index].size()
            );
        } else {
            host_buffer.dense_weights[layer_index].clear();
            host_buffer.dense_biases[layer_index].clear();
        }

        if(layer.type == Layer_type::Conv){
            host_buffer.conv_weights[layer_index].resize(layer.conv_params.filters.size());
            host_buffer.conv_biases[layer_index].resize(layer.conv_params.bias.size());
            buffer.conv_weights[layer_index].copy_to_host(
                host_buffer.conv_weights[layer_index].data(),
                host_buffer.conv_weights[layer_index].size()
            );
            buffer.conv_biases[layer_index].copy_to_host(
                host_buffer.conv_biases[layer_index].data(),
                host_buffer.conv_biases[layer_index].size()
            );
        } else {
            host_buffer.conv_weights[layer_index].clear();
            host_buffer.conv_biases[layer_index].clear();
        }
    }
}

void sync_cuda_parameters_to_architecture(LayerList &architecture, const CudaParameterBuffer &buffer){
    ParameterBuffer host_buffer;
    sync_cuda_parameters_to_cpu(architecture, buffer, host_buffer);

    for(std::size_t layer_index = 0; layer_index < architecture.size(); layer_index++){
        Layer &layer = architecture[layer_index];
        if(layer.type == Layer_type::Dense){
            layer.dense_params.weights = std::move(host_buffer.dense_weights[layer_index]);
            layer.dense_params.bias = std::move(host_buffer.dense_biases[layer_index]);
        } else if(layer.type == Layer_type::Conv){
            layer.conv_params.filters = std::move(host_buffer.conv_weights[layer_index]);
            layer.conv_params.bias = std::move(host_buffer.conv_biases[layer_index]);
        }
    }
}
} // namespace cuda_backend

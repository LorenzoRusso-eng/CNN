#include "math/loss_cuda.hpp"

#include "kernels/general.hpp"

#include <cstddef>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

float Loss_fun(
    cuda_backend::CudaBatchLayerRuntime &output_runtime,
    const Loss &loss,
    const cuda_backend::CudaBatchTensor<float> &desired_output
){

    float loss_value = 0.0f;
    int total_output_size = output_runtime.y.size();
    if(total_output_size > 0){
        int GridDim = (total_output_size + 256 -1) / 256;
        compute_loss <<< GridDim, 256 >>>(
            output_runtime.y.data(), desired_output.data(),
            total_output_size, loss.kind, loss.reduction(),
            output_runtime.a.data()
        );

        float *loss_sum_device = nullptr;
        cuda_backend::check_cuda(cudaMalloc(reinterpret_cast<void **>(&loss_sum_device), sizeof(float)), "cudaMalloc loss sum");

        void *temp_storage = nullptr;
        std::size_t temp_storage_bytes = 0;
        cub::DeviceReduce::Sum(
            temp_storage,
            temp_storage_bytes,
            output_runtime.a.data(),
            loss_sum_device,
            total_output_size
        );

        cuda_backend::check_cuda(cudaMalloc(&temp_storage, temp_storage_bytes), "cudaMalloc loss reduction temp storage");
        cub::DeviceReduce::Sum(
            temp_storage,
            temp_storage_bytes,
            output_runtime.a.data(),
            loss_sum_device,
            total_output_size
        );

        cuda_backend::check_cuda(cudaMemcpy(&loss_value, loss_sum_device, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy loss sum");
        cuda_backend::check_cuda(cudaFree(temp_storage), "cudaFree loss reduction temp storage");
        cuda_backend::check_cuda(cudaFree(loss_sum_device), "cudaFree loss sum");
    }
    return loss_value;
}

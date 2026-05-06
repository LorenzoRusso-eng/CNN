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
    int flat_size = output_runtime.y.channels() * output_runtime.y.width() * output_runtime.y.height();
    if(total_output_size > 0){
        output_runtime.loss_values.resize(
            output_runtime.y.batch_size(),
            output_runtime.y.height(),
            output_runtime.y.width(),
            output_runtime.y.channels()
        );
        output_runtime.loss_sum.resize(1, 1, 1, 1);

        int GridDim = (total_output_size + 256 -1) / 256;
        compute_loss <<< GridDim, 256 >>>(
            output_runtime.y.data(), desired_output.data(),
            total_output_size, flat_size, loss.kind, loss.reduction(), loss.beta,
            output_runtime.loss_values.data()
        );
        cuda_backend::check_cuda_kernel("compute_loss");

        void *temp_storage = nullptr;
        std::size_t temp_storage_bytes = 0;
        cuda_backend::check_cuda(
            cub::DeviceReduce::Sum(
                temp_storage,
                temp_storage_bytes,
                output_runtime.loss_values.data(),
                output_runtime.loss_sum.data(),
                total_output_size
            ),
            "loss reduction temp size"
        );

        const std::size_t temp_storage_ints =
            (temp_storage_bytes + sizeof(int) - 1) / sizeof(int);
        output_runtime.loss_reduce_temp.resize(
            1,
            static_cast<int>(temp_storage_ints),
            1,
            1
        );
        temp_storage = output_runtime.loss_reduce_temp.data();

        cuda_backend::check_cuda(
            cub::DeviceReduce::Sum(
                temp_storage,
                temp_storage_bytes,
                output_runtime.loss_values.data(),
                output_runtime.loss_sum.data(),
                total_output_size
            ),
            "loss reduction sum"
        );

        cuda_backend::check_cuda(cudaMemcpy(&loss_value, output_runtime.loss_sum.data(), sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy loss sum");
    }
    return loss_value;
}

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include <vector>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int voxel_pooling_forward_wrapper(int batch_size, int num_points, int num_channels, int num_voxel_x, int num_voxel_y, int num_voxel_z, at::Tensor geom_xyz_tensor,
                       at::Tensor input_features_tensor, at::Tensor output_features_tensor, at::Tensor pos_memo_tensor);

void voxel_pooling_forward_kernel_launcher(int batch_size, int num_points, int num_channels, int num_voxel_x, int num_voxel_y, int num_voxel_z, const int *geom_xyz, const float *input_features,
                                float *output_features, int *pos_memo, cudaStream_t stream);

int voxel_pooling_forward_wrapper(
    int batch_size, int num_points, int num_channels, 
    int voxel_size, int grid_size, 
    at::Tensor geom_xyz_tensor, at::Tensor input_features_tensor, 
    at::Tensor output_features_tensor, at::Tensor output_idx_tensor) {
  
    CHECK_INPUT(geom_xyz_tensor);
    CHECK_INPUT(input_features_tensor);
    CHECK_INPUT(output_features_tensor);
    CHECK_INPUT(output_idx_tensor);

    at::cuda::CUDAGuard device_guard(geom_xyz_tensor.device());
    
    return 1;
}

// Method Definitions
static PyMethodDef VoxelPoolingMethods[] = {
    {nullptr, nullptr, 0, nullptr}  // Sentinel
};

// Module Definition
static struct PyModuleDef voxel_pooling_extmodule = {
    PyModuleDef_HEAD_INIT,
    "voxel_pooling_ext",
    nullptr,
    -1,
    VoxelPoolingMethods
};

// Initialize the module
PyMODINIT_FUNC PyInit_voxel_pooling_ext(void) {
    return PyModule_Create(&voxel_pooling_extmodule);
}


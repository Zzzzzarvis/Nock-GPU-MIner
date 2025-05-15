// field_ops_ffi.cu - FFI implementation for Finite Field Operations CUDA module

#include "field_ops_ffi.h"
#include "field_ops.cu" // Include the actual CUDA kernel implementation
#include <cuda_runtime.h>
#include <cstdio> // For printf

// Helper to check CUDA errors (can be shared or redefined)
static CudaFFIErrorCode check_cuda_error_field_ops(cudaError_t err, const char* operation) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error during %s: %s\n", operation, cudaGetErrorString(err));
        if (err == cudaErrorMemoryAllocation) return CUDA_ERROR_FFI_ALLOC;
        if (err == cudaErrorInvalidValue) return CUDA_ERROR_FFI_INVALID_ARGS;
        return CUDA_ERROR_FFI_KERNEL_LAUNCH;
    }
    return CUDA_SUCCESS_FFI;
}

// Generic helper function to execute a binary vector operation kernel
static CudaFFIErrorCode execute_binary_vector_op(const uint64_t* h_vec_a, const uint64_t* h_vec_b, uint64_t* h_vec_res, int n,
                                                 void (*kernel_launcher)(const uint64_t*, const uint64_t*, uint64_t*, int, int, int),
                                                 const char* op_name) {
    if (!h_vec_a || !h_vec_b || !h_vec_res || n <= 0) {
        return CUDA_ERROR_FFI_INVALID_ARGS;
    }

    uint64_t* d_vec_a = nullptr;
    uint64_t* d_vec_b = nullptr;
    uint64_t* d_vec_res = nullptr;
    CudaFFIErrorCode ffi_err_code = CUDA_SUCCESS_FFI;
    cudaError_t cuda_err;

    size_t size = n * sizeof(uint64_t);

    cuda_err = cudaMalloc((void**)&d_vec_a, size);
    if ((ffi_err_code = check_cuda_error_field_ops(cuda_err, "cudaMalloc d_vec_a")) != CUDA_SUCCESS_FFI) goto cleanup;
    cuda_err = cudaMalloc((void**)&d_vec_b, size);
    if ((ffi_err_code = check_cuda_error_field_ops(cuda_err, "cudaMalloc d_vec_b")) != CUDA_SUCCESS_FFI) goto cleanup;
    cuda_err = cudaMalloc((void**)&d_vec_res, size);
    if ((ffi_err_code = check_cuda_error_field_ops(cuda_err, "cudaMalloc d_vec_res")) != CUDA_SUCCESS_FFI) goto cleanup;

    cuda_err = cudaMemcpy(d_vec_a, h_vec_a, size, cudaMemcpyHostToDevice);
    if ((ffi_err_code = check_cuda_error_field_ops(cuda_err, "cudaMemcpy h_vec_a to d_vec_a")) != CUDA_SUCCESS_FFI) goto cleanup;
    cuda_err = cudaMemcpy(d_vec_b, h_vec_b, size, cudaMemcpyHostToDevice);
    if ((ffi_err_code = check_cuda_error_field_ops(cuda_err, "cudaMemcpy h_vec_b to d_vec_b")) != CUDA_SUCCESS_FFI) goto cleanup;

    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    kernel_launcher(d_vec_a, d_vec_b, d_vec_res, n, blocks_per_grid, threads_per_block);
    cuda_err = cudaGetLastError();
    if ((ffi_err_code = check_cuda_error_field_ops(cuda_err, op_name)) != CUDA_SUCCESS_FFI) goto cleanup;
    cuda_err = cudaDeviceSynchronize();
    if ((ffi_err_code = check_cuda_error_field_ops(cuda_err, "cudaDeviceSynchronize")) != CUDA_SUCCESS_FFI) goto cleanup;

    cuda_err = cudaMemcpy(h_vec_res, d_vec_res, size, cudaMemcpyDeviceToHost);
    if ((ffi_err_code = check_cuda_error_field_ops(cuda_err, "cudaMemcpy d_vec_res to h_vec_res")) != CUDA_SUCCESS_FFI) goto cleanup;

cleanup:
    if (d_vec_a) cudaFree(d_vec_a);
    if (d_vec_b) cudaFree(d_vec_b);
    if (d_vec_res) cudaFree(d_vec_res);
    return ffi_err_code;
}

// Kernel launcher functions to match the signature required by execute_binary_vector_op
void launch_vector_field_add(const uint64_t* d_a, const uint64_t* d_b, uint64_t* d_res, int n, int blocks, int threads) {
    vector_field_add_kernel<<<blocks, threads>>>(d_a, d_b, d_res, n);
}

void launch_vector_field_mul(const uint64_t* d_a, const uint64_t* d_b, uint64_t* d_res, int n, int blocks, int threads) {
    vector_field_mul_kernel<<<blocks, threads>>>(d_a, d_b, d_res, n);
}

void launch_vector_field_pow(const uint64_t* d_base, const uint64_t* d_exp, uint64_t* d_res, int n, int blocks, int threads) {
    vector_field_pow_kernel<<<blocks, threads>>>(d_base, d_exp, d_res, n);
}

extern "C" CudaFFIErrorCode cuda_vector_field_add(const uint64_t* h_vec_a, const uint64_t* h_vec_b, uint64_t* h_vec_res, int n) {
    return execute_binary_vector_op(h_vec_a, h_vec_b, h_vec_res, n, launch_vector_field_add, "vector_field_add_kernel launch");
}

extern "C" CudaFFIErrorCode cuda_vector_field_mul(const uint64_t* h_vec_a, const uint64_t* h_vec_b, uint64_t* h_vec_res, int n) {
    return execute_binary_vector_op(h_vec_a, h_vec_b, h_vec_res, n, launch_vector_field_mul, "vector_field_mul_kernel launch");
}

extern "C" CudaFFIErrorCode cuda_vector_field_pow(const uint64_t* h_vec_base, const uint64_t* h_vec_exp, uint64_t* h_vec_res, int n) {
    return execute_binary_vector_op(h_vec_base, h_vec_exp, h_vec_res, n, launch_vector_field_pow, "vector_field_pow_kernel launch");
}


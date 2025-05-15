// hash_ops_ffi.cu - FFI implementation for Hashing Operations CUDA module

#include "hash_ops_ffi.h"
#include "hash_ops.cu" // Include the actual CUDA kernel implementation
#include <cuda_runtime.h>
#include <cstdio> // For printf

// Helper to check CUDA errors (can be shared or redefined)
static CudaFFIErrorCode check_cuda_error_hash_ops(cudaError_t err, const char* operation) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error during %s: %s\n", operation, cudaGetErrorString(err));
        if (err == cudaErrorMemoryAllocation) return CUDA_ERROR_FFI_ALLOC;
        if (err == cudaErrorInvalidValue) return CUDA_ERROR_FFI_INVALID_ARGS;
        return CUDA_ERROR_FFI_KERNEL_LAUNCH;
    }
    return CUDA_SUCCESS_FFI;
}

extern "C" CudaFFIErrorCode cuda_sha_hash(const uint8_t* h_input_data, uint32_t input_len, uint32_t* h_output_hash) {
    if (!h_input_data || input_len == 0 || !h_output_hash) {
        return CUDA_ERROR_FFI_INVALID_ARGS;
    }

    uint8_t* d_input_data = nullptr;
    uint32_t* d_output_hash = nullptr;
    CudaFFIErrorCode ffi_err_code = CUDA_SUCCESS_FFI;
    cudaError_t cuda_err;

    // Assuming SHA256, output is 32 bytes (8 * uint32_t)
    size_t output_size_bytes = 8 * sizeof(uint32_t);

    cuda_err = cudaMalloc((void**)&d_input_data, input_len * sizeof(uint8_t));
    if ((ffi_err_code = check_cuda_error_hash_ops(cuda_err, "cudaMalloc d_input_data (SHA)")) != CUDA_SUCCESS_FFI) goto cleanup;
    cuda_err = cudaMalloc((void**)&d_output_hash, output_size_bytes);
    if ((ffi_err_code = check_cuda_error_hash_ops(cuda_err, "cudaMalloc d_output_hash (SHA)")) != CUDA_SUCCESS_FFI) goto cleanup;

    cuda_err = cudaMemcpy(d_input_data, h_input_data, input_len * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if ((ffi_err_code = check_cuda_error_hash_ops(cuda_err, "cudaMemcpy h_input_data to d_input_data (SHA)")) != CUDA_SUCCESS_FFI) goto cleanup;

    // Kernel launch parameters for SHA (highly dependent on actual kernel implementation)
    // For the placeholder, we launch a single thread.
    int threads_per_block = 1;
    int blocks_per_grid = 1;

    sha_kernel<<<blocks_per_grid, threads_per_block>>>(d_input_data, input_len, d_output_hash);
    cuda_err = cudaGetLastError();
    if ((ffi_err_code = check_cuda_error_hash_ops(cuda_err, "sha_kernel launch")) != CUDA_SUCCESS_FFI) goto cleanup;
    cuda_err = cudaDeviceSynchronize();
    if ((ffi_err_code = check_cuda_error_hash_ops(cuda_err, "cudaDeviceSynchronize after sha_kernel")) != CUDA_SUCCESS_FFI) goto cleanup;

    cuda_err = cudaMemcpy(h_output_hash, d_output_hash, output_size_bytes, cudaMemcpyDeviceToHost);
    if ((ffi_err_code = check_cuda_error_hash_ops(cuda_err, "cudaMemcpy d_output_hash to h_output_hash (SHA)")) != CUDA_SUCCESS_FFI) goto cleanup;

cleanup:
    if (d_input_data) cudaFree(d_input_data);
    if (d_output_hash) cudaFree(d_output_hash);
    return ffi_err_code;
}

extern "C" CudaFFIErrorCode cuda_tip5_hash(const uint64_t* h_input_state, uint64_t* h_output_state, int state_len) {
    if (!h_input_state || !h_output_state || state_len <= 0) {
        return CUDA_ERROR_FFI_INVALID_ARGS;
    }
    // Assuming TIP5 state_len is fixed, e.g., 5 for some sponge constructions.
    // Add validation for state_len if necessary.

    uint64_t* d_input_state = nullptr;
    uint64_t* d_output_state = nullptr;
    CudaFFIErrorCode ffi_err_code = CUDA_SUCCESS_FFI;
    cudaError_t cuda_err;

    size_t state_size_bytes = state_len * sizeof(uint64_t);

    cuda_err = cudaMalloc((void**)&d_input_state, state_size_bytes);
    if ((ffi_err_code = check_cuda_error_hash_ops(cuda_err, "cudaMalloc d_input_state (TIP5)")) != CUDA_SUCCESS_FFI) goto cleanup;
    cuda_err = cudaMalloc((void**)&d_output_state, state_size_bytes);
    if ((ffi_err_code = check_cuda_error_hash_ops(cuda_err, "cudaMalloc d_output_state (TIP5)")) != CUDA_SUCCESS_FFI) goto cleanup;

    cuda_err = cudaMemcpy(d_input_state, h_input_state, state_size_bytes, cudaMemcpyHostToDevice);
    if ((ffi_err_code = check_cuda_error_hash_ops(cuda_err, "cudaMemcpy h_input_state to d_input_state (TIP5)")) != CUDA_SUCCESS_FFI) goto cleanup;

    // Kernel launch parameters for TIP5 (highly dependent on actual kernel implementation)
    // For the placeholder, we launch a single thread.
    int threads_per_block = 1;
    int blocks_per_grid = 1;

    tip5_kernel<<<blocks_per_grid, threads_per_block>>>(d_input_state, d_output_state);
    cuda_err = cudaGetLastError();
    if ((ffi_err_code = check_cuda_error_hash_ops(cuda_err, "tip5_kernel launch")) != CUDA_SUCCESS_FFI) goto cleanup;
    cuda_err = cudaDeviceSynchronize();
    if ((ffi_err_code = check_cuda_error_hash_ops(cuda_err, "cudaDeviceSynchronize after tip5_kernel")) != CUDA_SUCCESS_FFI) goto cleanup;

    cuda_err = cudaMemcpy(h_output_state, d_output_state, state_size_bytes, cudaMemcpyDeviceToHost);
    if ((ffi_err_code = check_cuda_error_hash_ops(cuda_err, "cudaMemcpy d_output_state to h_output_state (TIP5)")) != CUDA_SUCCESS_FFI) goto cleanup;

cleanup:
    if (d_input_state) cudaFree(d_input_state);
    if (d_output_state) cudaFree(d_output_state);
    return ffi_err_code;
}


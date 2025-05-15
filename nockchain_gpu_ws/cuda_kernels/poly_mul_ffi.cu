// poly_mul_ffi.cu - FFI implementation for polynomial multiplication CUDA module

#include "poly_mul_ffi.h"
#include "poly_mul.cu" // Include the actual CUDA kernel implementation
#include <cuda_runtime.h>
#include <cstdio> // For printf in case of errors

// Helper to check CUDA errors and map to FFI error codes
static CudaFFIErrorCode check_cuda_error(cudaError_t err, const char* operation) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error during %s: %s\n", operation, cudaGetErrorString(err));
        // Map specific CUDA errors to FFI errors if needed, otherwise a general one
        if (err == cudaErrorMemoryAllocation) return CUDA_ERROR_FFI_ALLOC;
        if (err == cudaErrorInvalidValue) return CUDA_ERROR_FFI_INVALID_ARGS;
        // Add more mappings as necessary
        return CUDA_ERROR_FFI_KERNEL_LAUNCH; // General kernel/runtime error
    }
    return CUDA_SUCCESS_FFI;
}

extern "C" CudaFFIErrorCode cuda_poly_mul(const uint64_t* h_poly_a, int len_a,
                                          const uint64_t* h_poly_b, int len_b,
                                          uint64_t* h_poly_res, int len_res) {
    if (!h_poly_a || !h_poly_b || !h_poly_res || len_a <= 0 || len_b <= 0) {
        return CUDA_ERROR_FFI_INVALID_ARGS;
    }
    if (len_res != len_a + len_b - 1 && !(len_a == 1 && len_b == 1 && len_res ==1)) { // handle 1x1 case for testing
         if (len_res != len_a + len_b -1) {
            fprintf(stderr, "Error: Result polynomial length is incorrect. Expected %d, got %d\n", len_a + len_b - 1, len_res);
            return CUDA_ERROR_FFI_INVALID_ARGS;
        }
    }

    uint64_t* d_poly_a = nullptr;
    uint64_t* d_poly_b = nullptr;
    uint64_t* d_poly_res = nullptr;
    CudaFFIErrorCode ffi_err_code = CUDA_SUCCESS_FFI;
    cudaError_t cuda_err;

    // 1. Allocate memory on the device
    cuda_err = cudaMalloc((void**)&d_poly_a, len_a * sizeof(uint64_t));
    if ((ffi_err_code = check_cuda_error(cuda_err, "cudaMalloc d_poly_a")) != CUDA_SUCCESS_FFI) goto cleanup;
    cuda_err = cudaMalloc((void**)&d_poly_b, len_b * sizeof(uint64_t));
    if ((ffi_err_code = check_cuda_error(cuda_err, "cudaMalloc d_poly_b")) != CUDA_SUCCESS_FFI) goto cleanup;
    cuda_err = cudaMalloc((void**)&d_poly_res, len_res * sizeof(uint64_t));
    if ((ffi_err_code = check_cuda_error(cuda_err, "cudaMalloc d_poly_res")) != CUDA_SUCCESS_FFI) goto cleanup;

    // 2. Copy data from host to device
    cuda_err = cudaMemcpy(d_poly_a, h_poly_a, len_a * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if ((ffi_err_code = check_cuda_error(cuda_err, "cudaMemcpy h_poly_a to d_poly_a")) != CUDA_SUCCESS_FFI) goto cleanup;
    cuda_err = cudaMemcpy(d_poly_b, h_poly_b, len_b * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if ((ffi_err_code = check_cuda_error(cuda_err, "cudaMemcpy h_poly_b to d_poly_b")) != CUDA_SUCCESS_FFI) goto cleanup;

    // 3. Define kernel launch parameters
    // For poly_mul_kernel, each thread calculates one coefficient of the result.
    // The number of threads should be equal to len_res.
    int threads_per_block = 256;
    int blocks_per_grid = (len_res + threads_per_block - 1) / threads_per_block;

    // 4. Launch the kernel
    poly_mul_kernel<<<blocks_per_grid, threads_per_block>>>(d_poly_a, len_a, d_poly_b, len_b, d_poly_res, len_res);
    cuda_err = cudaGetLastError(); // Check for errors during kernel launch
    if ((ffi_err_code = check_cuda_error(cuda_err, "poly_mul_kernel launch")) != CUDA_SUCCESS_FFI) goto cleanup;
    
    // Synchronize to ensure kernel completion before copying back (optional for cudaMemcpyDeviceToHost if not on default stream and using events)
    cuda_err = cudaDeviceSynchronize();
    if ((ffi_err_code = check_cuda_error(cuda_err, "cudaDeviceSynchronize after kernel")) != CUDA_SUCCESS_FFI) goto cleanup;

    // 5. Copy data from device to host
    cuda_err = cudaMemcpy(h_poly_res, d_poly_res, len_res * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    if ((ffi_err_code = check_cuda_error(cuda_err, "cudaMemcpy d_poly_res to h_poly_res")) != CUDA_SUCCESS_FFI) goto cleanup;

cleanup:
    // 6. Free device memory
    if (d_poly_a) cudaFree(d_poly_a);
    if (d_poly_b) cudaFree(d_poly_b);
    if (d_poly_res) cudaFree(d_poly_res);

    return ffi_err_code;
}


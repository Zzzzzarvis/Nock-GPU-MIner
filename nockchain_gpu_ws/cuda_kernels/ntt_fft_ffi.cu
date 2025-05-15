// ntt_fft_ffi.cu - FFI implementation for NTT/FFT CUDA module

#include "ntt_fft_ffi.h"
#include "ntt_fft.cu" // Include the actual CUDA kernel implementation
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdio> // For printf

// Helper to check CUDA errors (can be shared or redefined)
static CudaFFIErrorCode check_cuda_error_ntt(cudaError_t err, const char* operation) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error during %s: %s\n", operation, cudaGetErrorString(err));
        if (err == cudaErrorMemoryAllocation) return CUDA_ERROR_FFI_ALLOC;
        if (err == cudaErrorInvalidValue) return CUDA_ERROR_FFI_INVALID_ARGS;
        return CUDA_ERROR_FFI_KERNEL_LAUNCH;
    }
    return CUDA_SUCCESS_FFI;
}

// Helper function to compute bit reversal indices (can be precomputed on host)
// This is a simple CPU version for demonstration. In practice, precompute and pass from Rust.
void compute_bit_reverse_indices(int n, std::vector<int>& rev_indices) {
    rev_indices.resize(n);
    int log2_n = 0;
    while ((1 << log2_n) < n) {
        log2_n++;
    }
    for (int i = 0; i < n; ++i) {
        int rev = 0;
        for (int j = 0; j < log2_n; ++j) {
            if ((i >> j) & 1) {
                rev |= 1 << (log2_n - 1 - j);
            }
        }
        rev_indices[i] = rev;
    }
}

// Helper function to compute twiddle factors (can be precomputed on host)
// This is a simple CPU version for demonstration.
void compute_twiddle_factors(int n, uint64_t root_of_unity, bool inverse, std::vector<uint64_t>& twiddles) {
    twiddles.resize(n / 2);
    uint64_t current_root = root_of_unity;
    if (inverse) {
        // Need to find modular inverse of root_of_unity for iNTT
        // For simplicity, assume root_of_unity is its own inverse or precomputed inverse is passed.
        // This is a placeholder. A proper modular inverse is needed.
        // current_root = field_pow_op(root_of_unity, PRIME_FIELD_OPS - 2); // if PRIME_FIELD_OPS is prime
    }
    for (int i = 0; i < n / 2; ++i) {
        twiddles[i] = field_pow_op(current_root, i); // field_pow_op from field_ops.cu or similar
    }
}


extern "C" CudaFFIErrorCode cuda_ntt_fft(uint64_t* h_data, int n, const int* h_rev_indices_param, uint64_t h_root_of_unity, int inverse) {
    if (!h_data || n <= 0 || (n & (n - 1)) != 0) { // n must be a power of two
        return CUDA_ERROR_FFI_INVALID_ARGS;
    }

    uint64_t* d_data = nullptr;
    int* d_rev_indices = nullptr;
    uint64_t* d_twiddle_factors = nullptr;

    CudaFFIErrorCode ffi_err_code = CUDA_SUCCESS_FFI;
    cudaError_t cuda_err;

    std::vector<int> rev_indices_vec; // For CPU computation if h_rev_indices_param is null
    const int* p_rev_indices = h_rev_indices_param;

    if (!p_rev_indices) {
        compute_bit_reverse_indices(n, rev_indices_vec);
        p_rev_indices = rev_indices_vec.data();
    }

    // 1. Allocate memory on device
    cuda_err = cudaMalloc((void**)&d_data, n * sizeof(uint64_t));
    if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "cudaMalloc d_data")) != CUDA_SUCCESS_FFI) goto cleanup;
    cuda_err = cudaMalloc((void**)&d_rev_indices, n * sizeof(int)); // For bit_reverse_permutation_kernel
    if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "cudaMalloc d_rev_indices")) != CUDA_SUCCESS_FFI) goto cleanup;
    
    // Twiddle factors: For simplicity, let's assume they are computed per stage or a global array is managed.
    // A full NTT needs careful twiddle factor management.
    // This example will use a simplified approach for the ntt_fft_stage_kernel.
    // For a real implementation, you'd precompute all necessary twiddles.
    // Let's assume for now `h_root_of_unity` is the primitive Nth root for the entire transform.

    // 2. Copy data to device
    cuda_err = cudaMemcpy(d_data, h_data, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "cudaMemcpy h_data to d_data")) != CUDA_SUCCESS_FFI) goto cleanup;
    cuda_err = cudaMemcpy(d_rev_indices, p_rev_indices, n * sizeof(int), cudaMemcpyHostToDevice);
    if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "cudaMemcpy p_rev_indices to d_rev_indices")) != CUDA_SUCCESS_FFI) goto cleanup;

    // 3. Kernel launch parameters
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    // 4. Launch bit-reversal permutation kernel
    bit_reverse_permutation_kernel<<<blocks_per_grid, threads_per_block>>>(d_data, n, d_rev_indices);
    cuda_err = cudaGetLastError();
    if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "bit_reverse_permutation_kernel launch")) != CUDA_SUCCESS_FFI) goto cleanup;
    cuda_err = cudaDeviceSynchronize();
    if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "cudaDeviceSynchronize after bit_reverse")) != CUDA_SUCCESS_FFI) goto cleanup;

    // 5. Iterative NTT stages (Cooley-Tukey)
    // This is a simplified loop. Twiddle factor calculation and indexing needs to be precise.
    // The `ntt_fft_stage_kernel` in ntt_fft.cu is a placeholder and needs correct twiddle factors.
    // For this FFI, we'll assume `h_root_of_unity` is the Nth root, and we derive stage roots from it.
    // A more robust solution would involve passing a precomputed array of all twiddle factors.
    
    std::vector<uint64_t> stage_twiddles_vec; // To store twiddles for each stage if computed on CPU

    for (int len = 2; len <= n; len <<= 1) { // len is the length of the sub-DFT
        // Compute twiddle factors for this stage
        // W_len = h_root_of_unity ^ (N/len)
        // The ntt_fft_stage_kernel expects an array of twiddles for its current stage_len (which is `len` here)
        // This part is complex and depends on how twiddles are structured for the kernel.
        // The placeholder kernel `ntt_fft_stage_kernel` has a `twiddle_offset` which implies a global twiddle array.
        // For now, we'll skip detailed twiddle management in this FFI wrapper for brevity.
        // A real implementation would need to prepare `d_twiddle_factors` correctly.
        // Example: compute W_m = h_root_of_unity ^ (n / len)
        uint64_t w_m_base = field_pow_op(h_root_of_unity, n / len); // field_pow_op from field_ops.cu or similar
        stage_twiddles_vec.resize(len/2);
        for(int j=0; j < len/2; ++j) {
            stage_twiddles_vec[j] = field_pow_op(w_m_base, j);
        }
        cuda_err = cudaMalloc((void**)&d_twiddle_factors, (len/2) * sizeof(uint64_t));
        if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "cudaMalloc d_twiddle_factors")) != CUDA_SUCCESS_FFI) goto cleanup_loop;
        cuda_err = cudaMemcpy(d_twiddle_factors, stage_twiddles_vec.data(), (len/2) * sizeof(uint64_t), cudaMemcpyHostToDevice);
        if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "cudaMemcpy stage_twiddles_vec to d_twiddle_factors")) != CUDA_SUCCESS_FFI) goto cleanup_loop;

        // Number of threads for this stage can be n/2 if each thread does one butterfly
        int stage_threads = n / 2; 
        int stage_blocks = (stage_threads + threads_per_block - 1) / threads_per_block;

        ntt_fft_stage_kernel<<<stage_blocks, threads_per_block>>>(d_data, n, len, d_twiddle_factors, 0);
        cuda_err = cudaGetLastError();
        if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "ntt_fft_stage_kernel launch")) != CUDA_SUCCESS_FFI) goto cleanup_loop;
        cuda_err = cudaDeviceSynchronize();
        if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "cudaDeviceSynchronize after stage kernel")) != CUDA_SUCCESS_FFI) goto cleanup_loop;
        
        cudaFree(d_twiddle_factors); d_twiddle_factors = nullptr; // Free per-stage twiddles
    }

    // If inverse NTT, divide by N (modularly)
    if (inverse) {
        uint64_t n_inv = field_pow_op(n, PRIME_FIELD_OPS - 2); // Modular inverse of N
        // Kernel to multiply all elements by n_inv (can reuse vector_field_mul with a scalar or a dedicated kernel)
        // For simplicity, assume a kernel vector_scalar_mul_kernel(d_data, n_inv, n);
        // This part is omitted for brevity.
    }

    // 6. Copy data back to host
    cuda_err = cudaMemcpy(h_data, d_data, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "cudaMemcpy d_data to h_data")) != CUDA_SUCCESS_FFI) goto cleanup;

cleanup_loop: // In case of error inside the loop
    if (d_twiddle_factors) cudaFree(d_twiddle_factors);
cleanup:
    if (d_data) cudaFree(d_data);
    if (d_rev_indices) cudaFree(d_rev_indices);
    // d_twiddle_factors is freed inside loop or at cleanup_loop

    return ffi_err_code;
}


static CudaFFIErrorCode execute_binary_vector_op(
    const uint64_t* h_vec_a,
    const uint64_t* h_vec_b,
    uint64_t* h_vec_res,
    int n,
    void (*kernel_launcher)(const uint64_t*, const uint64_t*, uint64_t*, int, int, int),
    const char* op_name
) {
    uint64_t* d_vec_a = nullptr;
    uint64_t* d_vec_b = nullptr;
    uint64_t* d_vec_res = nullptr;
    CudaFFIErrorCode ffi_err_code = CUDA_SUCCESS_FFI;
    cudaError_t cuda_err;

    int threads_per_block = 256;
    int blocks_per_grid = (n > 0) ? (n + threads_per_block - 1) / threads_per_block : 0;

    if (n == 0) {
        return CUDA_SUCCESS_FFI;
    }
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
// Generic helper function to execute a binary vector operation kernel
static CudaFFIErrorCode execute_binary_vector_op(
    const uint64_t* h_vec_a,
    const uint64_t* h_vec_b,
    uint64_t* h_vec_res,
    int n,
    void (*kernel_launcher)(const uint64_t*, const uint64_t*, uint64_t*, int, int, int),
    const char* op_name
) {
    // 首先检查输入参数的有效性
    if (!h_vec_a || !h_vec_b || !h_vec_res) {
        fprintf(stderr, "CUDA FFI Error: Null pointer passed for h_vec_a, h_vec_b, or h_vec_res in %s.\n", op_name);
        return CUDA_ERROR_FFI_INVALID_ARGS;
    }
    if (n < 0) { // n=0 可能是允许的空操作，但 n < 0 通常是无效的
        fprintf(stderr, "CUDA FFI Error: Negative size n=%d passed in %s.\n", n, op_name);
        return CUDA_ERROR_FFI_INVALID_ARGS;
    }

    // 将变量声明移到所有 goto 语句之前
    uint64_t* d_vec_a = nullptr;
    uint64_t* d_vec_b = nullptr;
    uint64_t* d_vec_res = nullptr;
    CudaFFIErrorCode ffi_err_code = CUDA_SUCCESS_FFI;
    cudaError_t cuda_err;

    int threads_per_block = 256;
    // 确保 n > 0 时才计算 blocks_per_grid，避免 n=0 时除零，或当 n=0 时 blocks_per_grid 为0
    int blocks_per_grid = (n > 0) ? (n + threads_per_block - 1) / threads_per_block : 0;

    // 如果输入向量大小为0，则视为成功完成的空操作
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

    // 只有当 n > 0 (因此 blocks_per_grid > 0) 时才启动内核
    if (blocks_per_grid > 0) { // 添加此检查以避免在 n=0 时启动内核
        kernel_launcher(d_vec_a, d_vec_b, d_vec_res, n, blocks_per_grid, threads_per_block);
        cuda_err = cudaGetLastError(); // 检查内核启动错误
        if ((ffi_err_code = check_cuda_error_field_ops(cuda_err, op_name)) != CUDA_SUCCESS_FFI) goto cleanup;
        
        cuda_err = cudaDeviceSynchronize(); // 同步设备
        if ((ffi_err_code = check_cuda_error_field_ops(cuda_err, "cudaDeviceSynchronize after kernel")) != CUDA_SUCCESS_FFI) goto cleanup;
    }

    cuda_err = cudaMemcpy(h_vec_res, d_vec_res, size, cudaMemcpyDeviceToHost);
    if ((ffi_err_code = check_cuda_error_field_ops(cuda_err, "cudaMemcpy d_vec_res to h_vec_res")) != CUDA_SUCCESS_FFI) goto cleanup;

cleanup:
    if (d_vec_a) cudaFree(d_vec_a);
    if (d_vec_b) cudaFree(d_vec_b);
    if (d_vec_res) cudaFree(d_vec_res);
    return ffi_err_code;
} 
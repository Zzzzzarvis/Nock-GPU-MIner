// field_ops_ffi.h - FFI header for Finite Field Operations CUDA module

#ifndef FIELD_OPS_FFI_H
#define FIELD_OPS_FFI_H

#include <cstdint>
#include "poly_mul_ffi.h" // For CudaFFIErrorCode, or define it commonly

#ifdef __cplusplus
extern "C" {
#endif

// FFI-safe function for element-wise vector addition in the field
CudaFFIErrorCode cuda_vector_field_add(const uint64_t* h_vec_a, const uint64_t* h_vec_b, uint64_t* h_vec_res, int n);

// FFI-safe function for element-wise vector multiplication in the field
CudaFFIErrorCode cuda_vector_field_mul(const uint64_t* h_vec_a, const uint64_t* h_vec_b, uint64_t* h_vec_res, int n);

// FFI-safe function for element-wise vector exponentiation in the field
// Note: Exponent might be a single value or an array. This example assumes an array of exponents.
CudaFFIErrorCode cuda_vector_field_pow(const uint64_t* h_vec_base, const uint64_t* h_vec_exp, uint64_t* h_vec_res, int n);


#ifdef __cplusplus
}
#endif

#endif // FIELD_OPS_FFI_H


// poly_mul_ffi.h - FFI header for polynomial multiplication CUDA module

#ifndef POLY_MUL_FFI_H
#define POLY_MUL_FFI_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// FFI-safe struct to pass polynomial data (if needed, or use raw pointers)
// typedef struct {
//     uint64_t* coeffs;
//     int length;
// } FFIPolynomial;

// Error codes to be returned by FFI functions
typedef enum {
    CUDA_SUCCESS_FFI = 0,
    CUDA_ERROR_FFI_ALLOC = 1,
    CUDA_ERROR_FFI_MEMCPY = 2,
    CUDA_ERROR_FFI_KERNEL_LAUNCH = 3,
    CUDA_ERROR_FFI_INVALID_ARGS = 4,
    // Add more specific error codes as needed
} CudaFFIErrorCode;

/**
 * @brief Launches the CUDA kernel for polynomial multiplication.
 *
 * Multiplies two polynomials poly_a and poly_b and stores the result in poly_res.
 * All polynomials are represented as arrays of uint64_t coefficients.
 * The caller is responsible for allocating and deallocating all host and device memory
 * appropriately, or this function can handle device memory internally.
 *
 * @param h_poly_a Pointer to the host array неуда coefficients of the first polynomial.
 * @param len_a Length of the first polynomial.
 * @param h_poly_b Pointer to the host array of coefficients of the second polynomial.
 * @param len_b Length of the second polynomial.
 * @param h_poly_res Pointer to the host array where the result polynomial coefficients will be stored.
 *                 Its length must be len_a + len_b - 1.
 * @param len_res Length of the result polynomial (should be len_a + len_b - 1).
 * @return CudaFFIErrorCode indicating success or failure.
 */
CudaFFIErrorCode cuda_poly_mul(const uint64_t* h_poly_a, int len_a,
                               const uint64_t* h_poly_b, int len_b,
                               uint64_t* h_poly_res, int len_res);

#ifdef __cplusplus
}
#endif

#endif // POLY_MUL_FFI_H


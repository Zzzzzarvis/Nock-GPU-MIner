// ntt_fft_ffi.h - FFI header for NTT/FFT CUDA module

#ifndef NTT_FFT_FFI_H
#define NTT_FFT_FFI_H

#include <cstdint>
#include "poly_mul_ffi.h" // For CudaFFIErrorCode, or define it commonly

#ifdef __cplusplus
extern "C" {
#endif

// FFI-safe function to perform NTT/FFT on the given data.
// The caller is responsible for pre-calculating bit-reversal indices and twiddle factors if needed by the kernel.
// This is a simplified interface; a real one might need more parameters for twiddle factors, roots of unity, etc.
/**
 * @brief Launches CUDA kernels for NTT/FFT processing.
 *
 * @param h_data Pointer to the host array of data to be transformed (in-place or out-of-place based on implementation).
 * @param n Length of the data array (must be a power of two).
 * @param h_rev_indices Pointer to pre-calculated bit-reversal indices for permutation (if applicable).
 * @param h_root_of_unity The N-th root of unity for the transformation.
 * @param inverse If non-zero, performs an inverse NTT/FFT.
 * @return CudaFFIErrorCode indicating success or failure.
 */
CudaFFIErrorCode cuda_ntt_fft(uint64_t* h_data, int n, const int* h_rev_indices, uint64_t h_root_of_unity, int inverse);

#ifdef __cplusplus
}
#endif

#endif // NTT_FFT_FFI_H


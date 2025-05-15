// hash_ops_ffi.h - FFI header for Hashing Operations CUDA module

#ifndef HASH_OPS_FFI_H
#define HASH_OPS_FFI_H

#include <cstdint>
#include "poly_mul_ffi.h" // For CudaFFIErrorCode, or define it commonly

#ifdef __cplusplus
extern "C" {
#endif

// FFI-safe function for SHA hashing (e.g., SHA256)
// output_hash should be pre-allocated by the caller to the correct size (e.g., 32 bytes for SHA256)
CudaFFIErrorCode cuda_sha_hash(const uint8_t* h_input_data, uint32_t input_len, uint32_t* h_output_hash);

// FFI-safe function for TIP5 hashing
// input_state and output_state are arrays representing the TIP5 state (e.g., 5 uint64_t values)
// The length of the state should be known and consistent (e.g., passed as a parameter or fixed).
CudaFFIErrorCode cuda_tip5_hash(const uint64_t* h_input_state, uint64_t* h_output_state, int state_len);


#ifdef __cplusplus
}
#endif

#endif // HASH_OPS_FFI_H


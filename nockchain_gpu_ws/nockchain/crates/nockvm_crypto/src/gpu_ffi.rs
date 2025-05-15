// src/gpu_ffi.rs (or a suitable module path within nockvm_crypto)

use std::os::raw::{c_int, c_uchar, c_uint};
use std::ffi::c_void; // For potential opaque pointers if needed later

// Mirror the CudaFFIErrorCode enum from the C++ FFI headers
#[repr(C)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum CudaFFIErrorCode {
    Success = 0,
    ErrorAlloc = 1,
    ErrorMemcpy = 2,
    ErrorKernelLaunch = 3,
    ErrorInvalidArgs = 4,
    // Add more as defined in the C++ headers if they expand
}

impl CudaFFIErrorCode {
    pub fn is_success(&self) -> bool {
        *self == CudaFFIErrorCode::Success
    }
    pub fn to_string(&self) -> String {
        match self {
            CudaFFIErrorCode::Success => "CUDA Success".to_string(),
            CudaFFIErrorCode::ErrorAlloc => "CUDA Error: Memory Allocation Failed".to_string(),
            CudaFFIErrorCode::ErrorMemcpy => "CUDA Error: Memory Copy Failed".to_string(),
            CudaFFIErrorCode::ErrorKernelLaunch => "CUDA Error: Kernel Launch Failed".to_string(),
            CudaFFIErrorCode::ErrorInvalidArgs => "CUDA Error: Invalid Arguments".to_string(),
        }
    }
}

// FFI bindings to the C++/CUDA library (libnockchain_gpu_kernels.a)
#[link(name = "nockchain_gpu_kernels", kind = "static")] // Name from build.rs
extern "C" {
    // From poly_mul_ffi.h
    pub fn cuda_poly_mul(
        h_poly_a: *const u64,
        len_a: c_int,
        h_poly_b: *const u64,
        len_b: c_int,
        h_poly_res: *mut u64,
        len_res: c_int,
    ) -> CudaFFIErrorCode;

    // From ntt_fft_ffi.h
    pub fn cuda_ntt_fft(
        h_data: *mut u64, 
        n: c_int,
        h_rev_indices: *const c_int, 
        h_root_of_unity: u64,
        inverse: c_int, 
    ) -> CudaFFIErrorCode;

    // From field_ops_ffi.h
    pub fn cuda_vector_field_add(
        h_vec_a: *const u64,
        h_vec_b: *const u64,
        h_vec_res: *mut u64,
        n: c_int,
    ) -> CudaFFIErrorCode;

    pub fn cuda_vector_field_mul(
        h_vec_a: *const u64,
        h_vec_b: *const u64,
        h_vec_res: *mut u64,
        n: c_int,
    ) -> CudaFFIErrorCode;

    pub fn cuda_vector_field_pow(
        h_vec_base: *const u64,
        h_vec_exp: *const u64, 
        h_vec_res: *mut u64,
        n: c_int,
    ) -> CudaFFIErrorCode;

    // From hash_ops_ffi.h
    pub fn cuda_sha_hash(
        h_input_data: *const u8, 
        input_len: u32,          
        h_output_hash: *mut u32, 
    ) -> CudaFFIErrorCode;

    pub fn cuda_tip5_hash(
        h_input_state: *const u64,
        h_output_state: *mut u64,
        state_len: c_int,
    ) -> CudaFFIErrorCode;
}

// --- Safe Rust Wrappers ---

fn handle_cuda_result(status: CudaFFIErrorCode, operation_name: &str) -> Result<(), String> {
    if status.is_success() {
        Ok(())
    } else {
        Err(format!("{} failed: {}", operation_name, status.to_string()))
    }
}

pub fn gpu_poly_mul(poly_a: &[u64], poly_b: &[u64]) -> Result<Vec<u64>, String> {
    let len_a = poly_a.len();
    let len_b = poly_b.len();
    if len_a == 0 || len_b == 0 {
        // Consistent with how Rust's empty poly mul might work, or return error
        if len_a + len_b > 1 { // only if result is not a single 0 or empty
             return Ok(vec![0; len_a + len_b -1]);
        } else {
            return Ok(Vec::new());
        }
    }
    let len_res = len_a + len_b - 1;
    let mut res_vec: Vec<u64> = vec![0; len_res];

    let status = unsafe {
        cuda_poly_mul(
            poly_a.as_ptr(),
            len_a as c_int,
            poly_b.as_ptr(),
            len_b as c_int,
            res_vec.as_mut_ptr(),
            len_res as c_int,
        )
    };
    handle_cuda_result(status, "Polynomial Multiplication")?;
    Ok(res_vec)
}

pub fn gpu_ntt_fft(data: &mut [u64], rev_indices: Option<&[c_int]>, root_of_unity: u64, inverse: bool) -> Result<(), String> {
    let n = data.len();
    if n == 0 || (n & (n-1)) != 0 {
        return Err("NTT/FFT data length must be a power of two and non-zero.".to_string());
    }
    let rev_indices_ptr = match rev_indices {
        Some(indices) => {
            if indices.len() != n {
                return Err("Bit reversal indices length must match data length.".to_string());
            }
            indices.as_ptr()
        }
        None => std::ptr::null(), // C++ side might compute if null
    };

    let status = unsafe {
        cuda_ntt_fft(
            data.as_mut_ptr(),
            n as c_int,
            rev_indices_ptr,
            root_of_unity,
            if inverse { 1 } else { 0 },
        )
    };
    handle_cuda_result(status, "NTT/FFT")
}

pub fn gpu_vector_field_add(vec_a: &[u64], vec_b: &[u64]) -> Result<Vec<u64>, String> {
    if vec_a.len() != vec_b.len() || vec_a.is_empty() {
        return Err("Input vectors for field addition must have the same non-zero length".to_string());
    }
    let n = vec_a.len();
    let mut res_vec: Vec<u64> = vec![0; n];
    let status = unsafe {
        cuda_vector_field_add(
            vec_a.as_ptr(),
            vec_b.as_ptr(),
            res_vec.as_mut_ptr(),
            n as c_int,
        )
    };
    handle_cuda_result(status, "Vector Field Addition")?;
    Ok(res_vec)
}

pub fn gpu_vector_field_mul(vec_a: &[u64], vec_b: &[u64]) -> Result<Vec<u64>, String> {
    if vec_a.len() != vec_b.len() || vec_a.is_empty() {
        return Err("Input vectors for field multiplication must have the same non-zero length".to_string());
    }
    let n = vec_a.len();
    let mut res_vec: Vec<u64> = vec![0; n];
    let status = unsafe {
        cuda_vector_field_mul(
            vec_a.as_ptr(),
            vec_b.as_ptr(),
            res_vec.as_mut_ptr(),
            n as c_int,
        )
    };
    handle_cuda_result(status, "Vector Field Multiplication")?;
    Ok(res_vec)
}

pub fn gpu_vector_field_pow(vec_base: &[u64], vec_exp: &[u64]) -> Result<Vec<u64>, String> {
    if vec_base.len() != vec_exp.len() || vec_base.is_empty() {
        return Err("Input vectors for field exponentiation must have the same non-zero length".to_string());
    }
    let n = vec_base.len();
    let mut res_vec: Vec<u64> = vec![0; n];
    let status = unsafe {
        cuda_vector_field_pow(
            vec_base.as_ptr(),
            vec_exp.as_ptr(),
            res_vec.as_mut_ptr(),
            n as c_int,
        )
    };
    handle_cuda_result(status, "Vector Field Exponentiation")?;
    Ok(res_vec)
}

// Assuming SHA256 output is 32 bytes (8 * u32)
pub const SHA256_OUTPUT_LEN_U32: usize = 8;
pub fn gpu_sha_hash(input_data: &[u8]) -> Result<[u32; SHA256_OUTPUT_LEN_U32], String> {
    if input_data.is_empty() {
        // Or handle as an error, depending on SHA spec for empty input
        // return Err("Input data for SHA hash cannot be empty.".to_string());
    }
    let mut output_hash_arr = [0u32; SHA256_OUTPUT_LEN_U32];
    let status = unsafe {
        cuda_sha_hash(
            input_data.as_ptr(),
            input_data.len() as u32,
            output_hash_arr.as_mut_ptr(),
        )
    };
    handle_cuda_result(status, "SHA Hash")?;
    Ok(output_hash_arr)
}


pub fn gpu_tip5_hash(input_state: &[u64], state_len: usize) -> Result<Vec<u64>, String> {
    if input_state.len() != state_len || state_len == 0 {
         return Err(format!("Input state length {} must match provided state_len {} and be non-zero.", input_state.len(), state_len));
    }
    let mut output_state_vec: Vec<u64> = vec![0; state_len];
    let status = unsafe {
        cuda_tip5_hash(
            input_state.as_ptr(),
            output_state_vec.as_mut_ptr(),
            state_len as c_int,
        )
    };
    handle_cuda_result(status, "TIP5 Hash")?;
    Ok(output_state_vec)
}



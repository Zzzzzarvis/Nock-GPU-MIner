// build.rs for nockvm_crypto (or a new GPU FFI crate)

use std::env;
use std::path::PathBuf;

fn main() {
    // Get the directory of the current crate (e.g., nockvm_crypto)
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // Define the path to the CUDA kernels. This assumes a specific workspace structure.
    // Adjust this path if your cuda_kernels directory is located elsewhere relative to this build.rs.
    // For example, if cuda_kernels is in a sibling directory to the `nockchain` project root:
    // let cuda_kernels_path = crate_dir.join("..").join("..").join("nockchain_gpu_ws").join("cuda_kernels");
    // For this example, let's assume it's two levels up from `crates/nockvm_crypto` and then into `nockchain_gpu_ws`
    let base_path = crate_dir.join("..").join(".."); // Assuming this build.rs is in crates/some_crate
    let cuda_kernels_path = base_path.join("nockchain_gpu_ws").join("cuda_kernels");

    println!("cargo:rerun-if-changed={}", cuda_kernels_path.join("poly_mul.cu").display());
    println!("cargo:rerun-if-changed={}", cuda_kernels_path.join("poly_mul_ffi.h").display());
    println!("cargo:rerun-if-changed={}", cuda_kernels_path.join("poly_mul_ffi.cu").display());
    println!("cargo:rerun-if-changed={}", cuda_kernels_path.join("ntt_fft.cu").display());
    println!("cargo:rerun-if-changed={}", cuda_kernels_path.join("ntt_fft_ffi.h").display());
    println!("cargo:rerun-if-changed={}", cuda_kernels_path.join("ntt_fft_ffi.cu").display());
    println!("cargo:rerun-if-changed={}", cuda_kernels_path.join("field_ops.cu").display());
    println!("cargo:rerun-if-changed={}", cuda_kernels_path.join("field_ops_ffi.h").display());
    println!("cargo:rerun-if-changed={}", cuda_kernels_path.join("field_ops_ffi.cu").display());
    println!("cargo:rerun-if-changed={}", cuda_kernels_path.join("hash_ops.cu").display());
    println!("cargo:rerun-if-changed={}", cuda_kernels_path.join("hash_ops_ffi.h").display());
    println!("cargo:rerun-if-changed={}", cuda_kernels_path.join("hash_ops_ffi.cu").display());

    // Use the cc crate to compile the CUDA C++ FFI wrapper files.
    // NVCC will be invoked by cc if it detects .cu files or through specific flags.
    // Alternatively, use the `cudarc` crate or direct `Command` calls to nvcc.

    // For simplicity, we will use the `cc` crate and assume it can handle .cu files
    // or that nvcc is used appropriately by the system's C++ compiler configuration.
    // A more robust solution might involve explicitly calling nvcc.

    // Specify CUDA include paths and library paths
    // These might need to be configured based on the system environment (e.g., CUDA_HOME)
    let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let cuda_include_path = PathBuf::from(&cuda_home).join("include");
    let cuda_lib_path = PathBuf::from(&cuda_home).join("lib64"); // Or lib, depending on system

    println!("cargo:rustc-link-search=native={}", cuda_lib_path.display());
    println!("cargo:rustc-link-lib=dylib=cudart"); // Link against CUDA runtime
    // Add other CUDA libraries if needed (e.g., cublas, cufft)

    // Compile the .cu FFI wrapper files
    // Note: The `cc` crate might need specific setup to use `nvcc` for `.cu` files.
    // This setup can be complex. A common approach is to compile .cu to .o with nvcc separately
    // and then link the .o files, or compile to a static library.

    // Let's try a simplified approach using cc::Build, assuming it can delegate to g++/clang++
    // which in turn might be configured to use nvcc for .cu files, or we compile .cu to .o first.
    // A more direct way is to use `Command` to run `nvcc`.

    // Example of compiling .cu files into a static library `libgpukernels.a`
    // This requires nvcc to be in PATH.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let lib_name = "nockchain_gpu_kernels";

    let files_to_compile = vec![
        cuda_kernels_path.join("poly_mul_ffi.cu"),
        cuda_kernels_path.join("ntt_fft_ffi.cu"),
        cuda_kernels_path.join("field_ops_ffi.cu"),
        cuda_kernels_path.join("hash_ops_ffi.cu"),
    ];

    // Compile .cu files using cc-rs which can invoke nvcc if configured or if the C++ compiler is nvcc-aware.
    // For a more explicit nvcc build, you might use a custom build script or a crate like `build-deps`.
    cc::Build::new()
        .cuda(true) // Enable CUDA support in cc-rs (requires compatible toolchain)
        .cpp(true)  // Treat as C++
        .include(&cuda_kernels_path) // For includes like "poly_mul.cu"
        .include(&cuda_include_path)
        .files(files_to_compile.iter())
        // Add necessary NVCC flags if cc-rs doesn't set them appropriately
        // e.g., .flag("-arch=sm_75") for a specific GPU architecture
        .compiler("nvcc") // Explicitly try to use nvcc, though cc-rs might manage this
        .cargo_metadata(true)
        .static_flag(true)
        .compile(lib_name);

    // Tell cargo to link the compiled static library.
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static={}", lib_name);

    // If not using cc::Build's CUDA capabilities directly, you might do:
    /*
    use std::process::Command;
    for cu_file_path in files_to_compile {
        let stem = cu_file_path.file_stem().unwrap().to_str().unwrap();
        let obj_file = out_dir.join(format!("{}.o", stem));
        let status = Command::new("nvcc")
            .arg("-c")
            .arg("-o").arg(&obj_file)
            .arg("-I").arg(&cuda_include_path)
            .arg("-I").arg(&cuda_kernels_path) // For includes within cuda_kernels dir
            // Add architecture flags like -gencode arch=compute_75,code=sm_75
            .arg(cu_file_path)
            .status()
            .expect("Failed to execute nvcc");
        if !status.success() {
            panic!("nvcc compilation failed for {}", cu_file_path.display());
        }
        Command::new("ar")
            .arg("rcs")
            .arg(out_dir.join(format!("lib{}.a", lib_name)))
            .arg(obj_file)
            .status()
            .expect("Failed to execute ar");
    }
    */

    println!("cargo:rerun-if-env-changed=CUDA_HOME");
}


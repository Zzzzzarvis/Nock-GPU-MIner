# Nockchain GPU Mining Version: User Guide & Deployment Manual

## 1. Overview

This document provides instructions for setting up, compiling, running, and troubleshooting the GPU-accelerated version of the Nockchain miner. This version offloads computationally intensive ZK-STARK proof generation tasks to NVIDIA GPUs using CUDA, offering significant performance improvements over the standard CPU-only miner.

Refer to the `benchmark_report.md` for detailed performance comparisons.

## 2. System Requirements

### 2.1. Hardware

*   **NVIDIA GPU**: A CUDA-enabled NVIDIA GPU is required. Performance will vary based on the GPU model (e.g., RTX 20xx series or newer, Tesla series recommended for optimal performance).
*   **CPU**: A modern multi-core CPU (e.g., Intel Core i5/i7/i9, AMD Ryzen 5/7/9 or equivalent).
*   **RAM**: Minimum 16GB RAM, 32GB or more recommended for complex proofs.
*   **Storage**: Sufficient disk space for the Nockchain source code, build artifacts, and any blockchain data (approx. 10GB for source and build, blockchain data varies).

### 2.2. Software

*   **Operating System**: Linux (Ubuntu 20.04 LTS or newer recommended). The build process is primarily tested on Linux.
*   **NVIDIA CUDA Toolkit**: Version 11.x (e.g., 11.4 or newer). Ensure the toolkit is compatible with your GPU and driver.
*   **NVIDIA GPU Driver**: A recent NVIDIA driver compatible with your CUDA Toolkit version.
*   **Rust Language**: Latest stable version (e.g., 1.70.0 or newer). Install via `rustup`.
*   **C++ Compiler**: A C++ compiler supporting C++17 (e.g., GCC 9.x or newer, Clang 10.x or newer).
*   **CMake**: Version 3.18 or newer (for building CUDA C++ FFI library).
*   **Git**: For cloning the Nockchain repository.
*   **Standard Build Tools**: `build-essential` on Debian/Ubuntu, or equivalent for other distributions (provides `make`, etc.).

## 3. Setup and Installation

### 3.1. Install Dependencies

#### 3.1.1. NVIDIA CUDA Toolkit and Driver

Follow the official NVIDIA documentation to install the CUDA Toolkit and the appropriate GPU driver for your system:
*   CUDA Toolkit Archive: [https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)
*   NVIDIA Drivers: [https://www.nvidia.com/Download/index.aspx](https://www.nvidia.com/Download/index.aspx)

Verify installation:
```bash
nvcc --version
nvidia-smi
```

#### 3.1.2. Rust

If you don't have Rust, install it via rustup:
```bash
curl --proto 	=https --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

#### 3.1.3. Build Tools

On Ubuntu/Debian:
```bash
sudo apt update
sudo apt install build-essential cmake git
```

### 3.2. Clone the Nockchain GPU Repository

(Assuming the modified code is in a specific branch or fork. For this example, we use the main Nockchain repository path and assume modifications are present.)

```bash
git clone https://github.com/zorp-corp/nockchain.git nockchain_gpu_miner
cd nockchain_gpu_miner
# If there's a specific branch for GPU mining, checkout to it:
# git checkout gpu-mining-branch
```

Our GPU modifications are primarily within the `nockchain_gpu_ws` directory (created during development) and integrated into the `crates/nockvm_crypto` part of the main Nockchain structure.

## 4. Compilation

The GPU-accelerated version requires compiling both the Rust Nockchain code and the C++/CUDA FFI library.

### 4.1. Compiling the CUDA FFI Library

The `build.rs` script within the `crates/nockvm_crypto` (or a similar relevant crate where FFI is integrated) is configured to compile the CUDA kernels and the C++ FFI wrapper code.

Ensure your `CUDA_HOME` environment variable is set if CUDA is not installed in a default location. For example:
```bash
export CUDA_HOME=/usr/local/cuda-11.4
```

### 4.2. Compiling the Nockchain Miner

Navigate to the root of the cloned `nockchain_gpu_miner` repository.

The Nockchain project uses Cargo for building. The `build.rs` script in `crates/nockvm_crypto` handles the CUDA compilation and linking automatically when you build the Rust project.

To build the Nockchain miner (assuming the main executable is `nock` or a specific miner binary):

```bash
cargo build --release
```

This command will:
1.  Compile the CUDA kernels (`.cu` files in `nockchain_gpu_ws/cuda_kernels/`) into an object file or static library (`libnockchain_gpu_kernels.a`).
2.  Compile the Rust crates, linking the `nockvm_crypto` crate (and others) against the CUDA FFI library.
3.  Produce the final executable(s) in `target/release/`.

If specific features or binaries need to be built, refer to Nockchain's original build instructions, ensuring the `nockvm_crypto` crate with GPU support is included.

## 5. Running the GPU Miner

After successful compilation, the miner executable will be in the `target/release/` directory.

### 5.1. Configuration

The GPU miner might have a command-line flag or a configuration file option to enable GPU acceleration. We implemented an optional flag.

Example (hypothetical command, actual command depends on Nockchain's CLI):

To run with GPU acceleration:
```bash
./target/release/nock_miner --gpu # or similar flag
# (Other necessary Nockchain arguments like --config path/to/config.toml, --mine, etc.)
```

To run with CPU-only (for comparison or if GPU is unavailable):
```bash
./target/release/nock_miner --cpu # or omit --gpu if CPU is default
# (Other necessary Nockchain arguments)
```

Refer to the specific Nockchain CLI documentation for all available arguments for mining, connecting to a network, etc. The GPU/CPU selection is an added feature.

### 5.2. Monitoring

While the miner is running, you can monitor GPU utilization using `nvidia-smi`:
```bash
watch -n 1 nvidia-smi
```
This will show GPU temperature, power usage, memory usage, and GPU utilization percentage.

## 6. Troubleshooting

### 6.1. Compilation Errors

*   **CUDA Not Found / `nvcc` errors**: 
    *   Ensure CUDA Toolkit is installed correctly and `nvcc` is in your `PATH`.
    *   Verify `CUDA_HOME` environment variable is set if CUDA is in a non-standard location.
    *   Check `build.rs` in `crates/nockvm_crypto` for correct CUDA paths and compilation flags.
*   **C++ Compiler Errors**: Ensure a compatible C++ compiler (GCC/Clang) is installed and meets version requirements.
*   **Linking Errors (undefined reference to CUDA functions)**:
    *   Verify that the CUDA FFI static library (`libnockchain_gpu_kernels.a`) was built correctly.
    *   Check the `#[link(name = 

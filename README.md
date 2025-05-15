# Nockchain GPU Miner

Nockchain GPU miner with one-click deployment script for Vast.ai.

This repository contains the source code, CUDA kernels, and build scripts for a GPU-accelerated version of the Nockchain miner. It also includes a user guide, benchmark report, and a one-click deployment script designed for use on Vast.ai instances.

## Features

*   **GPU Acceleration**: Utilizes CUDA to significantly speed up Nockchain mining operations, particularly NTT/FFT, polynomial multiplication, and hash calculations.
*   **Rust & C++/CUDA Integration**: Combines the safety and concurrency of Rust with the performance of C++/CUDA for optimal mining efficiency.
*   **FFI Interface**: A well-defined Foreign Function Interface (FFI) allows seamless communication between the Rust core and the C++/CUDA GPU modules.
*   **One-Click Deployment (Vast.ai)**: Includes a shell script to automate the setup and launch of the GPU miner on Vast.ai cloud GPU instances.
*   **Detailed Documentation**: Comes with a comprehensive `USER_GUIDE.md` for setup, compilation, and operation, as well as a `benchmark_report.md` detailing performance gains over CPU mining.

## Getting Started

### Prerequisites

Before you begin, ensure you have a compatible NVIDIA GPU and the necessary drivers installed. For compiling the miner, you will need:

*   CUDA Toolkit (version compatible with the kernels, e.g., 11.x or 12.x)
*   Rust programming language and Cargo
*   A C++ compiler (e.g., g++)
*   CMake

Detailed system requirements and setup instructions can be found in the [USER_GUIDE.md](USER_GUIDE.md).

### Installation and Usage

1.  **Clone the repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL of this repository
    cd nock-gpu-miner
    ```

2.  **Compile the Miner:**
    Follow the compilation steps outlined in the [USER_GUIDE.md](USER_GUIDE.md). This typically involves building the CUDA kernels and then the Rust project.

3.  **Run the Miner:**
    Once compiled, you can run the miner using the command-line interface. Refer to the [USER_GUIDE.md](USER_GUIDE.md) for detailed usage instructions and available options.

### One-Click Deployment on Vast.ai

For users of Vast.ai, a convenience script `deploy_vastai.sh` (to be created in a later step) will be provided to automate the setup and launch process. Instructions for using this script will be included here and in the `USER_GUIDE.md` once the script is finalized.

**Example (conceptual, actual script will be provided later):**
```bash
# Download the script
wget <script_url>/deploy_vastai.sh
chmod +x deploy_vastai.sh

# Run the script to setup and start mining
./deploy_vastai.sh --wallet-address YOUR_WALLET_ADDRESS
```

## Documentation

*   **[USER_GUIDE.md](USER_GUIDE.md)**: Comprehensive guide covering system requirements, environment setup, compilation, execution, configuration, monitoring, and troubleshooting.
*   **[benchmark_report.md](benchmark_report.md)**: Detailed performance comparison between the GPU and CPU versions of the miner.
*   **FFI Design, Implementation Plan, Feasibility & Risk Assessment**: These documents were part of the initial planning and are available at [https://ppifistk.manus.space](https://ppifistk.manus.space)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## Disclaimer

This software is provided for educational and experimental purposes. Use at your own risk. The developers are not responsible for any financial loss or other damages incurred from using this software.

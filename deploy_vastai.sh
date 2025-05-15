#!/bin/bash

# Nockchain GPU Miner One-Click Deployment Script for Vast.ai

# Exit immediately if a command exits with a non-zero status.
set -e

# Define variables (user might need to adjust these or they can be passed as arguments)
GITHUB_REPO_URL="https://github.com/YOUR_USERNAME/nock-gpu-miner.git" # Placeholder, user needs to change this
PROJECT_DIR="nock-gpu-miner"
WALLET_ADDRESS="YOUR_WALLET_ADDRESS" # Placeholder, user needs to provide this

# Function to print messages
log() {
    echo "[INFO] $(date +'%Y-%m-%d %H:%M:%S') - $1"
}

# --- Dependency Installation ---
log "Starting dependency installation..."

# Update package lists
sudo apt-get update -y

# Install essential build tools
log "Installing build-essential, cmake, pkg-config, libssl-dev..."
sudo apt-get install -y build-essential cmake pkg-config libssl-dev curl git

# Install Rust and Cargo
log "Installing Rust and Cargo..."
if ! command -v cargo &> /dev/null
then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
else
    log "Rust is already installed."
fi
rustc --version
cargo --version

# Install CUDA Toolkit (Vast.ai instances usually have drivers, but toolkit might be needed for compilation)
# This is a placeholder. The exact CUDA installation method can be complex and depends on the Vast.ai image.
# For many Vast.ai PyTorch/TensorFlow images, nvcc might already be available.
# If not, the user might need to select an image with CUDA pre-installed or install it manually.
log "Checking for CUDA (nvcc)..."
if ! command -v nvcc &> /dev/null
then
    log "CUDA (nvcc) not found. Attempting to install a common version."
    log "This step might take a while and might require a specific version based on your GPU and driver."
    # Example: Install CUDA Toolkit 11.8 (adjust version as needed)
    # wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    # sudo dpkg -i cuda-keyring_1.1-1_all.deb
    # sudo apt-get update
    # sudo apt-get -y install cuda-toolkit-11-8
    # export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
    # export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    log "CUDA installation is complex. Please ensure your Vast.ai image has CUDA installed or install it manually if this script fails at compilation."
    log "For now, proceeding with the assumption that CUDA is or will be available for compilation."
else
    log "CUDA (nvcc) found: $(nvcc --version)"
fi

log "Dependency installation phase completed."

# --- Clone Repository ---
log "Cloning the Nockchain GPU miner repository..."
if [ -d "$PROJECT_DIR" ]; then
    log "Directory $PROJECT_DIR already exists. Pulling latest changes."
    cd $PROJECT_DIR
    git pull
    cd ..
else
    git clone $GITHUB_REPO_URL $PROJECT_DIR
fi

cd $PROJECT_DIR

# --- Compile the Miner ---
log "Starting compilation of the Nockchain GPU miner..."
log "This may take a significant amount of time."

# The USER_GUIDE.md should contain the exact compilation steps.
# Assuming a Cargo build process for the Rust part and a separate build for CUDA kernels if needed.
# For example, if there's a build script in nockchain_gpu_ws:
if [ -d "nockchain_gpu_ws" ]; then
    cd nockchain_gpu_ws
    # Placeholder for actual build commands from USER_GUIDE.md
    # e.g., make cuda_kernels
    # cargo build --release
    log "Attempting to build using 'cargo build --release' in nockchain_gpu_ws/nockchain/crates/nockvm_crypto (adjust if needed based on USER_GUIDE.md)"
    # This path is based on previous file structure, it might need adjustment
    if [ -d "nockchain/crates/nockvm_crypto" ]; then
        cd nockchain/crates/nockvm_crypto
        # Ensure build.rs can find CUDA if it links CUDA libraries
        # Potentially set CUDA_HOME or other env vars if needed by build.rs
        cargo build --release
        log "Compilation finished. The executable should be in target/release/"
        cd ../../../.. # Back to nockchain_gpu_ws
    else
        log "Could not find nockchain/crates/nockvm_crypto. Please check USER_GUIDE.md for compilation instructions."
        exit 1
    fi
    cd .. # Back to PROJECT_DIR
else
    log "Directory nockchain_gpu_ws not found. Please check USER_GUIDE.md for compilation instructions."
    exit 1
fi

log "Compilation phase completed."

# --- Run the Miner ---
log "To run the miner:"
log "1. Navigate to the project directory: cd $HOME/$PROJECT_DIR"
log "2. Find the compiled executable (e.g., in nockchain_gpu_ws/nockchain/target/release/ or similar based on USER_GUIDE.md)"
log "3. Run the miner with your wallet address and other parameters as specified in USER_GUIDE.md."
log "Example (replace with actual executable path and parameters):"
log "   ./nockchain_gpu_ws/nockchain/target/release/nockvm_miner --wallet YOUR_WALLET_ADDRESS --gpu-enabled"

log "Script finished. Please check the USER_GUIDE.md for detailed run instructions."

# Example of how to make the script accept parameters like wallet address and repo URL:
# #!/bin/bash
# set -e
# GITHUB_REPO_URL="${1:-https://github.com/YOUR_USERNAME/nock-gpu-miner.git}"
# WALLET_ADDRESS="${2:-YOUR_WALLET_ADDRESS}"
# log "Using GitHub Repo: $GITHUB_REPO_URL"
# log "Using Wallet Address: $WALLET_ADDRESS"
# # ... rest of the script
# # Then run as: ./deploy_vastai.sh <your_repo_url> <your_wallet_address>

exit 0


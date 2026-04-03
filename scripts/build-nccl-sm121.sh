#!/usr/bin/env bash
# =============================================================================
# Build NCCL from Source for Blackwell sm_121 (GB10)
#
# Pre-built NCCL packages lack GPU kernels for sm_121. This script builds
# NCCL v2.28.9 from source with the correct gencode flag.
#
# This must be run on EVERY node in a multi-node cluster.
#
# Output: ~/nccl/build/lib/libnccl.so.2.28.9
# =============================================================================
set -euo pipefail

NCCL_DIR="${NCCL_DIR:-$HOME/nccl}"
NCCL_VERSION="${NCCL_VERSION:-v2.28.9-1}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NPROC="${NPROC:-$(nproc)}"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
die()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# Check prerequisites
[[ -d "$CUDA_HOME" ]] || die "CUDA_HOME not found at $CUDA_HOME. Install CUDA 13.0 first."
command -v nvcc &>/dev/null || die "nvcc not found. Ensure CUDA is in PATH."
command -v make &>/dev/null || die "make not found. Install build-essential."
command -v git &>/dev/null  || die "git not found."

info "Building NCCL $NCCL_VERSION for sm_121"
info "CUDA_HOME: $CUDA_HOME"
info "Build threads: $NPROC"
info "Output dir: $NCCL_DIR/build/lib/"

# Clone or update
if [[ -d "$NCCL_DIR/.git" ]]; then
    info "NCCL repo already exists at $NCCL_DIR"
    cd "$NCCL_DIR"
    git fetch --tags
else
    info "Cloning NCCL repository..."
    git clone https://github.com/NVIDIA/nccl.git "$NCCL_DIR"
    cd "$NCCL_DIR"
fi

info "Checking out $NCCL_VERSION..."
git checkout "$NCCL_VERSION"

# Clean previous build
if [[ -d build ]]; then
    info "Cleaning previous build..."
    make clean 2>/dev/null || true
fi

# Build with sm_121 gencode
info "Compiling NCCL (this takes 5-10 minutes)..."
make -j"$NPROC" src.build \
    NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121" \
    CUDA_HOME="$CUDA_HOME"

# Verify build
if [[ -f "$NCCL_DIR/build/lib/libnccl.so" ]]; then
    BUILT_VERSION="$(ls "$NCCL_DIR/build/lib/libnccl.so."* 2>/dev/null | head -1 | sed 's/.*libnccl.so.//')"
    ok "NCCL built successfully: libnccl.so.${BUILT_VERSION}"
    ok "Library path: $NCCL_DIR/build/lib/"
    echo ""
    echo "To use this NCCL build, add to your LD_LIBRARY_PATH:"
    echo "  export LD_LIBRARY_PATH=$NCCL_DIR/build/lib:\$LD_LIBRARY_PATH"
else
    die "Build completed but libnccl.so not found. Check build output above."
fi

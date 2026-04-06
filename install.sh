#!/usr/bin/env bash
# =============================================================================
# vLLM GB10 Installer
# Installs and patches vLLM for NVIDIA DGX Spark GB10 (ARM aarch64, sm_121)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"
VENV_DIR="$HOME/vllm-env"
NCCL_DIR="$HOME/nccl"
NCCL_VERSION="v2.28.9-1"
VLLM_VERSION="0.18.1"
PYTHON="python3.12"

# -- Pre-flight checks -------------------------------------------------------

info "Checking platform..."

ARCH="$(uname -m)"
if [[ "$ARCH" != "aarch64" ]]; then
    warn "Architecture is $ARCH, not aarch64. This installer is designed for DGX Spark GB10."
    read -rp "Continue anyway? [y/N] " ans
    [[ "$ans" =~ ^[Yy] ]] || exit 0
fi

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")"
    info "Detected GPU: $GPU_NAME"
else
    warn "nvidia-smi not found. Ensure CUDA 13.0 is installed."
fi

if ! command -v "$PYTHON" &>/dev/null; then
    die "Python 3.12 not found. Install with: sudo apt install python3.12 python3.12-venv python3.12-dev"
fi

if ! dpkg -s python3.12-dev &>/dev/null 2>&1; then
    warn "python3.12-dev not found. Installing (required for Triton compilation)..."
    sudo apt update && sudo apt install -y python3.12-dev
fi

# -- System dependencies -----------------------------------------------------

info "Installing system dependencies..."
sudo apt update -qq
sudo apt install -y -qq build-essential git python3.12-venv python3.12-dev

# -- Virtual environment ------------------------------------------------------

if [[ -d "$VENV_DIR" ]]; then
    info "Virtual environment already exists at $VENV_DIR"
    read -rp "Recreate it? [y/N] " ans
    if [[ "$ans" =~ ^[Yy] ]]; then
        rm -rf "$VENV_DIR"
        $PYTHON -m venv "$VENV_DIR"
        ok "Created fresh virtualenv at $VENV_DIR"
    fi
else
    info "Creating virtual environment at $VENV_DIR..."
    $PYTHON -m venv "$VENV_DIR"
    ok "Created virtualenv at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# -- Install vLLM + dependencies ---------------------------------------------

info "Upgrading pip..."
pip install --upgrade pip setuptools wheel -q

info "Installing vLLM $VLLM_VERSION..."
pip install "vllm==$VLLM_VERSION" -q

info "Installing aligned package versions..."
pip install "triton==3.6.0" "transformers==4.57.6" -q

ok "vLLM $VLLM_VERSION installed"

# -- Apply CUTLASS FP8 patch (Fix 5) -----------------------------------------

info "Applying CUTLASS FP8 patch for sm_121..."
if ! bash "$SCRIPT_DIR/patches/cutlass-fp8-sm121.patch"; then
    die "CUTLASS FP8 patch failed. Check output above."
fi
ok "CUTLASS FP8 patch applied"

# -- Build NCCL from source (Fix 1) ------------------------------------------

if [[ -f "$NCCL_DIR/build/lib/libnccl.so" ]]; then
    info "NCCL custom build already exists at $NCCL_DIR/build/lib/"
    read -rp "Rebuild NCCL? [y/N] " ans
    if [[ "$ans" =~ ^[Yy] ]]; then
        bash "$SCRIPT_DIR/scripts/build-nccl-sm121.sh"
    else
        info "Skipping NCCL build"
    fi
else
    info "Building NCCL from source for sm_121..."
    bash "$SCRIPT_DIR/scripts/build-nccl-sm121.sh"
fi

# -- Configure Ray memory threshold (Fix 7) ----------------------------------

info "Configuring Ray for unified memory..."
bash "$SCRIPT_DIR/patches/ray-memory-threshold.sh"
ok "Ray memory threshold configured"

# -- Verify installation ------------------------------------------------------

info "Running verification..."
bash "$SCRIPT_DIR/scripts/verify-install.sh"

# -- Done ---------------------------------------------------------------------

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}  vLLM GB10 installation complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "To use vLLM, source the appropriate environment file:"
echo ""
echo "  Single node:     source configs/single-node.env"
echo "  Multi-node head: source configs/multi-node-head.env"
echo "  Multi-node worker: source configs/multi-node-worker.env"
echo ""
echo "Then launch a model:"
echo ""
echo "  vllm serve /path/to/model --trust-remote-code --enforce-eager"
echo ""
echo "See configs/example-launch.sh for tested model configurations."
echo ""

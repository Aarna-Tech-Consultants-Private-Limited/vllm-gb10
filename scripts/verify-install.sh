#!/usr/bin/env bash
# =============================================================================
# Verify vLLM GB10 Installation
# Checks that all components are correctly installed and patched.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

PASS=0
FAIL=0
WARN_COUNT=0

pass()      { echo -e "  ${GREEN}PASS${NC}  $*"; ((PASS++)); }
fail()      { echo -e "  ${RED}FAIL${NC}  $*"; ((FAIL++)); }
warn()      { echo -e "  ${YELLOW}WARN${NC}  $*"; ((WARN_COUNT++)); }
info()      { echo -e "${CYAN}[CHECK]${NC} $*"; }

VENV_DIR="${VENV_DIR:-$HOME/vllm-env}"
NCCL_DIR="${NCCL_DIR:-$HOME/nccl}"

echo ""
echo "============================================"
echo "  vLLM GB10 Installation Verification"
echo "============================================"
echo ""

# -- 1. Platform --------------------------------------------------------------
info "Platform"

ARCH="$(uname -m)"
if [[ "$ARCH" == "aarch64" ]]; then
    pass "Architecture: aarch64"
else
    warn "Architecture: $ARCH (expected aarch64 for GB10)"
fi

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")"
    if [[ -n "$GPU_NAME" ]]; then
        pass "GPU detected: $GPU_NAME"
    else
        warn "nvidia-smi found but could not query GPU name"
    fi
else
    fail "nvidia-smi not found"
fi

# -- 2. Python + Virtualenv ---------------------------------------------------
info "Python environment"

if [[ -d "$VENV_DIR" ]]; then
    pass "Virtualenv exists: $VENV_DIR"
else
    fail "Virtualenv not found: $VENV_DIR"
fi

if [[ -f "$VENV_DIR/bin/python" ]]; then
    PY_VERSION="$("$VENV_DIR/bin/python" --version 2>&1)"
    if [[ "$PY_VERSION" == *"3.12"* ]]; then
        pass "Python version: $PY_VERSION"
    else
        fail "Python version: $PY_VERSION (expected 3.12)"
    fi
fi

if dpkg -s python3.12-dev &>/dev/null 2>&1; then
    pass "python3.12-dev installed"
else
    fail "python3.12-dev not installed (needed for Triton compilation)"
fi

# -- 3. vLLM -------------------------------------------------------------------
info "vLLM installation"

if [[ -f "$VENV_DIR/bin/python" ]]; then
    VLLM_VERSION="$("$VENV_DIR/bin/python" -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo "NOT FOUND")"
    if [[ "$VLLM_VERSION" != "NOT FOUND" ]]; then
        pass "vLLM version: $VLLM_VERSION"
    else
        fail "vLLM not importable"
    fi

    # Check triton
    TRITON_VERSION="$("$VENV_DIR/bin/python" -c 'import triton; print(triton.__version__)' 2>/dev/null || echo "NOT FOUND")"
    if [[ "$TRITON_VERSION" != "NOT FOUND" ]]; then
        pass "Triton version: $TRITON_VERSION"
    else
        fail "Triton not importable"
    fi

    # Check transformers
    TF_VERSION="$("$VENV_DIR/bin/python" -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo "NOT FOUND")"
    if [[ "$TF_VERSION" != "NOT FOUND" ]]; then
        pass "Transformers version: $TF_VERSION"
    else
        fail "Transformers not importable"
    fi

    # Check torch
    TORCH_VERSION="$("$VENV_DIR/bin/python" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "NOT FOUND")"
    if [[ "$TORCH_VERSION" != "NOT FOUND" ]]; then
        pass "PyTorch version: $TORCH_VERSION"
        CUDA_AVAIL="$("$VENV_DIR/bin/python" -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo "False")"
        if [[ "$CUDA_AVAIL" == "True" ]]; then
            pass "PyTorch CUDA available"
        else
            fail "PyTorch CUDA not available"
        fi
    else
        fail "PyTorch not importable"
    fi
fi

# -- 4. CUTLASS FP8 Patch -----------------------------------------------------
info "CUTLASS FP8 patch (Fix 5)"

W8A8_FILE="$VENV_DIR/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py"
if [[ -f "$W8A8_FILE" ]]; then
    # Functional check: actually import and call the patched functions
    FP8_RESULT="$("$VENV_DIR/bin/python" -c 'from vllm.model_executor.layers.quantization.utils.w8a8_utils import cutlass_fp8_supported; print(cutlass_fp8_supported())' 2>/dev/null || echo "IMPORT_ERROR")"
    if [[ "$FP8_RESULT" == "False" ]]; then
        pass "cutlass_fp8_supported() returns False"
    elif [[ "$FP8_RESULT" == "IMPORT_ERROR" ]]; then
        fail "Could not import cutlass_fp8_supported() (vLLM may not be installed correctly)"
    else
        fail "cutlass_fp8_supported() returns $FP8_RESULT (expected False)"
    fi

    BLOCK_FP8_RESULT="$("$VENV_DIR/bin/python" -c 'from vllm.model_executor.layers.quantization.utils.w8a8_utils import cutlass_block_fp8_supported; print(cutlass_block_fp8_supported())' 2>/dev/null || echo "IMPORT_ERROR")"
    if [[ "$BLOCK_FP8_RESULT" == "False" ]]; then
        pass "cutlass_block_fp8_supported() returns False"
    elif [[ "$BLOCK_FP8_RESULT" == "IMPORT_ERROR" ]]; then
        fail "Could not import cutlass_block_fp8_supported()"
    else
        fail "cutlass_block_fp8_supported() returns $BLOCK_FP8_RESULT (expected False)"
    fi

    # Check module-level constants
    CONST_RESULT="$("$VENV_DIR/bin/python" -c 'from vllm.model_executor.layers.quantization.utils.w8a8_utils import CUTLASS_FP8_SUPPORTED, CUTLASS_BLOCK_FP8_SUPPORTED; print(CUTLASS_FP8_SUPPORTED, CUTLASS_BLOCK_FP8_SUPPORTED)' 2>/dev/null || echo "IMPORT_ERROR")"
    if [[ "$CONST_RESULT" == "False False" ]]; then
        pass "CUTLASS_FP8_SUPPORTED and CUTLASS_BLOCK_FP8_SUPPORTED constants are False"
    elif [[ "$CONST_RESULT" == "IMPORT_ERROR" ]]; then
        fail "Could not import CUTLASS FP8 constants"
    else
        fail "CUTLASS FP8 constants are '$CONST_RESULT' (expected 'False False')"
    fi
else
    fail "w8a8_utils.py not found (is vLLM installed?)"
fi

# -- 5. NCCL Custom Build -----------------------------------------------------
info "NCCL custom build (Fix 1)"

if [[ -f "$NCCL_DIR/build/lib/libnccl.so" ]]; then
    NCCL_SO="$(ls "$NCCL_DIR/build/lib/libnccl.so."* 2>/dev/null | head -1)"
    if [[ -n "$NCCL_SO" ]]; then
        pass "Custom NCCL built: $(basename "$NCCL_SO")"
    else
        pass "Custom NCCL built (version unknown)"
    fi
else
    warn "Custom NCCL not found at $NCCL_DIR/build/lib/ (needed for multi-node only)"
fi

# -- 6. Ray Memory Threshold --------------------------------------------------
info "Ray memory threshold (Fix 7)"

if [[ "${RAY_memory_usage_threshold:-}" == "1.0" ]]; then
    pass "RAY_memory_usage_threshold=1.0 (set in environment)"
elif grep -q "RAY_memory_usage_threshold" "$HOME/.bashrc" 2>/dev/null; then
    pass "RAY_memory_usage_threshold configured in .bashrc"
else
    warn "RAY_memory_usage_threshold not detected (set to 1.0 before running vLLM)"
fi

# -- 7. NCCL Environment Variables --------------------------------------------
info "NCCL environment variables (Fix 4)"

for var in NCCL_SOCKET_IFNAME NCCL_P2P_DISABLE NCCL_IB_DISABLE; do
    if [[ -n "${!var:-}" ]]; then
        pass "$var=${!var}"
    else
        warn "$var not set (needed for multi-node)"
    fi
done

# -- Summary -------------------------------------------------------------------
echo ""
echo "============================================"
echo "  Results: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}, ${YELLOW}$WARN_COUNT warnings${NC}"
echo "============================================"
echo ""

if [[ $FAIL -gt 0 ]]; then
    echo "Some checks failed. Review the output above and re-run the installer."
    exit 1
elif [[ $WARN_COUNT -gt 0 ]]; then
    echo "All critical checks passed. Warnings are informational (some apply only to multi-node setups)."
    exit 0
else
    echo "All checks passed. Installation is ready."
    exit 0
fi

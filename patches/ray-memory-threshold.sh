#!/usr/bin/env bash
# =============================================================================
# Ray Memory Threshold Configuration for DGX Spark GB10
#
# Problem: DGX Spark has unified CPU/GPU memory (128GB shared). Ray's default
# memory_usage_threshold of 0.95 triggers OOM kills after model weights are
# loaded into GPU memory, because Ray counts GPU-resident weights as system
# memory usage.
#
# Error without fix:
#   ray.exceptions.OutOfMemoryError: Memory on the node was 115.65GB / 121.69GB
#   (0.950369), which exceeds the memory usage threshold of 0.95.
#
# Fix: Set RAY_memory_usage_threshold=1.0 to disable Ray's OOM killer.
# =============================================================================
set -euo pipefail

echo "[INFO] Configuring Ray for GB10 unified memory..."

# Write to shell profile if not already present
PROFILE="$HOME/.bashrc"
MARKER="# vllm-gb10: Ray unified memory fix"

if grep -q "$MARKER" "$PROFILE" 2>/dev/null; then
    echo "[INFO] Ray memory threshold already configured in $PROFILE"
else
    cat >> "$PROFILE" << EOF

$MARKER
export RAY_memory_usage_threshold=1.0
EOF
    echo "[OK] Added RAY_memory_usage_threshold=1.0 to $PROFILE"
fi

# Also export for current session
export RAY_memory_usage_threshold=1.0

echo "[OK] Ray memory threshold set to 1.0 (OOM killer disabled)"
echo ""
echo "NOTE: For production, also tune these vLLM settings:"
echo "  --gpu-memory-utilization 0.80   (default 0.90 is too aggressive)"
echo "  --max-model-len 16384           (reduce if needed for large models)"

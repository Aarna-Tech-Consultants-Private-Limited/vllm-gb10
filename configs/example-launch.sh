#!/usr/bin/env bash
# =============================================================================
# Example vLLM Launch Commands for DGX Spark GB10
#
# These are reference commands for tested model configurations.
# Adjust paths, IPs, and parameters for your setup.
# =============================================================================

# =============================================================================
# MULTI-NODE: Qwen3.5-122B-A10B-FP8 (TP=2, 2 nodes)
# Production configuration for 122B parameter MoE model.
# Requires: 2 DGX Spark GB10 nodes, QSFP connected, Ray cluster running.
# =============================================================================

launch_qwen35_122b_multinode() {
    # Prerequisites:
    #   Head node:   ray start --head --port=6379 --num-gpus=1
    #   Worker node: ray start --address=<HEAD_QSFP_IP>:6379 --num-gpus=1

    source ~/vllm-env/bin/activate
    source configs/multi-node-head.env
    export VLLM_HOST_IP=192.168.100.10  # Set to YOUR head node QSFP IP

    vllm serve /home/atc/hf_models/Qwen3.5-122B-A10B-FP8 \
        --served-model-name manthan-general \
        --port 8000 \
        --tensor-parallel-size 2 \
        --distributed-executor-backend ray \
        --trust-remote-code \
        --gpu-memory-utilization 0.80 \
        --max-model-len 16384 \
        --enforce-eager \
        --enable-prefix-caching \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        --max-num-batched-tokens 4096
}

# =============================================================================
# SINGLE-NODE: Qwen3-Omni-30B
# Omni-modal model (text + audio + vision). Use --omni flag.
# =============================================================================

launch_qwen3_omni_30b() {
    source ~/vllm-env/bin/activate
    source configs/single-node.env

    vllm serve /home/atc/hf_models/Qwen3-Omni-30B --omni \
        --served-model-name manthan-omni \
        --port 8000 \
        --trust-remote-code \
        --gpu-memory-utilization 0.85 \
        --max-model-len 32768 \
        --enforce-eager \
        --enable-prefix-caching
}

# =============================================================================
# SINGLE-NODE: Qwen3.5-35B-A3B (Lightweight MoE)
# Small, fast MoE model. Only 3B active parameters.
# =============================================================================

launch_qwen35_35b_a3b() {
    source ~/vllm-env/bin/activate
    source configs/single-node.env

    vllm serve /home/atc/hf_models/Qwen3.5-35B-A3B \
        --served-model-name manthan-fast \
        --port 8000 \
        --trust-remote-code \
        --gpu-memory-utilization 0.85 \
        --max-model-len 32768 \
        --enforce-eager \
        --enable-prefix-caching \
        --reasoning-parser qwen3
}

# =============================================================================
# SINGLE-NODE: Generic model launch template
# =============================================================================

launch_generic() {
    local MODEL_PATH="${1:?Usage: launch_generic /path/to/model}"
    local MODEL_NAME="${2:-my-model}"
    local PORT="${3:-8000}"

    source ~/vllm-env/bin/activate
    source configs/single-node.env

    vllm serve "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --port "$PORT" \
        --trust-remote-code \
        --gpu-memory-utilization 0.85 \
        --max-model-len 32768 \
        --enforce-eager \
        --enable-prefix-caching
}

# =============================================================================
# Usage: source this file and call a function, or copy-paste commands.
#
#   source configs/example-launch.sh
#   launch_qwen35_122b_multinode
#   launch_qwen3_omni_30b
#   launch_qwen35_35b_a3b
#   launch_generic /path/to/model my-model-name 8000
# =============================================================================

echo "Example launch functions loaded. Available:"
echo "  launch_qwen35_122b_multinode  - Qwen3.5-122B-A10B-FP8 (2 nodes, TP=2)"
echo "  launch_qwen3_omni_30b         - Qwen3-Omni-30B (single node, --omni)"
echo "  launch_qwen35_35b_a3b         - Qwen3.5-35B-A3B (single node, lightweight)"
echo "  launch_generic <path> [name] [port]  - Generic single-node launch"

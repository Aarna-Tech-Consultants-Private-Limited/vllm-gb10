# vLLM for NVIDIA DGX Spark GB10

**Custom native vLLM for NVIDIA DGX Spark GB10 (ARM aarch64, Blackwell sm_121)**

Stock vLLM does not work on the DGX Spark GB10 out of the box. The GB10 is NVIDIA's first ARM-based desktop AI platform with a Blackwell-architecture GPU (compute capability sm_121) and 128GB of unified CPU/GPU memory. These characteristics break several assumptions baked into vLLM, NCCL, CUTLASS, and Ray. This repository packages the 8 required fixes into a one-command installer so you can go from a fresh DGX Spark to running large language models without wading through days of debugging.

## Why This Exists

The DGX Spark GB10 differs from datacenter GPUs in three fundamental ways:

1. **Blackwell sm_121 compute capability** -- Most prebuilt CUDA libraries (NCCL, CUTLASS) ship kernels for sm_120/sm_120a but not sm_121. Operations that rely on these kernels crash at runtime.
2. **Unified CPU/GPU memory (128GB shared)** -- Ray's out-of-memory killer sees GPU-resident model weights as system memory usage and kills workers that are functioning correctly.
3. **ARM aarch64 architecture** -- Some x86-assumed build paths and binary distributions do not apply.

Without the fixes in this repository, you will encounter cryptic NCCL transport failures, CUTLASS internal errors, Ray OOM kills after successful model loading, and silent IP mismatches in multi-node setups.

---

## Hardware Requirements

| Component | Specification |
|-----------|---------------|
| Platform | NVIDIA DGX Spark GB10 |
| Architecture | ARM aarch64 |
| GPU | NVIDIA Blackwell (sm_121) |
| Memory | 128GB unified CPU/GPU |
| Interconnect (multi-node) | 200GbE QSFP direct cable |
| QSFP interface | `enp1s0f0np0` (primary), `enp1s0f1np1` (secondary) |

## Software Prerequisites

| Component | Version |
|-----------|---------|
| Ubuntu | 24.04+ (aarch64) |
| CUDA | 13.0 |
| Python | 3.12 |
| python3.12-dev | Required for Triton compilation |
| git, build-essential | Required for NCCL build |

---

## Quick Start

### Single Node

```bash
git clone https://github.com/your-org/vllm-gb10.git
cd vllm-gb10
chmod +x install.sh
./install.sh
```

After installation completes, launch a model:

```bash
source ~/vllm-env/bin/activate
source configs/single-node.env
vllm serve /path/to/your/model \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --enforce-eager
```

### Multi-Node (TP=2)

Run the installer on **every node**. Then:

**On the head node:**
```bash
source configs/multi-node-head.env
ray start --head --port=6379
vllm serve /path/to/model \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --trust-remote-code \
  --gpu-memory-utilization 0.80 \
  --max-model-len 16384 \
  --enforce-eager
```

**On each worker node:**
```bash
source configs/multi-node-worker.env
ray start --address=<HEAD_QSFP_IP>:6379
```

See `configs/example-launch.sh` for complete launch commands for tested models.

---

## The 8 Fixes

### Fix 1: NCCL Custom Build for sm_121

**Error without fix:**
```
Message truncated: received 176 bytes instead of 172
...connection refused during multi-node init
```

**Why:** Pre-built NCCL packages (pip `2.28.9`, deb `2.29.3`) do not include GPU kernels for Blackwell sm_121. The GPU-side NCCL collectives silently fail or produce truncated messages because the required compute kernels were never compiled into the distributed binary.

**Fix:** Build NCCL v2.28.9 from source with `-gencode=arch=compute_121,code=sm_121` and place the resulting `libnccl.so` ahead of all other NCCL libraries in `LD_LIBRARY_PATH`. This must be done on **every node** in a multi-node setup.

**Script:** `scripts/build-nccl-sm121.sh`

---

### Fix 2: QSFP 200GbE MTU 9000 Configuration

**Error without fix:**
Extreme packet fragmentation on 200GbE links. NCCL tensor transfers degrade to a fraction of available bandwidth. Multi-node inference becomes unusably slow or times out.

**Why:** The default MTU of 1500 bytes on QSFP interfaces wastes bandwidth on 200GbE direct-connect links. Large tensor transfers benefit enormously from jumbo frames.

**Fix:** Configure MTU 9000 on QSFP interfaces via netplan on all nodes. Verify with `ping -M do -s 8972 <PEER_QSFP_IP>`.

**Script:** `scripts/setup-qsfp.sh`

---

### Fix 3: VLLM_HOST_IP for Ray IP Consistency

**Error without fix:**
```
RuntimeError: Expected 2 unique IPs but got 3
```

**Why:** When a node has multiple network interfaces (LAN + QSFP), Ray registers the node with the QSFP IP while vLLM's `get_ip()` uses socket-based detection and returns the LAN IP. vLLM sees 3 unique IPs instead of the expected 2 and refuses to create the distributed placement group.

**Fix:** Set `VLLM_HOST_IP` to the QSFP IP on every node. vLLM's `network_utils.py:get_ip()` checks this environment variable first, ensuring consistent IP reporting across Ray and vLLM.

**Config:** Set in `configs/multi-node-head.env` and `configs/multi-node-worker.env`.

---

### Fix 4: NCCL Environment Variables for QSFP Interface

**Error without fix:**
NCCL picks the LAN interface for collective operations. Multi-node transfers run over the slow management network instead of the 200GbE QSFP link. Timeouts and degraded performance.

**Why:** NCCL has no way to know which interface is the high-speed interconnect. Without explicit configuration, it selects whichever interface the OS routing table prefers, which is typically the LAN.

**Fix:** Set these environment variables on all processes:
```
NCCL_SOCKET_IFNAME=enp1s0f0np0   # Force QSFP primary interface
UCX_NET_DEVICES=enp1s0f0np0       # UCX transport also uses QSFP
NCCL_NET_GDR_LEVEL=SYS            # GPU Direct RDMA level
NCCL_P2P_DISABLE=1                # No P2P (separate physical nodes)
NCCL_IB_DISABLE=1                 # No InfiniBand (using Socket transport)
```

**Config:** Set in all env files under `configs/`.

---

### Fix 5: CUTLASS FP8 Patch (sm_121 Not in Prebuilt Kernels)

**Error without fix:**
```
RuntimeError: Error Internal
```
when calling `cutlass_scaled_mm` during FP8 model inference.

**Why:** vLLM's prebuilt `_C.abi3.so` contains CUTLASS FP8 kernels compiled for sm_120 and sm_120a, but not sm_121 (GB10 Blackwell). When vLLM detects the GPU supports sm_12x, it tries to use these kernels and crashes.

**Fix:** Patch `w8a8_utils.py` to force `cutlass_fp8_supported()` and `cutlass_block_fp8_supported()` to return `False`. This makes vLLM fall back to Triton-based FP8 kernels, which dynamically compile for the current GPU and work correctly on sm_121. Both the functions AND the module-level constants must be patched, because multiple callers invoke the functions directly.

**Patch:** `patches/cutlass-fp8-sm121.patch`

---

### Fix 6: Package Version Alignment Across Nodes

**Error without fix:**
```
ModuleNotFoundError: No module named 'triton'
```
or model inspection errors from transformers version mismatches between nodes.

**Why:** In a multi-node Ray cluster, the worker node executes model loading and inference code in its local Python environment. If packages are missing or versions differ from the head node, workers crash with import errors or produce incorrect model architectures.

**Fix:** The installer ensures identical package versions on all nodes:
- `vllm==0.18.1`
- `triton==3.6.0`
- `transformers==4.57.6`
- `python3.12-dev` system package

Run `install.sh` on **every node** to guarantee alignment.

---

### Fix 7: RAY memory_usage_threshold for Unified Memory

**Error without fix:**
```
ray.exceptions.OutOfMemoryError: Memory on the node (IP: 192.168.100.10)
was 115.65GB / 121.69GB (0.950369), which exceeds the memory usage threshold of 0.95.
```
This happens AFTER successfully loading all model shards, during the profiling step.

**Why:** The DGX Spark has unified CPU/GPU memory. When model weights are loaded onto the GPU, they appear as regular system memory usage to Ray's memory monitor. Ray's default threshold of 0.95 (95%) triggers the OOM killer even though the system is functioning correctly -- the memory is being used exactly as intended.

**Fix:** Set `RAY_memory_usage_threshold=1.0` to disable Ray's OOM killer. Combine with conservative vLLM settings (`gpu_memory_utilization=0.80`, reduced `max_model_len`) to leave headroom for KV cache and runtime allocations.

**Script:** `patches/ray-memory-threshold.sh`

---

### Fix 8: Streaming Reasoning Tokens

**Error without fix:**
Thinking/reasoning tokens from models like Qwen3 are invisible in streaming output, or raw `<think>` tags leak into the content display.

**Why:** vLLM with `--reasoning-parser qwen3` sends thinking tokens via `delta.reasoning` / `delta.reasoning_content` in SSE chunks, not as `<think>` tags inside `delta.content`. Applications that only parse `<think>` tags miss the reasoning output entirely.

**Fix:** Application-level fix. When consuming vLLM's streaming output with reasoning models, check for `delta.reasoning` and `delta.reasoning_content` fields first. Fall back to `<think>` tag parsing for models that do not use a reasoning parser. See the example in `configs/example-launch.sh` for the correct `--reasoning-parser` flag.

---

## Multi-Node TP=2 Setup Guide

This section walks through setting up two DGX Spark GB10 nodes for tensor-parallel inference.

### Prerequisites
- Two DGX Spark GB10 nodes connected via 200GbE QSFP direct cable
- Both nodes have run `install.sh` successfully
- The model is available at the same path on both nodes (e.g., via NFS or copied)

### Step 1: Configure QSFP Network

On each node, run:
```bash
sudo ./scripts/setup-qsfp.sh
```

Edit the generated netplan config to assign unique QSFP IPs:
- Head node: `192.168.100.10/24`
- Worker node: `192.168.100.11/24`

Verify connectivity:
```bash
ping -M do -s 8972 <PEER_QSFP_IP>
```

### Step 2: Start Ray Head (on head node)

```bash
source ~/vllm-env/bin/activate
source configs/multi-node-head.env

# Set your QSFP IP
export VLLM_HOST_IP=192.168.100.10
export RAY_memory_usage_threshold=1.0

ray start --head --port=6379 --num-gpus=1
```

### Step 3: Start Ray Worker (on worker node)

```bash
source ~/vllm-env/bin/activate
source configs/multi-node-worker.env

# Set YOUR QSFP IP (not the head's)
export VLLM_HOST_IP=192.168.100.11
export RAY_memory_usage_threshold=1.0

ray start --address=192.168.100.10:6379 --num-gpus=1
```

### Step 4: Launch vLLM (on head node)

```bash
export HF_HUB_OFFLINE=1

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
```

Model loading takes approximately 13 minutes (39 safetensors shards). The API server is ready when the health endpoint returns 200:
```bash
curl http://localhost:8000/health
```

---

## Single-Node Setup Guide

For models that fit in a single GB10's 128GB unified memory:

```bash
source ~/vllm-env/bin/activate
source configs/single-node.env
export HF_HUB_OFFLINE=1

vllm serve /path/to/model \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --enforce-eager \
  --enable-prefix-caching
```

---

## Tested Models

| Model | Nodes | TP | gpu_memory_utilization | max_model_len | Notes |
|-------|-------|----|------------------------|---------------|-------|
| Qwen3.5-122B-A10B-FP8 | 2 | 2 | 0.80 | 16384 | Primary production model, uses reasoning + tool parsers |
| Qwen3-Omni-30B | 1 | 1 | 0.85 | 32768 | Launch with `vllm serve --omni` |
| Qwen3.5-35B-A3B | 1 | 1 | 0.85 | 32768 | Lightweight MoE, fits easily on single node |

---

## Troubleshooting

### NCCL: "Message truncated" or connection refused
You are using a pre-built NCCL that lacks sm_121 kernels. Rebuild from source:
```bash
./scripts/build-nccl-sm121.sh
```
Verify the custom library is loaded first in `LD_LIBRARY_PATH`.

### RuntimeError: Error Internal (during FP8 inference)
The CUTLASS FP8 patch has not been applied. Run:
```bash
./patches/cutlass-fp8-sm121.patch
```

### Ray OOM after model loads successfully
Set `RAY_memory_usage_threshold=1.0` in your environment. The unified memory architecture causes Ray to miscount GPU memory as system memory.

### "Expected 2 unique IPs but got 3"
`VLLM_HOST_IP` is not set or is set to the wrong IP. Each node must export `VLLM_HOST_IP` set to its own QSFP IP.

### Triton compilation errors on worker node
Install `python3.12-dev`:
```bash
sudo apt install -y python3.12-dev
```

### Model loading extremely slow over network
NCCL is using the LAN interface instead of QSFP. Verify `NCCL_SOCKET_IFNAME=enp1s0f0np0` is set and the QSFP link is up with MTU 9000.

### nvidia-smi shows [N/A] for memory
This is normal on GB10. The unified memory architecture does not report GPU memory separately via nvidia-smi. Use `gpu_memory_utilization` to control how much memory vLLM allocates.

### vLLM hangs during "Profiling run" step
Reduce `gpu_memory_utilization` (try 0.75) and `max_model_len` (try 8192). The profiling step allocates KV cache and may exceed available memory on unified-memory systems.

---

## Repository Structure

```
vllm-gb10/
├── install.sh                      # Main installer (run on every node)
├── patches/
│   ├── cutlass-fp8-sm121.patch     # CUTLASS FP8 sm_121 patch script
│   └── ray-memory-threshold.sh     # Ray unified memory configuration
├── scripts/
│   ├── build-nccl-sm121.sh         # NCCL source build for sm_121
│   ├── setup-qsfp.sh              # QSFP 200GbE network configuration
│   └── verify-install.sh          # Installation verification
├── configs/
│   ├── single-node.env            # Environment for single-node inference
│   ├── multi-node-head.env        # Environment for Ray head node
│   ├── multi-node-worker.env      # Environment for Ray worker node
│   └── example-launch.sh          # Example launch commands
├── LICENSE                         # Apache 2.0
└── README.md                       # This file
```

## Contributing

Contributions are welcome. If you have a DGX Spark GB10 and have found additional fixes or improvements:

1. Fork the repository
2. Create a feature branch (`git checkout -b fix/description`)
3. Test on actual GB10 hardware -- this project cannot be meaningfully tested on other platforms
4. Submit a pull request with a clear description of the problem and fix

Please include the error message you encountered and the exact vLLM/CUDA/NCCL versions in your PR description.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Acknowledgments

Developed on NVIDIA DGX Spark GB10 hardware. All fixes were discovered through hands-on debugging of real multi-node deployments running Qwen3.5-122B-A10B-FP8 with tensor parallelism.

# vllm-windows-build

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Platform: Windows](https://img.shields.io/badge/Platform-Windows%2010%2F11-blue)
![vLLM: v0.14.1](https://img.shields.io/badge/vLLM-v0.14.1-orange)
![CUDA: 12.6](https://img.shields.io/badge/CUDA-12.6-76B900)
![Python: 3.10](https://img.shields.io/badge/Python-3.10-3776AB)

**Native Windows build of vLLM — no WSL, no Docker, no Linux VM.** 26 patched files, 370/370 CUDA kernels compiled, tested and running on MSVC 2022 + CUDA 12.6.

vLLM is the most popular open-source LLM serving engine, but it officially only supports Linux. This repo provides a complete patchset that gets vLLM v0.14.1 compiling and running natively on Windows with full CUDA acceleration.

---

## Quick Start

**1. Clone vLLM and check out the base version**

```batch
git clone https://github.com/vllm-project/vllm.git vllm-source
cd vllm-source
git checkout v0.14.1
```

**2. Apply the patch**

```batch
git apply ..\vllm-windows.patch
```

**3. Build**

Open a Visual Studio Developer Command Prompt (or run `vcvars64.bat` first), then:

```batch
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set TORCH_CUDA_ARCH_LIST=8.6
set VLLM_TARGET_DEVICE=cuda
set MAX_JOBS=8

pip install -e . --no-build-isolation -v
```

Or use the included build script:

```batch
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
build.bat
```

**4. Post-build: copy flash-attn wrappers**

The build places compiled `.pyd` files but not the Python interface code. The build script handles this automatically, but if you built manually:

```batch
xcopy /E /Y ".deps\vllm-flash-attn-src\vllm_flash_attn\*.py" "vllm\vllm_flash_attn\"
xcopy /E /Y ".deps\vllm-flash-attn-src\vllm_flash_attn\layers\*" "vllm\vllm_flash_attn\layers\"
xcopy /E /Y ".deps\vllm-flash-attn-src\vllm_flash_attn\ops\*" "vllm\vllm_flash_attn\ops\"
```

**5. Run**

```batch
set VLLM_ATTENTION_BACKEND=FLASH_ATTN
set VLLM_HOST_IP=127.0.0.1

python vllm_launcher.py --model E:\models\Qwen2.5-1.5B-Instruct --port 8100
```

That's it. The server starts on `http://127.0.0.1:8100` with an OpenAI-compatible API.

---

## Usage

### Why the Launcher?

vLLM's built-in `vllm.entrypoints.openai.api_server` uses `AsyncMPClient` with ZMQ multiprocessing, which doesn't work on Windows (no `fork`, no IPC sockets). The included `vllm_launcher.py` works around this by:

1. Stubbing `uvloop` (not available on Windows)
2. Using vLLM's synchronous `LLM` class (which uses `InprocClient` — the patched in-process engine)
3. Wrapping it in a lightweight FastAPI server with OpenAI-compatible `/v1/chat/completions` endpoint
4. Supporting both streaming (SSE) and non-streaming responses

### Running the Server

```batch
set VLLM_ATTENTION_BACKEND=FLASH_ATTN
set VLLM_HOST_IP=127.0.0.1

python vllm_launcher.py ^
    --model E:\models\Qwen2.5-1.5B-Instruct ^
    --port 8100 ^
    --gpu-memory-utilization 0.6 ^
    --max-num-seqs 64 ^
    --enforce-eager
```

| Flag | Default | Description |
|------|---------|------------|
| `--model` | (required) | HuggingFace model path or ID |
| `--port` | 8100 | Server port |
| `--host` | 127.0.0.1 | Bind address |
| `--gpu-memory-utilization` | 0.6 | Fraction of GPU VRAM to pre-allocate (0.1 - 1.0) |
| `--max-model-len` | (auto) | Max sequence length |
| `--max-num-seqs` | 64 | Max concurrent sequences in the KV cache |
| `--enforce-eager` | off | Disable CUDA graphs (required on Windows — no Triton) |
| `--tensor-parallel-size` | 1 | Number of GPUs (must be 1 on Windows) |

`--enforce-eager` is automatically added on Windows by the launcher's parent process, but include it if running the launcher directly.

### API Endpoints

Once running, the server exposes:

```
GET  /health                    # Returns {"status": "ok"}
GET  /v1/models                 # List loaded models
POST /v1/chat/completions       # OpenAI-compatible chat completions
```

### Calling the API

```python
import requests

response = requests.post("http://127.0.0.1:8100/v1/chat/completions", json={
    "model": "Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": False,
})
print(response.json()["choices"][0]["message"]["content"])
```

Works with any OpenAI-compatible client library (openai-python, LangChain, LiteLLM, etc.) — just point the `base_url` at `http://127.0.0.1:8100/v1`.

### VRAM and Memory

vLLM pre-allocates a large KV cache on startup using PagedAttention. This is by design — it's how vLLM achieves high concurrent throughput. The tradeoff is that even a small model uses significant VRAM:

| Model Size | Weights | KV Cache (0.6 util) | Total VRAM | Card |
|-----------|---------|-------------------|------------|------|
| 1.5B Q8 | ~1.8 GB | ~12 GB | ~14 GB | 24 GB RTX 3090 |
| 7B Q4 | ~4 GB | ~12 GB | ~16 GB | 24 GB RTX 3090 |

Lower `--gpu-memory-utilization` if you hit OOM during sampler warmup. Lower `--max-num-seqs` to reduce KV cache size.

---

## What's in the Patch

26 files modified, +254 lines / -121 lines. Every change is guarded behind `#ifdef _MSC_VER`, `sys.platform == "win32"`, or similar checks — nothing breaks on Linux.

### Build System Fixes

| File | What Changed |
|------|-------------|
| `CMakeLists.txt` | Force CUDA toolkit from `CUDA_HOME` (prevents wrong toolkit when multiple versions installed), add MSVC-specific include paths, `/Zc:preprocessor` flag, link `CUDA::cublas` |
| `cmake/utils.cmake` | Quote paths with spaces for `file(REAL_PATH)` |
| `setup.py` | Allow Windows CUDA builds (was hardcoded to `"empty"`), backslash→forward-slash paths for CMake, `nvcc.exe` detection |

### CUDA Kernel Fixes (MSVC Compatibility)

| File | What Changed |
|------|-------------|
| `csrc/quantization/utils.cuh` | Variable template → function template (MSVC can't apply `__host__`/`__device__` to variable templates) |
| `csrc/mamba/mamba_ssm/selective_scan_fwd.cu` | Replaced nested `BOOL_SWITCH` lambdas with explicit template dispatch (MSVC doesn't propagate constexpr through lambda captures) |
| `csrc/quantization/activation_kernels.cu` | Designated initializers `.x = val` → positional `{val}`, `__int128_t` → `int4` |
| `csrc/core/math.hpp` | `__builtin_clz` → portable bit-twiddling replacement |
| `csrc/quantization/awq/gemm_kernels.cu` | `__asm__ __volatile__` → `asm volatile` (MSVC PTX inline assembly syntax) |
| `csrc/quantization/gptq_allspark/allspark_qgemm_w8a16.cu` | `or` keyword → `\|\|` operator |
| `csrc/quantization/fused_kernels/layernorm_utils.cuh` | `quant_type_max_v<T>` → `quant_type_max_v<T>()` (variable template → function template call) |
| `csrc/quantization/fused_kernels/quant_conversions.cuh` | Same variable template fix |
| `csrc/quantization/w8a8/fp8/common.cu`, `common.cuh` | Same variable template fix |
| `csrc/quantization/marlin/sparse/common/base.h` | `typedef unsigned int uint;` for MSVC |
| `csrc/attention/merge_attn_states.cu` | `std::isinf` → `isinf` (CUDA device context), add `uint` typedef |
| `csrc/cumem_allocator.cpp` | `ssize_t` → `SSIZE_T` via `BaseTsd.h` |
| `csrc/fused_qknorm_rope_kernel.cu` | Add `uint` typedef for MSVC |
| `csrc/gptq_marlin/generate_kernels.py` | Flat `if` chains instead of `else if` (MSVC C1061 "blocks nested too deeply" with 700+ branches) |
| `csrc/moe/marlin_moe_wna16/generate_kernels.py` | Same flat `if` fix |

### Runtime Fixes (Windows Platform)

| File | What Changed |
|------|-------------|
| `vllm/model_executor/layers/fused_moe/routed_experts_capturer.py` | `fcntl.flock()` → `msvcrt.locking()` (Unix file locking API doesn't exist on Windows) |
| `vllm/utils/network_utils.py` | ZMQ IPC transport → `tcp://127.0.0.1` (Unix domain sockets not available) |
| `vllm/utils/system_utils.py` | Force `spawn` multiprocessing (`fork` doesn't exist on Windows) |
| `vllm/v1/engine/core_client.py` | Force `InprocClient` (multiprocess ZMQ fails with `spawn` context) |
| `vllm/distributed/parallel_state.py` | **Biggest fix.** Gloo TCP/UV transport isn't compiled in PyTorch Windows builds. Uses PyTorch's `FakeProcessGroup` backend with `FileStore` for single-GPU operation. |
| `vllm/entrypoints/openai/api_server.py` | Guard `SO_REUSEPORT` (doesn't exist on Windows) |
| `requirements/cuda.txt` | Comment out `flashinfer-python` (not available on Windows) |

---

## Required Environment Variables

```batch
:: Must be set for every vLLM run on Windows
set VLLM_ATTENTION_BACKEND=FLASH_ATTN
set VLLM_HOST_IP=127.0.0.1

:: Optional but recommended
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0
```

---

## System Requirements

| Component | Requirement |
|-----------|------------|
| OS | Windows 10 or 11 (64-bit) |
| GPU | NVIDIA with CUDA Compute Capability 7.0+ (Volta or newer) |
| CUDA Toolkit | 12.6 (tested; 12.x should work) |
| Compiler | Visual Studio 2022 with C++ Desktop workload |
| Python | 3.10.x (tested with 3.10.6) |
| PyTorch | 2.9.1+cu126 |
| RAM | 32 GB recommended (compilation is memory-heavy) |
| Disk | ~20 GB for build artifacts |

---

## Known Limitations

- **No Triton on Windows** — only `FLASH_ATTN` attention backend works. `TRITON_ATTN` and `FLEX_ATTENTION` both require Triton which has no Windows support.
- **Single GPU only** — no NCCL on Windows means no multi-GPU tensor parallelism. `world_size` must be 1.
- **No FlashInfer** — `flashinfer-python` doesn't publish Windows wheels. Commented out in requirements.
- **VRAM pre-allocation** — vLLM's PagedAttention aggressively pre-allocates KV cache. A 1.5B model can use 14+ GB at `gpu_memory_utilization=0.6`. Lower the value or set `max_num_seqs` if you hit OOM.

---

## Build Notes

The build compiles 370 CUDA/C++ source files. On an RTX 3090 system with 64 GB RAM and `MAX_JOBS=8`, expect roughly 30-45 minutes.

### Multiple CUDA Toolkits

If you have multiple CUDA versions installed (common if you use both PyTorch cu126 and the CUDA 13.1 compiler), the CMake patch forces toolkit selection from `CUDA_HOME`. Make sure this points at the version matching your PyTorch build:

```batch
:: Check which CUDA your PyTorch was built with
python -c "import torch; print(torch.version.cuda)"

:: Set CUDA_HOME to match
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
```

### GPU Compute Capability

Change `TORCH_CUDA_ARCH_LIST` to match your GPU:

| GPU Series | Compute Capability |
|-----------|-------------------|
| RTX 20xx (Turing) | 7.5 |
| RTX 30xx (Ampere) | 8.6 |
| RTX 40xx (Ada Lovelace) | 8.9 |
| RTX 50xx (Blackwell) | 12.0 |

---

## Directory Structure

```
vllm-windows-build/
├── vllm-windows.patch    # Complete git patch (apply to v0.14.1)
├── vllm_launcher.py      # OpenAI-compatible server wrapper for Windows
├── build.bat             # Automated build script
├── PATCHES.md            # Detailed per-file patch reference
├── LICENSE
└── README.md
```

---

## Tested With

- Qwen2.5-1.5B-Instruct Q8 GGUF — loaded via vLLM OpenAI-compatible API, 202.8 tok/s throughput
- RTX 3090 24 GB, `gpu_memory_utilization=0.6`, `max_num_seqs=64`
- Windows 10 Pro 10.0.19045, MSVC 19.43, CUDA 12.6, Python 3.10.6, PyTorch 2.9.1+cu126

---

## Credits

| Project | Description | License |
|---------|------------|---------|
| [vLLM](https://github.com/vllm-project/vllm) | High-throughput LLM serving engine | Apache 2.0 |
| [PyTorch](https://github.com/pytorch/pytorch) | Deep learning framework | BSD-3 |
| [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) | NVIDIA GPU computing | NVIDIA EULA |
| [Flash Attention](https://github.com/Dao-AILab/flash-attention) | Fast attention implementation | BSD-3 |

Built with [Claude Opus 4.6](https://claude.ai) as pair-programming partner.

## License

MIT License. See [LICENSE](LICENSE) for details.

The patch modifies vLLM source code which is licensed under Apache 2.0. This repo contains only the patch (derivative work) and build tooling, not the full vLLM source.

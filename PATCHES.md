# Patch Reference

Detailed breakdown of every change in the Windows patchset, organized by category.

## Build Environment

- **Base**: vLLM v0.14.1 (commit `d7de043`)
- **Compiler**: MSVC 2022 (Visual Studio Community)
- **CUDA**: 12.6 (v13.1 also installed — caused toolkit conflicts)
- **Python**: 3.10.6, PyTorch 2.9.1+cu126
- **GPU**: RTX 3090 (sm_86)
- **Build command**: `pip install -e . --no-build-isolation`

---

## Build-Time Patches (MSVC/CUDA Compatibility)

### CMakeLists.txt (+31 lines)

**CUDA toolkit version conflict** — When multiple CUDA versions are installed, CMake picked up CUDA 13.1 (compiler-only install) instead of 12.6. Fixed by forcing `CUDA_TOOLKIT_ROOT_DIR`, `CUDAToolkit_ROOT`, and `CUDA_BIN_PATH` from `CUDA_HOME` env var before `find_package(Torch)`.

**MSVC compiler flags** — Added `#ifdef MSVC` block for:
- Explicit CUDA include paths (quoted for spaces in `Program Files`)
- `_CRT_DECLARE_NONSTDC_NAMES=1` for POSIX compat
- `USE_CUDA` define (activates CCCL workaround in PyTorch's `compiled_autograd.h`)
- `/Zc:preprocessor` for correct variadic macro handling (fixes nested `BOOL_SWITCH` macros)

**Link libraries** — Added `CUDA::cublas` to the `_C` extension target.

### cmake/utils.cmake (+2/-1 lines)

Quoted `${EXECUTABLE}` in `file(REAL_PATH)` call — paths with spaces (like `C:\Program Files\...`) broke without quotes.

### setup.py (+38 lines)

- Allowed Windows CUDA builds when `VLLM_TARGET_DEVICE=cuda` is explicitly set (was hardcoded to `"empty"` on non-Linux)
- Backslash → forward-slash conversion for all paths passed to CMake
- `nvcc.exe` instead of `nvcc` for Windows

---

## CUDA Kernel Fixes

### csrc/quantization/utils.cuh — Variable Template

MSVC's `nvcc` can't apply `__host__`/`__device__` attributes to variable templates. Changed from:

```cpp
template <typename T>
MAYBE_HOST_DEVICE static constexpr T quant_type_max_v = quant_type_max<T>::val();
```

To a function template:

```cpp
template <typename T>
MAYBE_HOST_DEVICE static inline constexpr T quant_type_max_v() {
    return quant_type_max<T>::val();
}
```

All call sites updated from `quant_type_max_v<T>` to `quant_type_max_v<T>()`.

### csrc/mamba/mamba_ssm/selective_scan_fwd.cu — Nested Lambda Dispatch

MSVC doesn't propagate `constexpr` values through lambda captures. The original code used nested `BOOL_SWITCH` macros (each generating a lambda) to dispatch on three boolean template params. Replaced with:

1. A new helper function template `selective_scan_fwd_dispatch<..., kIsEvenLen, kHasZ, kVarlen>()` with all booleans as template args
2. An explicit 8-way `if/else` dispatch tree in `selective_scan_fwd_launch()`

### csrc/quantization/activation_kernels.cu — MSVC Type Issues

- **Designated initializers**: `__nv_bfloat16_raw{.x = 17376}` → `__nv_bfloat16_raw{17376}` (C99 designated initializers not supported in MSVC C++ mode)
- **`__int128_t`**: Not available in MSVC. Replaced with `int4` (CUDA's built-in 128-bit vector type) for shared memory pointers
- **`__int64_t`**: Replaced with `int64_t` (standard type)

### csrc/core/math.hpp — Compiler Builtin

`__builtin_clz` (GCC/Clang count-leading-zeros builtin) doesn't exist in MSVC. Replaced with portable bit-twiddling:

```cpp
uint32_t v = num - 1;
v |= v >> 1;
v |= v >> 2;
v |= v >> 4;
v |= v >> 8;
v |= v >> 16;
return v + 1;
```

### csrc/quantization/awq/gemm_kernels.cu — Inline Assembly

`__asm__ __volatile__` → `asm volatile` (MSVC accepts the standard keyword form for PTX inline assembly in CUDA files).

### csrc/quantization/gptq_allspark/allspark_qgemm_w8a16.cu — Keyword

`or` → `||` (the `or` alternative token for `||` is not enabled by default in MSVC).

### csrc/gptq_marlin/generate_kernels.py and csrc/moe/marlin_moe_wna16/generate_kernels.py — Nesting Depth

These Python scripts generate C++ kernel selector code with 700+ branches using `if/else if` chains. MSVC hits `C1061: compiler limit: blocks nested too deeply`. Fixed by generating flat `if` statements instead of `else if`.

### Other CUDA Fixes

- `csrc/attention/merge_attn_states.cu` — `std::isinf` → `isinf` in device code, added `uint` typedef
- `csrc/cumem_allocator.cpp` — `ssize_t` not defined on MSVC, added `#include <BaseTsd.h>` and `typedef SSIZE_T ssize_t`
- `csrc/fused_qknorm_rope_kernel.cu` — Added `uint` typedef for MSVC
- `csrc/quantization/marlin/sparse/common/base.h` — Added `uint` typedef for MSVC

---

## Runtime Patches (Windows Platform)

### 1. File Locking — routed_experts_capturer.py

`fcntl` is Unix-only. Added conditional import with `msvcrt.locking()` fallback on Windows.

### 2. ZMQ IPC Transport — network_utils.py

IPC transport (Unix domain sockets) not supported on Windows. `get_open_zmq_ipc_path()` returns `tcp://127.0.0.1:{port}` on Windows.

### 3. Multiprocessing Context — system_utils.py

`fork` doesn't exist on Windows. Forces `spawn` context.

### 4. Engine Core Client — v1/engine/core_client.py

`SyncMPClient` uses multiprocess ZMQ which fails with `spawn` (can't pickle socket objects). Forces `InprocClient` on Windows, which runs the engine in-process.

### 5. Distributed Backend — distributed/parallel_state.py

**This was the biggest blocker.** PyTorch Windows builds have `GLOO_HAVE_TRANSPORT_TCP=false` and `GLOO_HAVE_TRANSPORT_UV=false`. Tested all init methods (`file://`, `env://`, `tcp://`, `GLOO_SOCKET_IFNAME`) — all failed with `makeDeviceForHostname(): unsupported gloo device`.

**Solution**: Found `torch.testing._internal.distributed.fake_pg.FakeProcessGroup`. Importing it registers a `"fake"` distributed backend.

Added:
- `_get_cpu_backend()` helper — returns `"fake"` on Windows, `"gloo"` on Linux
- `init_distributed_environment()` — uses `FileStore` + fake backend on Windows
- `GroupCoordinator.__init__()` — uses fake backend for CPU group creation

This works because vLLM only needs the distributed API for rank/world_size bookkeeping when running single-GPU.

### 6. Socket Options — api_server.py

`SO_REUSEPORT` doesn't exist on Windows. Added `hasattr` guard.

### 7. Flash Attention Wrappers

Build only places compiled `.pyd` files in `vllm/vllm_flash_attn/`. The Python interface code (`__init__.py`, `flash_attn_interface.py`, `layers/`, `ops/`) must be copied from `.deps/vllm-flash-attn-src/vllm_flash_attn/`.

### 8. Attention Backend

Both `TRITON_ATTN` and `FLEX_ATTENTION` import Triton (not available on Windows):
- `TRITON_ATTN` → directly imports `triton` → fails
- `FLEX_ATTENTION` → uses `torch.compile`/inductor → generates Triton kernels → fails
- `FLASH_ATTN` → uses compiled CUDA kernels directly → **works**

Must set `VLLM_ATTENTION_BACKEND=FLASH_ATTN`.

---

## What's NOT Patched (Known Limitations)

- **Triton** — No Windows support upstream. Eliminates 2 of 3 attention backends.
- **NCCL** — No Windows support. No multi-GPU tensor parallelism.
- **FlashInfer** — `nvidia-cutlass-dsl-libs-base` not available on Windows. Commented out in requirements.
- **Gloo distributed** — Broken in PyTorch Windows builds. Worked around with fake backend (single-GPU only).

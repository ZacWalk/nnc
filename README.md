# nnc — Neural Net Compiler

I have always been fascinated by all the memory shuffling that happens to run a model on a GPU. What happens if you just run it on the CPU optimized for SIMD instructions. This is my learning project that JIT-compiles a neural-net computation graph into machine code tuned for the host CPU, then uses it to run **Gemma 3n inference end-to-end** on its own minimal tensor runtime.

![nnc prompt demo](ncc-prompt.gif)

No third-party JIT libraries, no ML frameworks — just a C++20 compiler
(MSVC on Windows, g++/clang on Linux) and hand-rolled x86-64 / AVX2 / FMA / F16C encoders.

Runs on **Windows x64 and Linux x64** (including WSL) from a single source
tree. All OS-specific calls live behind a small `sys.h` shim; the JIT
kernels themselves are bit-identical on both platforms.

![nnc architecture](nnc-diagram.png)


## What it does

- Loads a Gemma 3n GGUF model (BF16 weights) by mmapping the file and
  building `nnc_tensor` descriptors that point directly at the mapped bytes.
- **Optional in-place Q8_0 quantization** of the dominant weight tensors
  (per-layer Q/K/V/O, FFN gate/up/down, PLE, and `per_layer_model_proj`)
  after load. Cuts weight RAM ~1.78× (3.6 GB → 2.0 GB on E2B) and decode
  latency ~1.5× on a DRAM-bandwidth-bound CPU. **Enabled by default**;
  pass `--bf16` to keep raw BF16.
- Builds the per-token forward graph (embed → transformer blocks → final
  norm → logits) against the nnc runtime, with a per-token KV cache and the
  Gemma 3n PLE (per-layer-input embedding) gate.
- Emits specialized machine-code kernels for the hot ops directly into
  executable pages (Windows: `VirtualAlloc` / `VirtualProtect`; Linux:
  `mmap` / `mprotect`, both behind `sys_alloc_exec_pages` /
  `sys_protect_rx`):
    - FP32 dot, FP32 gemv
    - Fused FP16-weight / FP32-activation gemv (with bias-add and optional GELU)
    - Fused **BF16-weight / FP32-activation** gemv, 4-row variant with
      shared x-tile when `rows % 4 == 0`
    - **Q8_0-weight / FP32-activation** gemv (32-element blocks, scale per
      block, `vpmovsxbd` + `vcvtdq2ps` + `vmulps` + `vfmadd231ps`)
- Caches each kernel by `(op, dtype, rows, cols)` so it is JITted once on
  first use and reused for the rest of the run. Cache is guarded by a
  `shared_mutex` (read-mostly).
- Static **worker pool** (default 8 workers + main, override with
  `NNC_THREADS`) parallelises every gemv with `rows >= 256 && cols >= 256`
  along the row axis. Sub-µs dispatch via spin-then-yield.
- Runs softmax, layernorm, RMSNorm, RoPE, soft-cap, SwiGLU, BF16 row
  embedding lookup, and elementwise add as in-house AVX2/FMA SIMD kernels;
  anything else falls back to scalar reference loops in the runtime.
- Interactive chat REPL: wraps each user line in Gemma's
  `<start_of_turn>user … <end_of_turn>\n<start_of_turn>model\n` template,
  greedy-decodes until `<end_of_turn>`, and streams detokenized UTF-8 text
  to the console (`SetConsoleOutputCP(CP_UTF8)` on Windows; Linux
  terminals are already UTF-8).

Only Gemma 3n GGUF models are supported. The legacy GPT-2 `.bin` loader,
BPE/JSON vocab path, and Q4_0 / Q4_1 quantized types have been removed.

## Requirements

- x86-64 CPU with **AVX2 + FMA + F16C** (detected at startup; older CPUs are refused).
  AVX-512 is intentionally not used.
- One of:
  - **Windows x64** with MSVC v145+ (Visual Studio 2022).
  - **Linux x64** (or WSL2) with `g++` 10+ or `clang++` 12+ and `make`.
- C++20 / C17.

## Build

### Windows (MSVC, x64)

```powershell
$msbuild = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" `
    -latest -requires Microsoft.Component.MSBuild -find MSBuild\**\Bin\MSBuild.exe
& $msbuild nnc.sln /p:Configuration=Debug /p:Platform=x64 /m /v:minimal /nologo
```

Output: `exe\nnc-d.exe` (Debug) or `exe\nnc.exe` (Release).

### Linux / WSL (g++ or clang++, x86-64 with AVX2 + FMA + F16C)

```bash
make            # release  -> exe/nnc
make debug      # debug    -> exe/nnc-d
make test       # debug build + run --test
make clean
```

All OS-specific calls live behind `src/sys.h`; `src/sys_win.cpp` is built
on Windows and `src/sys_linux.cpp` is built on Linux. Each TU is
`#if`-guarded so both files can sit in the project on either platform.
The JIT kernels are written assuming the Windows x64 calling convention;
on Linux every kernel prepends a tiny SysV→Win64 register shuffle so the
same encoders are used unchanged.

## Run

Inference (interactive REPL — default model is
`models/gemma-4-E2B-it-BF16.gguf`, override with `-m`):

```powershell
# Windows
.\exe\nnc-d.exe
```

```bash
# Linux / WSL
./exe/nnc-d
```

The app prints `nnc: press Ctrl-C to exit.`, then loops on a `>` prompt.
Each line is run through the Gemma chat template and the model reply is
streamed back as plain text.



Tests (single binary, no framework):

```
exe/nnc-d --test           # Linux
.\exe\nnc-d.exe --test     # Windows
```

Inspection / smoke-test modes: `--gguf-info`, `--gguf-stats`,
`--gemma-info`, `--gemma-probe`, `--gemma-forward`, `--gemma-tokenize`,
`--gemma-prompt`, `--gemma-gen`.

Global flags:

- `--bf16` / `-bf16` — keep weights as raw BF16 (disables Q8_0 quantization).
- `--q8` / `-q8` — explicit Q8_0 (this is the default; flag kept for back-compat).
- `-m <file.gguf>` — model path. Defaults to `models\gemma-4-E2B-it-BF16.gguf`.
- `-s <seed>`, `-n <n_predict>`, `-b <batch>`, `--top_k`, `--top_p`,
  `--temp` — parsed by `gpt_params_parse`. The sampling flags are
  currently unused; decode is greedy/argmax.

Environment:

- `NNC_THREADS=<N>` — override worker pool size (default = min(8, CPUs-1)).

## Repository layout

Flat by design — everything lives directly under `src/`.

| File | Purpose |
| --- | --- |
| `main.cpp` | CLI entry, argument dispatch, Gemma model load + REPL driver |
| `runtime.{h,cpp}` | Arena, tensor, computation graph, forward dispatch |
| `nn_ops.{h,cpp}` | SIMD kernels + JIT-routed gemv + graph-level fusion |
| `utils.{h,cpp}` | CLI parsing |
| `gguf.{h,cpp}` | GGUF v2/v3 header + KV + tensor table parser, mmap |
| `gemma.{h,cpp}` | Gemma 3n loader, tokenizer, KV cache, forward pass |
| `jit_buffer.{h,cpp}` | Executable-memory allocator (W^X via the sys layer) |
| `emitter_x64.{h,cpp}` | Raw byte / REX / ModR/M / SIB encoders, plus the SysV→Win64 ABI shim |
| `emitter_avx2.{h,cpp}` | VEX-encoded AVX2 + FMA + F16C encoders |
| `jit_kernel.{h,cpp}` | CPU detection, typed wrappers, kernel cache |
| `jit_ops.{h,cpp}` | High-level kernel builders (dot, gemv, …) |
| `sys.h` / `sys_win.cpp` / `sys_linux.cpp` | OS abstraction: console, exec-page allocator, mmap, CPUID |
| `tests.cpp` | All tests (app + JIT) in one TU, run via `--test` |

## Status

Gemma 3n E2B runs end-to-end as an interactive chat REPL. All **22 tests**
pass.

Every weight projection in the model flows through a JITted, shape-
specialized BF16- or Q8_0-weight × FP32 gemv kernel, parallelised across
the worker pool when both dims are large.

Measured perf on a recent desktop CPU (E2B Release, synthetic 8-tok prompt
+ 64-tok decode, 3-run avg):

| weight dtype | prefill (ms/tok) | decode (ms/tok) | weight RAM |
| --- | --- | --- | --- |
| BF16 (`--bf16`) | 180.5 | 72.8 | 3615.8 MB |
| Q8_0 (default)  | 113.8 | 49.5 | 2033.9 MB |
| **speedup**     | **1.58×** | **1.47×** | 1.78× |

Decode is DRAM-bandwidth bound on the BF16 path (plateaus by 4 threads).
Q8_0 reduces per-token weight reads by 1.78× and recovers most of that as
latency. The remaining gap is the lm_head projection and input embedding
lookup, which both share `token_embd` and stay BF16.

Windows vs Linux: steady-state decode is essentially identical (within
~5 %) — the JIT kernels are byte-for-byte the same code on both, and
decode is bandwidth-bound. Cold prefill is noticeably slower under WSL
when the `.gguf` lives on the Windows side of `/mnt/c` because the
initial mmap pages cross the 9P bridge; placing the model on the Linux
filesystem (or running native Linux) closes that gap.

This is a personal learning project, not a production runtime.

# nnc — Neural Net Compiler

A small, from-scratch learning project that JIT-compiles a neural-net
computation graph into machine code tuned for the host CPU, then uses it to
run **GPT-2 inference end-to-end** on its own minimal tensor runtime.

No third-party JIT libraries, no ML frameworks — just MSVC, the Windows SDK,
and hand-rolled x86-64 / AVX2 / FMA / F16C encoders.

## What it does

- Loads a GPT-2 117M model file (FP16 weights, on-disk `ggml-`-prefixed
  format) into an arena-allocated tensor graph.
- Builds the per-token forward graph (embed → 12× transformer block → final
  layernorm → logits).
- Emits specialized machine-code kernels for the hot ops (FP32 dot, FP32
  gemv, fused FP16-weight / FP32-activation gemv with bias-add and optional
  GELU) directly into executable pages via `VirtualAlloc` /
  `VirtualProtect`.
- Caches each kernel by `(op, dtype, shape)` so it is JITted once and reused
  for the rest of the run.
- Runs softmax, layernorm, GELU, and elementwise add as in-house AVX2/FMA
  SIMD kernels; everything else falls back to scalar reference loops in the
  runtime.

## Requirements

- Windows x64
- MSVC v145+ (Visual Studio 2022), C++20 / C17
- A CPU with **AVX2 + FMA + F16C** (detected at startup; older CPUs are
  refused)

AVX-512 is intentionally not used.

## Build

```powershell
$msbuild = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" `
    -latest -requires Microsoft.Component.MSBuild -find MSBuild\**\Bin\MSBuild.exe
& $msbuild nnc.sln /p:Configuration=Debug /p:Platform=x64 /m /v:minimal /nologo
```

Output: `exe\nnc-d.exe` (Debug) or `exe\nnc.exe` (Release).

## Run

Inference:

```powershell
.\exe\nnc-d.exe -m models\ggml-model-gpt-2-117M.bin -p "Hello, my name is" -n 20 --seed 1
```

Tests (single binary, no framework):

```powershell
.\exe\nnc-d.exe --test
```

Useful flags: `-s/--seed`, `-n` (tokens to predict), `-b` (prompt batch
size), `--top_k`, `--top_p`, `--temp`.

## Repository layout

Flat by design — everything lives directly under `src/`.

| File | Purpose |
| --- | --- |
| `main.cpp` | CLI entry, argument dispatch, GPT-2 model load + driver |
| `runtime.{h,cpp}` | Arena, tensor, computation graph, forward dispatch |
| `nn_ops.{h,cpp}` | SIMD kernels + JIT-routed gemv + graph-level fusion |
| `utils.{h,cpp}` | CLI parsing, vocab, BPE tokenizer, sampler, quantization |
| `jit_buffer.{h,cpp}` | Executable-memory allocator (VirtualAlloc/Protect) |
| `emitter_x64.{h,cpp}` | Raw byte / REX / ModR/M / SIB encoders |
| `emitter_avx2.{h,cpp}` | VEX-encoded AVX2 + FMA + F16C encoders |
| `jit_kernel.{h,cpp}` | CPU detection, typed wrappers, kernel cache |
| `jit_ops.{h,cpp}` | High-level kernel builders (dot, gemv, …) |
| `tests.cpp` | All tests (app + JIT) in one TU, run via `--test` |

## Status

GPT-2 117M runs end-to-end and produces bit-identical output to the prior
scalar baseline. Current Debug-build performance is ≈11–12 ms/token on the
reference machine. All 11 tests pass.

This is a personal learning project, not a production runtime.

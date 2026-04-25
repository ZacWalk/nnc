# nnc — Neural Net Compiler

I have always been fascinated by all the memory shuffling that happens to run a model on a GPU. What happens if you just run it on the CPU optimized for SIMD instructions. This is my learning project that JIT-compiles a neural-net computation graph into machine code tuned for the host CPU, then uses it to run **Gemma 3n inference end-to-end** on its own minimal tensor runtime.

No third-party JIT libraries, no ML frameworks — just MSVC, the Windows SDK,
and hand-rolled x86-64 / AVX2 / FMA / F16C encoders.

## What it does

- Loads a Gemma 3n GGUF model (BF16 weights) by mmapping the file and
  building `nnc_tensor` descriptors that point directly at the mapped bytes.
- Builds the per-token forward graph (embed → transformer blocks → final
  norm → logits) against the nnc runtime, with a per-token KV cache and the
  Gemma 3n PLE (per-layer-input embedding) gate.
- Emits specialized machine-code kernels for the hot ops directly into
  executable pages via `VirtualAlloc` / `VirtualProtect`:
    - FP32 dot, FP32 gemv
    - Fused FP16-weight / FP32-activation gemv (with bias-add and optional GELU)
    - Fused **BF16-weight / FP32-activation** gemv (used by every Q/K/V/O,
      gate/up/down, PLE, and final-logits projection in Gemma)
- Caches each kernel by `(op, dtype, rows, cols)` so it is JITted once on
  first use and reused for the rest of the run.
- Runs softmax, layernorm, RMSNorm, RoPE, soft-cap, SwiGLU, BF16 row
  embedding lookup, and elementwise add as in-house AVX2/FMA SIMD kernels;
  anything else falls back to scalar reference loops in the runtime.
- Interactive chat REPL: wraps each user line in Gemma's
  `<start_of_turn>user … <end_of_turn>\n<start_of_turn>model\n` template,
  greedy-decodes until `<end_of_turn>`, and streams detokenized UTF-8 text
  to the console (`SetConsoleOutputCP(CP_UTF8)`).

Only Gemma 3n GGUF models are supported. The legacy GPT-2 `.bin` loader,
BPE/JSON vocab path, and Q4_0 / Q4_1 quantized types have been removed.

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

Inference (interactive REPL — default model is
`models\gemma-4-E2B-it-BF16.gguf`, override with `-m`):

```powershell
.\exe\nnc-d.exe
```

The app prints `nnc: press Ctrl-C to exit.`, then loops on a `>` prompt.
Each line is run through the Gemma chat template and the model reply is
streamed back as plain text.

Tests (single binary, no framework):

```powershell
.\exe\nnc-d.exe --test
```

Inspection / smoke-test modes: `--gguf-info`, `--gguf-stats`,
`--gemma-info`, `--gemma-probe`, `--gemma-forward`, `--gemma-tokenize`,
`--gemma-prompt`, `--gemma-gen`.

Useful flags: `-s/--seed`, `-n` (max tokens generated per turn, default 256),
`-b` (prompt batch size), `--top_k`, `--top_p`, `--temp` (sampling flags are
parsed but currently unused; the REPL uses argmax decode).

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
| `jit_buffer.{h,cpp}` | Executable-memory allocator (VirtualAlloc/Protect) |
| `emitter_x64.{h,cpp}` | Raw byte / REX / ModR/M / SIB encoders |
| `emitter_avx2.{h,cpp}` | VEX-encoded AVX2 + FMA + F16C encoders |
| `jit_kernel.{h,cpp}` | CPU detection, typed wrappers, kernel cache |
| `jit_ops.{h,cpp}` | High-level kernel builders (dot, gemv, …) |
| `tests.cpp` | All tests (app + JIT) in one TU, run via `--test` |

## Status

Gemma 3n E2B runs end-to-end as an interactive chat REPL. All **19 tests**
pass.

Every weight projection in the model now flows through a JITted, shape-
specialized BF16 × FP32 gemv kernel. On a recent desktop CPU this is
roughly **17 % faster decode** than the equivalent hand-written intrinsic
loop (~267 → ~220 ms/tok in Release on E2B).

This is a personal learning project, not a production runtime.

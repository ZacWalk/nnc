# nnc — Neural Net Compiler

[![Linux](https://github.com/ZacWalk/nnc/actions/workflows/linux.yml/badge.svg)](https://github.com/ZacWalk/nnc/actions/workflows/linux.yml)
[![Windows](https://github.com/ZacWalk/nnc/actions/workflows/windows.yml/badge.svg)](https://github.com/ZacWalk/nnc/actions/workflows/windows.yml)

I have always been fascinated by all the memory shuffling that happens to run a model on a GPU. What happens if you just run it on the CPU optimized for SIMD instructions. This is my learning project that JIT-compiles the hot operations of a transformer into machine code tuned for the host CPU, then uses it to run **Gemma inference end-to-end** on its own minimal tensor runtime.

![nnc prompt demo](ncc-prompt.gif)

No third-party JIT libraries, no ML frameworks — just a C++20 compiler
(MSVC on Windows, g++/clang on Linux) and hand-rolled x86-64 / AVX2 / FMA encoders.

Runs on **Windows x64 and Linux x64** (including WSL) from a single source
tree. All OS-specific calls live behind a small `sys.h` shim; the JIT
kernels themselves are bit-identical on both platforms.

![nnc architecture](nnc-diagram.png)

> **How it works:** see **[docs/design.md](docs/design.md)** — layering, the
> Q8_0 split layout, the JIT and its ABI, threading, the Gemma forward pass,
> profiler output, and where the remaining headroom is.
> Contributing? Read **[AGENTS.md](AGENTS.md)** for the house rules.

## What it does

- Loads a Gemma/llama GGUF by mmapping the file. F32/F16/BF16 weights are
  used in place; Q8_0 / Q4_K / Q5_K / Q6_K are decoded once at load time.
- **Q8_0 weight quantization, on by default.** Weights are decoded or
  converted straight into an nnc-specific split layout — no BF16
  intermediate, and all block metadata is BF16 rather than FP32.
  `--q4` cuts the bytes again (0.625 vs 1.0625 per weight) for another
  ~1.6× on decode, at a real accuracy cost. `--bf16` keeps the source
  values.
- **Batched prefill.** One pass over the weights serves 16 prompt tokens
  instead of one, which takes prefill from bandwidth-bound to
  compute-bound — 9.5× faster on BF16 weights.
- Runs the per-token forward pass (embed → transformer blocks → final
  norm → logits) directly against the SIMD/JIT kernels, with a per-token
  KV cache, shared-KV and sliding-window attention, and the Gemma 3n PLE
  (per-layer-input embedding) gate.
- Emits shape-specialized machine-code gemv kernels into executable pages
  (W^X, `VirtualAlloc`/`mmap` behind the sys layer) and caches them by
  shape, so each is JITted once and reused for the rest of the run.
- Static **worker pool** (default `min(8, CPUs)` threads, override with
  `NNC_THREADS`) parallelises every large gemv along the row axis.
- Runs softmax, layernorm, RMSNorm, RoPE, soft-cap, SwiGLU, GELU×up,
  embedding-row lookup and elementwise ops as in-house AVX2/FMA kernels.
- Interactive chat REPL: wraps each user line in Gemma's
  `<start_of_turn>user … <end_of_turn>\n<start_of_turn>model\n` template,
  greedy-decodes until `<end_of_turn>`, and streams detokenized UTF-8 text
  to the console.
- **`--perf`** prints a per-phase timing breakdown (ms, %, calls, µs/call,
  effective GB/s) after each reply.

The loader accepts `gemma4` (Gemma 3n), `gemma3` and `llama` architectures.

## Requirements

- x86-64 CPU with **AVX2 + FMA** (detected at startup; older CPUs are refused).
  AVX-512 is intentionally not used.
- **CMake 3.21+** and **Ninja**.
- One of:
  - **Windows x64** with MSVC v143+ (Visual Studio 2022 or later).
  - **Linux x64** (or WSL2) with `g++` 10+ or `clang++` 12+.
- C++20 / C17.

## Getting a model

On Windows, `dd.ps1` downloads known-good test models into `.\models`:

```powershell
.\dd.ps1 download                  # list what's available
.\dd.ps1 download gemma-3-1b-bf16  # ~1.9 GB, smallest end-to-end test
.\dd.ps1 download all
```

Any `.gguf` under `./models`, `$HOME/.lmstudio/models` or `$HOME/models` is
offered by the interactive picker on a bare `nnc`. Note that nnc reads
F32/F16/BF16/Q8_0/Q4_K/Q5_K/Q6_K — `--inspect-model <file>` will tell you
whether a given file is loadable.

**[docs/models.md](docs/models.md)** lists the models that have been run
end-to-end with measured speed and resident memory for each, plus the ones
that do not load and why.

## Build

One CMake + Ninja build serves both platforms. Two presets, `release` and
`debug`; both drop their binary in `exe/`.

```bash
cmake --preset release && cmake --build --preset release   # -> exe/nnc[.exe]
cmake --preset debug   && cmake --build --preset debug     # -> exe/nnc-d[.exe]
ctest --preset debug                                       # runs --test
```

Build trees live in `build/<HostSystem>-<preset>/`, so a Windows and a WSL
build of the same working tree do not fight over one cache.

### Windows (MSVC, x64)

Ninja invokes `cl.exe` directly, so run the commands from a **x64 Native
Tools Command Prompt** / VS Developer PowerShell. Or just use the dev
driver, which sets the environment up for you:

```powershell
.\dd.ps1 run       # configure + build Release, then run exe\nnc.exe
.\dd.ps1 test      # configure + build Debug, then run exe\nnc-d.exe --test
```

### Linux / WSL (g++ or clang++, x86-64 with AVX2 + FMA + F16C)

```bash
sudo apt-get install -y cmake ninja-build g++
cmake --preset release && cmake --build --preset release
```

All OS-specific calls live behind `src/sys.h`; `src/sys_win.cpp` is built
on Windows and `src/sys_linux.cpp` is built on Linux. Each TU is
`#if`-guarded so both files can sit in the build on either platform.
The JIT kernels are written assuming the Windows x64 calling convention;
on Linux every kernel prepends a tiny SysV→Win64 register shuffle so the
same encoders are used unchanged.

## Run

Inference (interactive REPL). With no `-m`, nnc scans `./models`,
`$HOME/.lmstudio/models` and `$HOME/models` and asks which model to load:

```powershell
# Windows
.\exe\nnc.exe
.\exe\nnc.exe -m models\gemma-3-1b-it-BF16.gguf
```

```bash
# Linux / WSL
./exe/nnc
```

The app loops on a `>` prompt. Each line is run through the Gemma chat
template and the model reply is streamed back as plain text. `/reset`
clears the conversation, `/exit` quits.

Tests (single binary, no framework):

```
exe/nnc-d --test           # Linux
.\exe\nnc-d.exe --test     # Windows
```

Inspection / smoke-test modes: `--list-models`, `--gguf-info`,
`--gguf-stats`, `--inspect-model`, `--inspect-all`, `--gemma-info`,
`--gemma-probe`, `--gemma-forward`, `--gemma-tokenize`, `--gemma-prompt`,
`--gemma-gen`.

Global flags:

- `--bf16` / `-bf16` — keep weights as raw BF16 (disables quantization).
- `--q8` / `-q8` — Q8_0 weights (this is the default).
- `--q4` / `-q4` — 4-bit weights: ~1.5× faster decode than Q8_0, lower quality.
- `--perf` / `-perf` — print a per-phase timing breakdown after generation.
- `-m <file.gguf>` — model path (skips the interactive picker).
- `-s <seed>`, `-n <n_predict>`, `-b <batch>`, `--top_k`, `--top_p`,
  `--temp` — parsed by `gpt_params_parse`. The sampling flags are
  currently unused; decode is greedy/argmax.

Environment:

- `NNC_THREADS=<N>` — override worker pool size (default = min(8, CPUs)).

## Repository layout

Flat by design — everything lives directly under `src/`. At the top level:
`CMakeLists.txt` + `CMakePresets.json` (the build), `dd.ps1` (Windows dev
driver), `.github/workflows/` (CI), `docs/` (design notes and per-model
measurements).

| File | Purpose |
| --- | --- |
| `main.cpp` | CLI entry, argument dispatch, model picker + REPL driver |
| `runtime.{h,cpp}` | Arena allocator, tensor descriptors, dtype table, profiler |
| `nn_ops.{h,cpp}` | SIMD kernels + JIT-routed gemv + worker pool + quantizers |
| `utils.{h,cpp}` | CLI parsing |
| `gguf.{h,cpp}` | GGUF v2/v3 header + KV + tensor table parser, mmap |
| `gemma.{h,cpp}` | Gemma/llama loader, tokenizer, KV cache, forward pass |
| `jit_buffer.{h,cpp}` | Executable-memory allocator (W^X via the sys layer) |
| `emitter_x64.{h,cpp}` | Raw byte / REX / ModR/M / SIB encoders, plus the SysV→Win64 ABI shim |
| `emitter_avx2.{h,cpp}` | VEX-encoded AVX2 + FMA encoders |
| `jit_kernel.{h,cpp}` | CPU detection, typed wrappers, kernel cache |
| `jit_ops.{h,cpp}` | High-level kernel builders (dot, gemv, …) |
| `sys.h` / `sys_win.cpp` / `sys_linux.cpp` | OS abstraction: console, exec-page allocator, mmap, CPUID |
| `tests.cpp` | All tests (app + JIT) in one TU, run via `--test` |

See [docs/design.md](docs/design.md) for how these fit together.

## Status

Gemma 3 / 3n and llama GGUFs run end-to-end as an interactive chat REPL.
All **30 tests** pass.

Every weight projection flows through a JITted, shape-specialized BF16,
Q8_0 or 4-bit gemv kernel, parallelised across the worker pool when both
dims are large.

Gemma 3 1B, Release, 8 threads, 40-token prompt + decode:

| weight dtype | prefill (ms/tok) | decode (ms/tok) | resident |
| --- | --- | --- | --- |
| BF16 (`--bf16`) | 9.0 | 69.9 | mmap only |
| Q8_0 (default)  | 6.3 | 36.7 | 1013 MB |
| Q4 (`--q4`)     | 7.2 | 22.6 | 596 MB |

Larger models, decode ms/tok: Gemma 3 4B 128 (Q8_0) / 81 (Q4);
Llama 3 8B 234 / 157, the latter at 4.8 GB resident.

Decode tracks bytes-per-weight almost exactly and runs close to this
machine's measured ~28 GB/s DRAM roof. Prefill is compute-bound: it costs
about the same regardless of weight format, which is what batching was for.
`--perf` shows the per-phase breakdown; see
[docs/design.md](docs/design.md#7-profiler) for how to read it,
[the roofline](docs/design.md#the-roofline) for what "bandwidth-bound"
is measured against, and
[the headroom list](docs/design.md#9-where-the-remaining-headroom-is) for
what is left.

Windows vs Linux: steady-state decode is essentially identical (within
~5 %) — the JIT kernels are byte-for-byte the same code on both, and
decode is bandwidth-bound. Cold prefill is noticeably slower under WSL
when the `.gguf` lives on the Windows side of `/mnt/c` because the
initial mmap pages cross the 9P bridge; placing the model on the Linux
filesystem (or running native Linux) closes that gap.

This is a personal learning project, not a production runtime.

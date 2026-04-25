# nnc — Neural Net Compiler

## Project identity

- **Name:** `nnc` (neural net compiler). Use this name in all new docs, comments, log prefixes, CLI help text, and identifiers.
- **Legacy name:** `ml-cpu` — fully retired. The repo folder still happens to be named `ml-cpu` but no project, file, or identifier uses it.
- **Executable:** one binary, `nnc.exe` (Debug: `nnc-d.exe`). No second exe for tests, no second exe for tooling.
- **Platform:** Windows x64 only. MSVC (v145+), C++20, C17.
- **CPU baseline:** AVX2 + FMA + F16C. Do **not** use AVX-512 intrinsics or codegen. Detect AVX2/FMA at startup; refuse to run on older CPUs.

## What nnc is

A learning project that JIT-compiles a neural-net computation graph into hybrid data + machine-code blobs tuned to the host CPU. It loads and runs Gemma 3n GGUF inference end-to-end on its own minimal tensor runtime. Only Gemma 3n GGUF (`general.architecture == "gemma4"`) is supported; legacy GPT-2 `.bin` support has been removed.

## Repository layout

Keep the layout **flat**. Do not create subfolders under `src/` (no `src/jit/`, no `src/ops/`).

```
nnc.sln
src/
  nnc.vcxproj
  main.cpp                   CLI entry, argument dispatch, Gemma REPL driver
  runtime.cpp / runtime.h    arena, tensor, cgraph, forward dispatch
  nn_ops.cpp / nn_ops.h      own SIMD kernels + JIT-routed gemv + graph fusion
  utils.cpp / utils.h        CLI parsing
  gguf.cpp / gguf.h          GGUF v2/v3 parser + mmap loader
  gemma.cpp / gemma.h        Gemma 3n loader, tokenizer, KV cache, forward
  jit_buffer.cpp/.h          executable-memory allocator (VirtualAlloc + VirtualProtect)
  emitter_x64.cpp/.h         raw byte / REX / ModR/M / SIB encoding helpers
  emitter_avx2.cpp/.h        VEX-encoded AVX2 + FMA + F16C instructions we use
  jit_kernel.cpp/.h          typed function-pointer wrappers, CPU detection, kernel cache
  jit_ops.cpp/.h             high-level kernel builders (dot, gemv, ...)
  tests.cpp                  ALL tests live here (app + jit). Single TU.
models/                      Gemma 3n weight files (.gguf)
exe/                         build output (nnc.exe, nnc-d.exe)
```

## CLI conventions

One executable, mode chosen by argv. The default model path is
`models\gemma-4-E2B-it-BF16.gguf`; bare `nnc` (with no `-m`) opens the chat
REPL on that file.

- `nnc` — load the default Gemma model and start the interactive chat REPL.
- `nnc -m <file.gguf>` — same, but with a different Gemma 3n GGUF.
- `nnc --test` (also `-test`, `/test`) — runs every test in `tests.cpp` and exits with non-zero on failure. Tests must be silent on success beyond a final summary line.
- `nnc --gguf-info <file>`, `--gguf-stats`, `--gemma-info`, `--gemma-probe`, `--gemma-forward`, `--gemma-tokenize`, `--gemma-prompt`, `--gemma-gen` — inspection / smoke-test modes.
- Existing flags from `utils.cpp` (`-s`, `--top_k`, `--top_p`, `--temp`, `-b`, `-n`) keep their names. `-n` controls max tokens generated per REPL turn (default 256). The sampling flags (`--top_k`, `--top_p`, `--temp`) are parsed but currently unused — the REPL uses argmax (greedy) decode.

Argument parsing lives in `utils.cpp::gpt_params_parse`. New flags get added there, not in `main.cpp`.

### Chat REPL specifics (`main.cpp`)

- On startup: `SetConsoleOutputCP(CP_UTF8)` + `SetConsoleCP(CP_UTF8)` so that BPE pieces containing U+2581 (▁), em-dashes, etc. render correctly. Always set this when adding new console output paths.
- `<windows.h>` must be included with `WIN32_LEAN_AND_MEAN` **and** `NOMINMAX` so the Windows `min` / `max` macros don't collide with `std::min` / `std::max`.
- Each line typed at the `>` prompt is wrapped in Gemma's chat template:
  `[<bos>] <start_of_turn> user \n <user text> <end_of_turn> \n <start_of_turn> model \n`.
  The `<start_of_turn>` / `<end_of_turn>` ids are looked up from the vocab. In Gemma's GGUF they are stored under the mangled piece strings `"<|turn>"` (id 105) and `"<turn|>"` (id 106); the lookup falls back to those if the literal `"<start_of_turn>"` / `"<end_of_turn>"` strings are not present.
- Decoding stops on `gf.eos_id` (= `<end_of_turn>`, id 106) or after `n_predict` tokens. Output is streamed token-by-token via `gemma_detokenize` so the user sees text appear as it is generated.

## Tests (`tests.cpp`)

- Single translation unit. No framework. Plain functions returning `bool`, collected in a static array, run sequentially.
- Each test prints `[PASS]` / `[FAIL] reason` with a stable name. Final line: `nnc: N passed, M failed`. Exit code = number of failures.
- Cover both app-level helpers and JIT/SIMD primitives (allocator, emitter encodings, dot/gemv numerical equivalence to a scalar reference, gelu/softmax/layernorm/swiglu/rope/softcap, fp16-W and bf16-W gemv, bf16 row embed).
- JIT numerical tests compare against a scalar reference within a relative tolerance (start at `1e-4`); FMA rounding will differ from naive scalar order — that is expected, not a bug.
- Tests must not require model files. Generate synthetic inputs.

Current test count: **19 tests, all passing**.

## Runtime (`runtime.cpp` / `runtime.h`)

- Arena allocator (`nnc_init` / `nnc_free`) — bump-pointer over a single mem buffer; no per-tensor frees.
- Tensor: `type` (F32 / F16 / I32 / BF16), `n_dims`, `ne[4]`, `nb[4]`, `op`, `src0/src1`, `data`, `op_params[4]`. Gemma weights are BF16; activations and norms are F32.
- Op enum and graph (`nnc_cgraph`, `NNC_MAX_NODES = 4096`). Build via `nnc_build_forward_expand` (DFS topo).
- `nnc_graph_compute` first calls `nnc_graph_prefuse` (in `nn_ops.cpp`) to identify `mul_mat → repeat(bias) → add [→ gelu]` patterns and mutate skipped nodes' `op` to `NNC_OP_NONE`. Then dispatches each node.
- VIEW / RESHAPE / PERMUTE / TRANSPOSE are no-ops at compute time: their result tensors share the source's `data` pointer and carry adjusted `ne`/`nb`. Permute semantics: `result->ne[axes[i]] = src->ne[i]` (axes specify destination slot). CPY is a logical row-major flatten into the destination's contiguous buffer.

## JIT design rules

- **ABI:** Windows x64 only. Document register usage at the top of each kernel builder. Prefer volatile registers (RAX, RCX, RDX, R8–R11, XMM0–XMM5). The existing gemv builders save and restore RSI/RDI when they need extra GPRs — follow the same pattern (push rsi/rdi, no extra `sub rsp`).
- **`vzeroupper` before every `ret`.** Non-negotiable. Otherwise the caller's SSE code stalls.
- **No third-party JIT libs.** No AsmJit, no xbyak, no LLVM. Hand-rolled encoders only (`emitter_x64.cpp`, `emitter_avx2.cpp`).
- **Kernel cache:** `jit_kernel_cache` in `jit_kernel.cpp` holds one `std::unordered_map<uint64_t, entry>` per kernel family, keyed on `pack(rows, cols)`. JIT once on first use, reuse forever. Lock with `global_cache_mutex()` (in `nn_ops.cpp`) when constructing.
- **Specialization:** bake shape constants (`rows`, `cols`, `head_dim`) as 32-bit immediates so inner loops have no bounds-check and the optimizer can fully unroll small dimensions. Use 4-accumulator FMA unrolling whenever `cols % 32 == 0` to break the FMA latency chain.
- **BF16 → F32:** in JIT, decode BF16 with `vpmovzxwd` (8 u16 → 8 u32) + `vpslld ymm, ymm, 16` rather than a lookup or the FP16 path. This is the conversion used by `nnc_build_gemv_bf16w_f32x`.
- **Fallback:** anything not on the JIT/SIMD fast path runs through the reference loops in `runtime.cpp` or the scalar tail in the corresponding `nn_ops.cpp` wrapper. 

## Implementation status

Stages 0–5 from the original roadmap are complete plus a BF16-weight JIT path: cleanup done, JIT foundations in place, FP32 dot/gemv kernels, FP16-W/FP32-x gemv with bias-add and GELU fusion, **BF16-W/FP32-x gemv** (`nnc_build_gemv_bf16w_f32x` in `jit_ops.cpp`, encoded with `vpmovzxwd` + `vpslld 16` + 4-accumulator FMA unroll when `cols % 32 == 0`), plus SIMD softmax, layernorm/RMSNorm, RoPE, soft-cap, SwiGLU. Every Q/K/V/O, gate/up/down, PLE inp_gate/proj, and the final logits gemv in Gemma now JITs through the BF16 cache. The historical `ggml.*` and intermediate `nnc.cpp/.h` ports, the legacy GPT-2 `.bin` loader, and the JSON/BPE tokenizer are all gone. The runtime is Gemma 3n GGUF only.

Released perf reference (E2B, recent desktop CPU): decode ≈ 220 ms/tok with the JIT, vs ≈ 267 ms/tok using the equivalent intrinsic loop — roughly a 1.2× speedup. Memory bandwidth on the BF16 weights dominates the remaining cost.

Next areas of work: per-head fixed-shape attention kernels, more aggressive fusion (e.g. fused gate*up*silu + down for the FFN), and further Release-build perf tuning.

## Build & run (reference)

MSBuild via `vswhere`:

```powershell
$msbuild = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" `
    -latest -requires Microsoft.Component.MSBuild -find MSBuild\**\Bin\MSBuild.exe
& $msbuild nnc.sln /p:Configuration=Debug /p:Platform=x64 /m /v:minimal /nologo
```

Smoke run (uses the default model path automatically):

```powershell
.\exe\nnc-d.exe
```

Tests:

```powershell
.\exe\nnc-d.exe --test
```

## House style

- C++20, exceptions allowed but not used in hot paths.
- **Asserts:** use `NNC_ASSERT(expr)` from `runtime.h`. It prints `nnc: ASSERT failed: <expr> at <file>:<line>` to stderr and `_exit`s — no modal message box. Plain `assert(...)` is acceptable inside the emitters for build-time invariants but `NNC_ASSERT` is preferred in any new runtime / op code.
- Tabs for indentation in existing files (matches current code); do not reformat unrelated lines.
- New files: same brace style as `main.cpp` (Allman, tabs).
- No new dependencies without an explicit ask. The project must build with only MSVC + the Windows SDK.
- Do not add docstrings, comments, or refactors to code you are not otherwise changing.
- When concatenating C/C++ files on Windows, strip any 0x1A (Ctrl+Z) bytes — MSVC treats them as EOF in text mode.

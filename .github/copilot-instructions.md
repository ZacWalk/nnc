# nnc — Neural Net Compiler

## Project identity

- **Name:** `nnc` (neural net compiler). Use this name in all new docs, comments, log prefixes, CLI help text, and identifiers.
- **Legacy name:** `ml-cpu` — fully retired. The repo folder still happens to be named `ml-cpu` but no project, file, or identifier uses it.
- **Executable:** one binary, `nnc.exe` (Debug: `nnc-d.exe`). No second exe for tests, no second exe for tooling.
- **Platform:** Windows x64 only. MSVC (v145+), C++20, C17.
- **CPU baseline:** AVX2 + FMA + F16C. Do **not** use AVX-512 intrinsics or codegen. Detect AVX2/FMA at startup; refuse to run on older CPUs.

## What nnc is

A learning project that JIT-compiles a neural-net computation graph into hybrid data + machine-code blobs tuned to the host CPU. It loads and runs GPT-2 inference end-to-end on its own minimal tensor runtime

## Repository layout

Keep the layout **flat**. Do not create subfolders under `src/` (no `src/jit/`, no `src/ops/`).

```
nnc.sln
src/
  nnc.vcxproj
  main.cpp                   CLI entry, argument dispatch, GPT-2 driver
  runtime.cpp / runtime.h    arena, tensor, cgraph, forward dispatch
  nn_ops.cpp / nn_ops.h      own SIMD kernels + JIT-routed gemv + graph fusion
  utils.cpp / utils.h        CLI parsing, vocab, tokenizer, sampler
  jit_buffer.cpp/.h          executable-memory allocator (VirtualAlloc + VirtualProtect)
  emitter_x64.cpp/.h         raw byte / REX / ModR/M / SIB encoding helpers
  emitter_avx2.cpp/.h        VEX-encoded AVX2 + FMA + F16C instructions we use
  jit_kernel.cpp/.h          typed function-pointer wrappers, CPU detection, kernel cache
  jit_ops.cpp/.h             high-level kernel builders (dot, gemv, ...)
  tests.cpp                  ALL tests live here (app + jit). Single TU.
models/                      gpt-2 weight files (.bin)
exe/                         build output (nnc.exe, nnc-d.exe)
```

## CLI conventions

One executable, mode chosen by argv:

- `nnc -p "..." -m models\ggml-model-gpt-2-117M.bin` — normal inference. (The model **file** still has the historical `ggml-` prefix; that is the on-disk format name, not our code.)
- `nnc --test` (also accepts `-test`, `/test`) — runs every test in `tests.cpp` and exits with non-zero on failure. Tests must be silent on success beyond a final summary line.
- Existing flags from `utils.cpp` (`-s`, `--top_k`, `--top_p`, `--temp`, `-b`, `-n`) keep their names.

Argument parsing lives in `utils.cpp::gpt_params_parse`. New flags get added there, not in `main.cpp`.

## Tests (`tests.cpp`)

- Single translation unit. No framework. Plain functions returning `bool`, collected in a static array, run sequentially.
- Each test prints `[PASS]` / `[FAIL] reason` with a stable name. Final line: `nnc: N passed, M failed`. Exit code = number of failures.
- Cover both app-level helpers and JIT/SIMD primitives (allocator, emitter encodings, dot/gemv numerical equivalence to a scalar reference, gelu/softmax/layernorm/gemv-f16-w numerical checks).
- JIT numerical tests compare against a scalar reference within a relative tolerance (start at `1e-4`); FMA rounding will differ from naive scalar order — that is expected, not a bug.
- Tests must not require model files. Generate synthetic inputs.

Current test count: **11 tests, all passing** (`cpu_has_avx2_fma`, `jit_return_42`, `jit_add_two_ints`, `jit_dot_f32`, `jit_gemv_f32`, `jit_gemv_cache_reuse`, `nnc_gelu_f32`, `nnc_dot_f16_to_f32`, `nnc_softmax_f32`, `nnc_layernorm_f32`, `nnc_gemv_f16w_f32x`).

## Runtime (`runtime.cpp` / `runtime.h`)

- Arena allocator (`nnc_init` / `nnc_free`) — bump-pointer over a single mem buffer; no per-tensor frees.
- Tensor: `type` (F32 / F16 / Q4_0 / Q4_1 / I32), `n_dims`, `ne[4]`, `nb[4]`, `op`, `src0/src1`, `data`, `op_params[4]`. Q4_0/Q4_1 entries exist only so the model loader's storage math is correct; no compute path uses them (GPT-2 117M is F16).
- Op enum and graph (`nnc_cgraph`, `NNC_MAX_NODES = 4096`). Build via `nnc_build_forward_expand` (DFS topo).
- `nnc_graph_compute` first calls `nnc_graph_prefuse` (in `nn_ops.cpp`) to identify `mul_mat → repeat(bias) → add [→ gelu]` patterns and mutate skipped nodes' `op` to `NNC_OP_NONE`. Then dispatches each node.
- VIEW / RESHAPE / PERMUTE / TRANSPOSE are no-ops at compute time: their result tensors share the source's `data` pointer and carry adjusted `ne`/`nb`. Permute semantics: `result->ne[axes[i]] = src->ne[i]` (axes specify destination slot). CPY is a logical row-major flatten into the destination's contiguous buffer.

## JIT design rules

- **ABI:** Windows x64 only. Document register usage at the top of each kernel builder. Use only volatile registers (RAX, RCX, RDX, R8–R11, XMM0–XMM5, YMM upper halves) so kernels need no callee-save prologue.
- **Stack:** every kernel does `sub rsp, 40` on entry, `add rsp, 40` before `ret` (32 bytes shadow + 8 bytes to re-align after `call`).
- **`vzeroupper` before every `ret`.** Non-negotiable. Otherwise the caller's SSE code stalls.
- **No third-party JIT libs.** No AsmJit, no xbyak, no LLVM. Hand-rolled encoders only.
- **Kernel cache:** `std::unordered_map<KernelKey, jit_kernel*>` keyed on `(op, dtype, shape...)`. JIT once on first use, reuse forever.
- **Specialization:** bake shape constants (`rows`, `cols`, `head_dim`) as immediates so inner loops have no bounds-check and the optimizer can fully unroll small dimensions.
- **Fallback:** anything not on the JIT/SIMD fast path runs through the reference loops in `runtime.cpp`. 

## Implementation status

Stages 0–5 from the original roadmap are complete: cleanup done, JIT foundations in place, FP32 dot/gemv kernels, FP16-W/FP32-x gemv with bias-add and GELU fusion, plus SIMD softmax and layernorm. The historical `ggml.*` and intermediate `nnc.cpp/.h` ports are gone; the runtime is a fresh ~700-line implementation.

Current best on GPT-2 117M (Debug build, `-n 20 --seed 1`): ≈11–12 ms/token, bit-identical output to the prior baseline (`"Hello, my name is Braid. As a young boy I did all the hard work of building up my skills at my"`).

Next areas of work: per-head fixed-shape attention kernels, more aggressive fusion, and Release-build performance tuning.

## Build & run (reference)

MSBuild via `vswhere`:

```powershell
$msbuild = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" `
    -latest -requires Microsoft.Component.MSBuild -find MSBuild\**\Bin\MSBuild.exe
& $msbuild nnc.sln /p:Configuration=Debug /p:Platform=x64 /m /v:minimal /nologo
```

Smoke run:

```powershell
.\exe\nnc-d.exe -m models\ggml-model-gpt-2-117M.bin -p "Hello, my name is" -n 20 --seed 1
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

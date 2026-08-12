# nnc — Neural Net Compiler

**Read [docs/design.md](docs/design.md) first.** It is the single source of
truth for how nnc works: layering, dtypes and the Q8_0 split layout, the JIT
and its ABI, threading, the Gemma forward pass, the profiler, and the current
performance picture. This file covers only the rules for *changing* the code.
When you change behaviour, update `docs/design.md`; `README.md` and this file
link to it rather than restating it.

## Project identity

- **Name:** `nnc` (neural net compiler). Use this name in all new docs, comments, log prefixes, CLI help text, and identifiers.
- **Executable:** one binary, `nnc.exe` (Debug: `nnc-d.exe`) on Windows; `nnc` / `nnc-d` on Linux. No second exe for tests, no second exe for tooling.
- **Platforms:** Windows x64 (MSVC v145+) and Linux x64 (g++ 10+ / clang++ 12+, including WSL2). C++20, C17.
- **CPU baseline:** AVX2 + FMA. Do **not** use AVX-512 intrinsics or codegen. Detect AVX2/FMA at startup; refuse to run on older CPUs.
- **No new dependencies** without an explicit ask. Must build with only MSVC + the Windows SDK, or g++/clang + libstdc++.

## Repository layout

Keep the layout **flat**. Do not create subfolders under `src/` (no `src/jit/`, no `src/ops/`).

```
nnc.sln
Makefile                     Linux build (release/debug/test/clean)
dd.ps1                       Windows dev driver: run / test / download
docs/design.md               how it works — keep this current
docs/models.md               per-model measured speed / footprint / what won't load
src/
  nnc.vcxproj
  main.cpp                   CLI entry, argument dispatch, model picker, REPL
  runtime.cpp / runtime.h    arena allocator, dtype table, tensor descriptors, timers, profiler
  nn_ops.cpp / nn_ops.h      own SIMD kernels + JIT-routed gemv + worker pool + (de)quantizers
  utils.cpp / utils.h        CLI parsing
  gguf.cpp / gguf.h          GGUF v2/v3 parser + mmap loader
  gemma.cpp / gemma.h        Gemma/llama loader, tokenizer, KV cache, forward pass
  jit_buffer.cpp/.h          executable-memory allocator (W^X via the sys layer)
  emitter_x64.cpp/.h         raw byte / REX / ModR/M / SIB encoding helpers + SysV→Win64 ABI shim
  emitter_avx2.cpp/.h        VEX-encoded AVX2 + FMA instructions we use
  jit_kernel.cpp/.h          typed function-pointer wrappers, CPU detection, kernel cache
  jit_ops.cpp/.h             high-level kernel builders (dot, gemv, ...)
  sys.h                      OS abstraction (console, exec pages, mmap, CPUID)
  sys_win.cpp                Windows impl of sys.h (#if defined(_WIN32))
  sys_linux.cpp              Linux/POSIX impl of sys.h (#if !defined(_WIN32))
  tests.cpp                  ALL tests live here (app + jit). Single TU.
models/                      weight files (.gguf), fetched by `dd.ps1 download`
exe/                         build output (nnc(.exe), nnc-d(.exe))
```

**OS isolation rule:** no `windows.h`, `<intrin.h>`, `<sys/mman.h>`,
`<unistd.h>` etc. outside `sys_win.cpp` / `sys_linux.cpp`. New OS
functionality goes through a new `sys_*` function declared in `sys.h`
and implemented in both backends.

**No computation graph.** `gemma.cpp` calls kernels directly. A graph /
op-dispatch layer existed once, went unused, and was deleted. Do not
reintroduce one without a concrete need.

## CLI conventions

One executable, mode chosen by argv. Bare `nnc` (with no `-m`) scans `./models`,
`$HOME/.lmstudio/models` and `$HOME/models` for `.gguf` files and prompts for a
letter to pick one, then opens the chat REPL on it. **Q8_0 weight quantization
is on by default** — pass `--bf16` to keep weights as raw BF16. See
[docs/design.md §3](docs/design.md#3-data-types) for what that actually does.

- `nnc` — pick a model interactively and start the chat REPL.
- `nnc -m <file.gguf>` — skip the picker and load that file.
- `nnc --list-models` — print the same candidate list without loading anything.
- `nnc --bf16` (also `-bf16`) — keep weights as raw BF16.
- `nnc --q8` (also `-q8`) — Q8_0 weights; the default.
- `nnc --q4` (also `-q4`) — 4-bit weights.
- `nnc --perf` (also `-perf`) — enable the phase profiler and print a breakdown after generation.
- `nnc --test` (also `-test`, `/test`) — runs every test in `tests.cpp` and exits with non-zero on failure. Tests must be silent on success beyond a final summary line.
- `nnc --gguf-info <file>`, `--gguf-stats`, `--inspect-model`, `--inspect-all`, `--gemma-info`, `--gemma-probe`, `--gemma-forward`, `--gemma-tokenize`, `--gemma-prompt`, `--gemma-gen` — inspection / smoke-test modes.
- Existing flags from `utils.cpp` (`-s`, `--top_k`, `--top_p`, `--temp`, `-b`, `-n`) keep their names. `-n` controls max tokens generated per REPL turn (default 256). The sampling flags (`--top_k`, `--top_p`, `--temp`) are parsed but currently unused — the REPL uses argmax (greedy) decode.

**Any new global flag must be added in two places:** the pre-scan in `main.cpp`
*and* the `gpt_params_parse` switch in `utils.cpp` — that parser rejects unknown
arguments, so a flag handled only by the pre-scan will be refused on the REPL path.

Quantization is not a separate step: `gemma_load(path, out, quantize_q8)` owns it
end to end, so every call site passes the flag and nothing else has to remember.

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
- Cover both app-level helpers and JIT/SIMD primitives (allocator, emitter encodings, dot/gemv numerical equivalence to a scalar reference, gelu/softmax/layernorm/rmsnorm/swiglu/rope/softcap, bf16-W and Q8_0-W gemv, Q8_0/K-quant dequant, both embedding-row lookups).
- JIT numerical tests compare against a scalar reference within a relative tolerance (start at `1e-4`); FMA rounding will differ from naive scalar order — that is expected, not a bug.
- Tests must not require model files. Generate synthetic inputs.

Current test count: **30 tests, all passing**.

## Design invariants worth restating

These are the ones most easily broken by a well-meaning change. The reasoning
behind each lives in [docs/design.md](docs/design.md).

- **No computation graph.** Call kernels directly from `gemma.cpp`.
- **`vzeroupper` before every `ret`** in emitted code. Non-negotiable.
- **Kernels assume the Win64 ABI**; every builder starts with
  `emit_win64_arg_shuffle(n_int_args)` so Linux works with the same encoders.
- **Decode is DRAM-bandwidth bound, against a *measured* roof.** This
  machine tops out near 28 GB/s and reaches 25.3 GB/s on a **single**
  thread, so threads are there to cover dequantisation, not to accumulate
  bandwidth. A kernel that needs many threads to approach the roof has a
  kernel problem. Check `--perf` before and after, and compare GB/s per
  phase.
- **Benchmark by interleaving the two binaries run-by-run, with an
  untouched control** (`--bf16` shares no kernel with Q8_0/Q4). Session-to-
  session variance is ~20%, which is bigger than most wins worth having;
  measuring all of A then all of B will invent results.
- **Every gemv kernel uses 4 independent FMA accumulators.** One
  accumulator chains four 4-cycle FMAs per block and caps a thread at
  ~5 GB/s. To afford the registers, x is a memory operand on the FMA and
  Q4 isolates low nibbles with a `vpslld 28`/`vpsrld 28` pair rather than
  holding a `0x0F` mask.
- **Block metadata (Q8_0 / Q4 scales and biases) is BF16, never FP16.** A
  block scale can be arbitrarily small; FP16 would flush it to subnormal.
  BF16 also decodes with `vpbroadcastw` + `vpslld 16`, needing no F16C.
- **Query heads sharing a KV head are processed together** so the K/V cache
  is walked once per KV head, not once per query head.
- **Prefill is compute-bound** since batching landed — the opposite regime.
  Do not assume a change helps both.
- **`layer_forward` is one code path for decode and prefill** (`n_tok` 1 or
  16). Do not fork it; a batched copy will drift.
- **The per-token path allocates nothing** — all scratch lives in
  `gemma_kv_cache`, sized once in `gemma_kv_init`.
- **`nnc_gemv_pool` thread count is capped at `MAX_THREADS`** because dispatch
  and the argmax reduction index fixed-size stack arrays by thread id.
- **Packed weights are decoded exactly once, at load**, straight into the
  target layout — never via an intermediate whole-tensor BF16 or F32 image.

## Runtime (`runtime.cpp` / `runtime.h`)

Arena allocator, dtype table, tensor descriptors, timers, and the `--perf`
phase profiler. Nothing else. The Q8_0 split layout, the load-time decode
path and the `token_embd` handling are described in
[docs/design.md §3](docs/design.md#3-data-types).

The only rule that bites here: **`nnc_new_tensor_1d` is the sole descriptor
allocator**, and `gemma.cpp::tensor_from_gguf` overwrites `ne`/`nb`/`data`
after calling it. Keep the arena descriptor-only — weight bytes live in the
mmap or in a `gemma_file`-owned buffer.

## JIT design rules

Full detail (kernel list, encodings, W^X pool, cache) is in
[docs/design.md §4](docs/design.md#4-the-jit). The rules for changing it:

- **ABI:** kernels assume Windows x64 (int args in RCX, RDX, R8, R9). Every builder calls `x64_emitter::emit_win64_arg_shuffle(n_int_args)` as its first instruction — a no-op on Windows, the SysV→Win64 moves on Linux. Document register usage at the top of each builder. Prefer volatile registers (RAX, RCX, RDX, R8–R11, XMM0–XMM5); save/restore RSI/RDI with push/pop if you need more (no extra `sub rsp` — kernels make no calls).
- **`vzeroupper` before every `ret`.** Non-negotiable.
- **No third-party JIT libs.** Hand-rolled encoders only.
- **Specialization:** bake shape constants as 32-bit immediates. Use 4-accumulator FMA unrolling whenever `cols % 32 == 0` to break the FMA latency chain.
- **Fallback:** anything off the JIT fast path runs through the scalar tail in the corresponding `nn_ops.cpp` wrapper. Keep that tail correct — it is what the tests compare against.
- Add a numerical test against a scalar reference for every new kernel.

## Before claiming a performance win

Decode is DRAM-bandwidth bound (see
[docs/design.md §7](docs/design.md#7-profiler)). Run `--perf` before and
after and compare the effective GB/s per phase, not just wall time. A change
that removes instructions without removing bytes read will measure as noise.

## Build & run (reference)

`dd.ps1` is the Windows dev driver:

```powershell
.\dd.ps1 run  [args...]     # build Release, run nnc.exe
.\dd.ps1 test               # build Debug, run nnc-d.exe --test
.\dd.ps1 download           # list downloadable test models
.\dd.ps1 download all       # fetch them into .\models
```

### Windows (MSBuild via `vswhere`)

```powershell
$msbuild = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" `
    -latest -requires Microsoft.Component.MSBuild -find MSBuild\**\Bin\MSBuild.exe
& $msbuild nnc.sln /p:Configuration=Debug /p:Platform=x64 /m /v:minimal /nologo
```

Smoke run / tests:

```powershell
.\exe\nnc-d.exe
.\exe\nnc-d.exe --test
```

### Linux / WSL

```bash
make debug          # -> exe/nnc-d
make                # release -> exe/nnc
make test           # build debug + run --test
./exe/nnc-d --test
```

## House style

- C++20, exceptions allowed but not used in hot paths.
- **Asserts:** use `NNC_ASSERT(expr)` from `runtime.h`. It prints `nnc: ASSERT failed: <expr> at <file>:<line>` to stderr and exits via `std::_Exit` — no modal message box. Plain `assert(...)` is acceptable inside the emitters for build-time invariants but `NNC_ASSERT` is preferred in any new runtime / op code.
- Tabs for indentation in existing files (matches current code); do not reformat unrelated lines.
- New files: same brace style as `main.cpp` (Allman, tabs).
- No new dependencies without an explicit ask. The project must build with only MSVC + the Windows SDK on Windows, and only g++/clang + libstdc++ on Linux.
- Do not add docstrings, comments, or refactors to code you are not otherwise changing.
- When concatenating C/C++ files on Windows, strip any 0x1A (Ctrl+Z) bytes — MSVC treats them as EOF in text mode.
- Standard-library portability: explicitly include `<cstddef>` for `size_t`, `<cstring>` for `memcpy`/`memset`, `<cmath>` for math — don't rely on transitive includes that happen to work on MSVC.

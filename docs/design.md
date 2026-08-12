# nnc — design

This is the single source of truth for how nnc is put together. `README.md`
is the front page (what it is, how to build, how to run) and `AGENTS.md`
holds the house rules for making changes; both defer here for design detail.

---

## 1. What nnc is

nnc JIT-compiles the hot operations of a transformer into machine-code blobs
specialized to the host CPU and the model's exact shapes, then runs Gemma
GGUF inference end-to-end on its own minimal tensor runtime.

Deliberate constraints, because this is a learning project:

- **No third-party JIT library.** No AsmJit, no xbyak, no LLVM. Every byte of
  emitted code comes from the hand-rolled encoders in `emitter_x64.cpp` and
  `emitter_avx2.cpp`.
- **No ML framework, no BLAS.** All kernels are ours.
- **AVX2 + FMA only.** No AVX-512, in codegen or intrinsics. Detected at
  startup; the process refuses to run on older CPUs.
- **One binary, flat source tree.** `nnc(.exe)` release, `nnc-d(.exe)` debug.
  No subdirectories under `src/`.
- **Windows x64 and Linux x64 from one tree.** All OS calls sit behind
  `sys.h`, implemented twice (`sys_win.cpp`, `sys_linux.cpp`).

## 2. Layer cake

```
main.cpp        CLI dispatch, model picker, chat REPL
   |
gemma.cpp       hparams, tokenizer, KV cache, the per-token forward pass
   |
nn_ops.cpp      SIMD kernels + worker pool + JIT dispatch
   |            (gelu, softmax, rmsnorm, rope, softcap, quantize, dequantize)
jit_ops.cpp     kernel builders — emit a whole specialized kernel
   |
emitter_avx2.cpp / emitter_x64.cpp   instruction encoders
   |
jit_buffer.cpp  executable-memory pool (W^X)
   |
sys_win.cpp / sys_linux.cpp          OS primitives
```

`runtime.cpp` sits off to the side: arena allocator, dtype table, tensor
descriptors, timers, and the phase profiler. `gguf.cpp` is a standalone
GGUF v2/v3 parser + mmap loader.

**There is no computation graph.** `gemma.cpp` calls kernels directly, in
order, against scratch buffers pre-allocated in `gemma_kv_cache`. An earlier
version had a `nnc_cgraph` / op-dispatch layer; it was never used by the
model path and has been removed. Do not reintroduce one without a concrete
need.

## 3. Data types

| enum | on disk | in memory | used for |
| --- | --- | --- | --- |
| `NNC_TYPE_F32` | F32 | mmap, in place | norms, biases, RoPE tables |
| `NNC_TYPE_F16` | F16 | mmap, in place | rare |
| `NNC_TYPE_BF16` | BF16 | mmap, in place | weights when `--bf16` |
| `NNC_TYPE_Q8_0` | — | loader-owned buffer | weights by default |
| `NNC_TYPE_Q4_S` | — | loader-owned buffer | weights when `--q4` |

Activations, the residual stream and the KV cache are always F32.

Bytes per weight, which is the only number that matters for decode:

| format | layout per 32 weights | bytes/weight |
| --- | --- | --- |
| BF16 | 64 B | 2.0 |
| Q8_0 | 32 B quants + 2 B scale | 1.0625 |
| Q4_S | 16 B nibbles + 2 B scale + 2 B bias | 0.625 |

**All block metadata is BF16, not FP32 or FP16.** FP32 was wasteful (it cost
11% of Q8_0's bytes and 33% of Q4's). FP16 would be the obvious replacement,
but a block scale can be arbitrarily small — K-quant super-block scales
routinely land in FP16's subnormal range, which is exactly what the
`fp16_subnormal_dequant` regression test exists to catch. BF16 keeps FP32's
exponent range so it cannot flush, and it decodes with two instructions
(`vpbroadcastw` + `vpslld 16`) rather than needing F16C.

BF16 carries 8 significand bits, so a scale is exact to ~0.4%. Against Q4's
3.3% quantization step that is invisible, and even for Q8_0 the per-block
scale error is random across blocks and averages out over a dot product
— verified by Q8_0 producing a token-identical continuation before and
after the change.

### Q8_0 split layout (nnc-specific)

Standard ggml Q8_0 interleaves a fp16 scale with each 32-byte quant block.
nnc instead keeps one allocation per tensor holding

```
int8  qs[rows * cols]                 <- tensor->data points here
bf16  scales[rows * cols / 32]        <- at (uint8_t*)data + rows*cols
```

Row-major, one BF16 scale per 32-element block. The split keeps the JIT
inner loop free of per-block unpacking: it can broadcast the scale and
stream `qs` contiguously.

Quantization is absmax per block: `scale = max(|w|)/127`,
`q = round(w/scale)` clamped to `[-127, 127]`. The scale is rounded to
BF16 *before* the values are quantized against it, so the encoder targets
exactly the number the kernel will multiply by.

### Q4_S split layout

Same idea one step further, and asymmetric because 4 bits cannot afford to
waste a sign:

```
uint8 qs[rows * cols / 2]             <- tensor->data; nibbles
bf16  scales[rows * cols / 32]        <- at data + rows*cols/2
bf16  biases[rows * cols / 32]        <- after scales
```

Reconstruction is `w = scale * q + bias` with `q` an unsigned nibble in
`[0, 15]`; the quantizer sets `scale = (max-min)/15`, `bias = min` per
32-element block. An absmax/symmetric scheme would throw away a bit of
range for no saving.

Nibble order inside a block: byte `i` holds element `i` in its low nibble
and element `i+16` in its high nibble. That is what lets one
`vpmovzxbd` + (`vpand` | `vpsrld 4`) pair yield two independent 8-element
groups with no shuffles.

The bias term is deliberately **not** in the JIT kernel. Expanding the dot
product,

```
sum_k w_k x_k = sum_b [ scale_b * sum_{k in b} q_k x_k  +  bias_b * S_b ]
                                                            ^^^^^^^^^^^
         where S_b = sum_{k in b} x[k] depends only on x, not on the row
```

so the whole correction is one f32 dot of the row's biases against a
`cols/32`-element prefix computed once per gemv (`nnc_block_sums_f32`).
That keeps the kernel to four arguments and off the awkward path of
horizontally reducing per block.

Quality: 4-bit output stays fluent and on-topic but does diverge from the
Q8_0 token stream within a few tokens. It is a real accuracy trade, not a
free win — which is why Q8_0 remains the default.

### Load-time decode

Packed on-disk types are decoded exactly once, at load, into either the Q8_0
split layout (default) or BF16 (`--bf16`):

| ggml type | block | path |
| --- | --- | --- |
| Q8_0 (8) | 32 elems / 34 B | → F32 → nnc Q8_0, Q4_S or BF16 |
| Q4_K (12) | 256 elems / 144 B | → F32 → nnc Q8_0, Q4_S or BF16 |
| Q5_K (13) | 256 elems / 176 B | → F32 → nnc Q8_0, Q4_S or BF16 |
| Q6_K (14) | 256 elems / 210 B | → F32 → nnc Q8_0, Q4_S or BF16 |

The decode runs in 16384-element chunks (a multiple of both 32 and 256, so a
source block never straddles a chunk). This matters: a whole-tensor F32
staging buffer for `token_embd` on a 4B model is ~2.7 GB.

Crucially the intermediate BF16 image is **never materialised** when
quantizing — the chunk goes F32 → Q8_0 directly. Loading a 2.3 GB Q4_K_M
model peaks at ~6.6 GB rather than ~13 GB.

BF16 tensors that were used in place from the mmap are converted by a
post-pass (`gemma_quantize_q8_0`) after all tensors are constructed.

### `token_embd` is quantized too

`token_embd` doubles as the input embedding table and (unless the model ships
a separate `output`) the lm_head. Both readers are dtype-aware —
`gw_embed_row` for the row lookup, `gw_gemv` for the projection — so it is
quantized like any other weight. This matters a lot: the lm_head is ~30% of
decode time on Gemma (vocab = 262144), and leaving it BF16 forfeited most of
the Q8_0 win.

## 4. The JIT

### Why JIT at all

Every gemv in a transformer has fixed `rows` and `cols` for the life of the
process. Baking them as 32-bit immediates removes the loop bounds check,
lets the inner loop be unrolled to exactly the right shape, and lets the
row-stride add be a constant. There is no runtime shape dispatch in the
emitted code — the kernel *is* the shape.

### ABI

Kernels are written assuming the **Windows x64** convention: integer args in
RCX, RDX, R8, R9. On Linux every builder calls
`x64_emitter::emit_win64_arg_shuffle(n_int_args)` first, which emits the
SysV → Win64 register moves (and is a no-op on Windows). One set of encoders
serves both platforms.

Register discipline:

- Prefer volatile registers: RAX, RCX, RDX, R8–R11, XMM0–XMM5.
- Builders that need extra GPRs `push rsi` / `push rdi` in the prologue and
  pop in the epilogue. Two pushes = 16 bytes, which realigns the stack to 16
  (entry is 8 mod 16 after the `call`), and no kernel makes calls, so no
  extra `sub rsp` is needed.
- **`vzeroupper` before every `ret`.** Non-negotiable — otherwise the
  caller's SSE code eats a transition penalty.

Document register usage in a comment block at the top of each builder.

### Kernels

| builder | shape | notes |
| --- | --- | --- |
| `nnc_build_dot_f32` | runtime `n` | reference/test kernel |
| `nnc_build_gemv_f32` | rows, cols baked | reference/test kernel |
| `nnc_build_gemv_bf16w_f32x` | rows, cols baked | 4 accumulators when `cols % 32 == 0` |
| `nnc_build_gemv_bf16w_f32x_4row` | rows (×4), cols baked | shared x-tile across 4 rows |
| `nnc_build_gemv_q8_0_f32x_1row` | cols baked | caller walks rows; 4 accumulators |
| `nnc_build_gemv_q4_s_f32x_1row` | cols baked | caller walks rows; 4 accumulators |

BF16 → F32 in JIT is a free shift, not a lookup:

```
vpmovzxwd ymm, [m128]    ; 8 u16 -> 8 u32, zero-extended
vpslld    ymm, ymm, 16   ; bf16 bits become the high half of the f32
```

Q8_0 → F32 is `vpmovsxbd` (8 i8 → 8 i32) + `vcvtdq2ps` + `vmulps` by the
broadcast block scale, then `vfmadd231ps` against x.

Q4_S → F32 widens one 8-byte load into two 8-element groups:

```
vpmovzxbd y5, [rcx + rax]      ; bytes 0..7 -> 8 u32
vpslld    y7, y5, 28
vpsrld    y7, y7, 28           ; low nibbles  = elements 0..7
vpsrld    y5, y5, 4            ; high nibbles = elements 16..23
```

One counter drives both streams in the Q4 kernel: `rax` walks the packed
byte offset, and since 16 quant bytes cover 32 floats, the x address is
exactly `[rdx + rax*8]`.

### Four accumulators, and why the register budget is tight

Each block issues four FMAs. Pointed at a single accumulator they form a
dependency chain, and FMA latency on Zen 3 is 4 cycles, so a block costs
~16 cycles no matter how much memory bandwidth is spare. For Q4 that is
20 bytes per 16 cycles ≈ **5 GB/s per thread** — and the measured figure
was 4.0, against 25.3 for a plain load loop ([§8](#the-roofline)). Both
quantized kernels originally did this; the BF16 ones did not, which is why
BF16 sat at the roof and Q4 did not.

Splitting across four independent accumulators costs registers, and the
emitter only exposes `ymm0`–`ymm7`. Two moves buy the room:

- **x becomes a memory operand on the FMA** (`vfmadd231ps ymm, ymm, m256`)
  instead of a separate `vmovups` into a register — one fewer instruction
  and one fewer live register per group.
- **Q4 drops the `0x0F` mask register**, isolating low nibbles with a
  `vpslld 28` / `vpsrld 28` pair instead. That trades one held register for
  one extra instruction, which is the right way round when the constraint
  is registers.

Q4 then lands exactly on eight: `y0`–`y3` accumulators, `y4` scale,
`y5`/`y6` the two loaded byte groups, `y7` scratch. Reusing `y7` (and later
`y5`/`y6`, once each is dead) across groups is free — register renaming
breaks the write-after-write, so the four FMA chains stay independent.

Effect, interleaved A/B against the same binary otherwise:

| | prefill | decode @2 threads | decode @8 threads |
| --- | --- | --- | --- |
| Q8_0 | +40% | +28% | +3.4% |
| Q4 | +31% | +40% | 0% |

The shape of that table is the point. Prefill is compute-bound so it takes
the win in full; decode at 8 threads on a 16-core machine was already
sitting on the memory roof, so it cannot. What the fix really buys decode
is reaching that roof with about half the threads.

The 4-row BF16 variant loads the `x[k..k+7]` tile once and FMAs it into four
independent row accumulators, cutting x-side bandwidth 4×. Its tail reduces
four ymm partial sums to four contiguous floats with a 3-`vhaddps` tree plus
one `vextractf128` + `vaddps`, and stores them with a single 16-byte write.

### Kernel cache

`jit_kernel_cache` holds one `std::unordered_map` per kernel family, keyed on
`pack(rows, cols)` (or just `cols` for the Q8_0 1-row kernel). Lookups take a
`std::shared_lock`; only first-time codegen takes the exclusive lock, and the
map is re-checked under it so a race builds the kernel once. Entries are
never evicted — a model has a few dozen distinct shapes.

### Executable memory (W^X)

`jit_buffer::commit()` copies staged bytes into the next 16-byte-aligned slot
of a shared 64 KB page pool. Pages live as `PAGE_EXECUTE_READ` and flip to
`PAGE_READWRITE` only for the memcpy, then back. Modern Intel parts treat
permanently-RWX pages as potentially self-modifying, which hurts the uop
cache for hot kernels. The two extra `VirtualProtect` calls are paid once per
kernel at startup and never during generation.

Packing many small kernels into one page also drops per-kernel overhead from
~4 KB (a page each, 95% wasted) to ~16 bytes of alignment padding, and keeps
related kernels physically close, which helps the iTLB.

## 5. Threading

A single static pool (`nnc_gemv_pool`) of `min(8, CPUs) - 1` workers plus the
calling thread. Override the total with `NNC_THREADS`; it is hard-capped at
`MAX_THREADS` (64) because dispatch and the argmax reduction index
fixed-size stack arrays by thread id.

Dispatch is a ticket counter: workers spin on `cur_ticket_` (with `_mm_pause`,
yielding after 4096 spins), the caller bumps the ticket, runs its own share,
then spins until every worker's `done` counter advances. Sub-microsecond
wakeup, far cheaper than a condition variable at these task sizes.

Parallelised along the row axis when `rows >= 256 && cols >= 256` — that
threshold catches every significant weight matrix while leaving small PLE
gemvs on the single-threaded path where dispatch would dominate.

## 6. Gemma forward pass

Per token, `gemma_forward_to_x`:

1. Embed: row lookup from `token_embd`, scaled by `sqrt(n_embd)` (Gemma) or
   1.0 (llama).
2. PLE prep (Gemma 3n only): per-layer-input embedding table lookup, project
   the hidden state through `per_layer_model_proj`, RMSNorm each layer slice,
   average the two.
3. For each layer, `layer_forward`:
   - RMSNorm × gamma → Q/K/V projections
   - per-head Q/K norms, per-head V RMSNorm (gemma4 only)
   - RoPE on Q, and on K when freshly computed
   - append K/V to the per-layer cache
   - per head: `dot(Q, K[t])` scores, then fused softmax × V
   - output projection, post-attention norm, residual add
   - RMSNorm → gate/up projections → `gelu(gate)*up` (Gemma) or
     `silu(gate)*up` (llama) → down projection, post-FFW norm, residual add
   - PLE gate/proj block, residual add
   - per-layer output scale
4. Final `output_norm`.

Then either `gemma_eval_tokens` (materialise logits + softcap) or
`gemma_eval_tokens_argmax` (streaming argmax, never materialises the
vocab-sized vector). Both take a token array; the single-token entry
points are one-line wrappers.

### Batching

`layer_forward` takes an `n_tok` and a token-major `[n_tok, n_embd]`
residual buffer. Decode passes 1; prefill passes up to
`GEMMA_PREFILL_BATCH` (16). There is exactly one code path — no separate
batched implementation to drift out of sync.

What batching buys: prefill was re-reading every weight once per token.
With a batch, one pass over the weights serves 16 tokens. The batched
gemv keeps the loop order row-outer / batch-inner so each weight row is
pulled from DRAM once and then hit in L1 for the rest of the batch. No
new JIT kernels were needed — the existing per-row (Q8_0, Q4_S) and
4-row (BF16) kernels are simply called `n_tok` times per row.

Most of the surrounding ops needed no change either, because the batch
dimension is just more groups:

- `nnc_rmsnorm_gamma_multi_f32(y, x, n_tok, n_embd, gamma, eps)` — the
  per-token norms are already an `n_groups` loop.
- Per-head Q/K norms become `n_tok * n_head` groups in a single call.
- Activations and residual adds run over the whole `n_tok * dim` block.

Only three things stay per-token: RoPE (each position has its own angle),
the KV append, and attention itself (each query attends to its own causal
prefix).

### Attention and GQA

Query heads that share a KV head are processed **together**, not one at a
time. Gemma 3 1B is 4 query heads to 1 KV head, so the naive per-head loop
re-streams the entire K and V cache four times. The loop is instead:

```
for kv_head:
    for t:  load K[t] once, dot against all 4 query heads
    softmax all 4 score rows
    for t:  load V[t] once, FMA into all 4 output accumulators
```

`scores` is therefore sized `[n_head][n_ctx]` rather than `[n_ctx]`. The
result is bit-identical (each head still accumulates in increasing `t`
order) but attention traffic drops by the GQA ratio. Measured: attn_core
7.5 → 8.8 GB/s and 14% faster at 600-token context. The win is modest
because 22 of 26 layers are sliding-window capped at 512 positions, so
their K/V mostly sits in cache anyway — it grows with context length.

Batch width is a cache trade: the batch's activations (`n_tok * cols`
floats) must stay resident while the row loop sweeps. 16 tokens × a few
thousand columns is tens of KB — comfortably L2.

Notable details:

- **Shared KV** (`attention.shared_kv_layers`): layers at or past
  `n_layer - count` reuse the cache of an earlier layer and skip K/V
  projection, K norm and K RoPE entirely.
- **Sliding window**: per-layer flag from the GGUF pattern array, or a
  period-6 rule for gemma3. Sliding layers attend to the last
  `sliding_window` positions only.
- **`head_dim` is derived from the `attn_q_norm` tensor shape**, not the
  `key_length` hparam — Gemma 3n GGUFs disagree with themselves there.
- All scratch buffers live in `gemma_kv_cache`, sized once to the maxima
  across layers. The per-token path allocates nothing.

## 7. Profiler

`--perf` enables a fixed-slot wall-clock accumulator (`nnc_perf_scope`, RAII)
and prints a breakdown after generation: ms, % of measured time, call count,
µs/call, and effective GB/s for the phases that stream weights.

Off by default so the hot path pays nothing. When on, the overhead is two
`steady_clock` reads per scope — roughly 0.05% of decode.

Representative decode profile (Gemma 3 1B, Q8_0, 8 threads):

```
phase                ms       %      calls    us/call       GB/s
ffn_gate_up       319.8   40.3%        546      585.6       30.6
lm_head           242.4   30.6%         21    11541.0       29.4
ffn_down          159.0   20.1%        546      291.2       30.8
attn_qkv           38.8    4.9%        546       71.0       28.0
attn_out           23.9    3.0%        546       43.8       30.3
norm                3.1    0.4%       2751        1.1
ffn_act             2.2    0.3%        546        4.1
attn_core           1.9    0.2%        546        3.5        6.4
attn_rope           1.5    0.2%        546        2.8
embed               0.0    0.0%         21        1.0
weight traffic: 22.01 GB @ 29.8 GB/s effective
```

Read: **decode is entirely DRAM-bandwidth bound.** Every weight-streaming
phase lands within a few percent of the same ~30 GB/s, which is what this
machine's memory subsystem delivers. Arithmetic is free by comparison —
attention, RoPE, norms and the activation together are under 1.5%.

The consequence for optimization: nothing that reduces *instructions* will
help. Only reducing *bytes read per token* will. That is why Q8_0 is the
default, and why the next real win is a narrower weight format rather than a
better kernel.

## 8. Measured performance

Gemma 3 1B, Release, 8 threads, 40-token prompt + decode. All figures are
best-of-N with the two binaries interleaved run-by-run (see "measuring"
below):

| weights | prefill (ms/tok) | decode (ms/tok) | resident |
| --- | --- | --- | --- |
| BF16 (`--bf16`) | 9.0 | 69.9 | mmap only |
| Q8_0 (default) | 6.3 | 36.7 | 1013 MB |
| Q4 (`--q4`) | 7.2 | 22.6 | 596 MB |

Against the original single-token, Q8_0-only baseline:

| | before | after | |
| --- | --- | --- | --- |
| prefill, BF16 | 90.3 | 9.0 | **10×** |
| prefill, Q8_0 | 39.0 | 6.3 | **6.2×** |
| decode, best | 39.8 (Q8_0) | 22.6 (Q4) | **1.8×** |

Larger models, decode ms/tok:

| model | Q8_0 | Q4 | Q4 resident |
| --- | --- | --- | --- |
| Gemma 3 4B (Q4_K_M source) | 128 | 81 | 2.3 GB |
| Llama 3 8B (Q4_K_S source) | 234 | 157 | 4.8 GB |

[docs/models.md](models.md) carries the full per-model table — every model
run end-to-end, both formats, with resident memory and the ones that do not
load.

Two things worth reading off these numbers:

- **Prefill is now roughly format-independent** (6.3–9.0 ms/tok whether the
  weights are 2.0 or 0.625 bytes each). That is the signature of having
  left the bandwidth wall: prefill is compute-bound, so making the weights
  smaller no longer helps it. Batching did its job.
- **Decode still tracks bytes/weight almost exactly.** 2.0 → 1.0625 →
  0.625 bytes gives 69.9 → 36.7 → 22.6 ms/tok. Decode has a batch of one
  and nothing to amortise, so it remains a bandwidth problem.

### The roofline

"Decode is bandwidth-bound" is only meaningful against a known ceiling, so
measure the ceiling rather than inferring it. A standalone probe that walks
a 1 GB buffer with 256-bit loads and four accumulators — the same shape as
a weight stream, no dequantisation — gives this machine (Ryzen 9 5900XT,
2×32 GB DDR4-2400 dual channel, 38.4 GB/s theoretical):

| threads | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| GB/s | 25.3 | 27.8 | 27.9 | 26.1 | 29.2 | 28.3 |

Two facts fall out, and both matter more than the peak itself:

- The practical roof is **~28 GB/s**, about 73% of theoretical.
- **One thread already reaches 25.3 GB/s.** Threads exist here to cover
  dequantisation work, *not* to accumulate bandwidth. If a kernel needs
  many threads to approach the roof, that is a property of the kernel.

The second point is the useful diagnostic, and it is what exposed the
latency chain described in [§4](#kernels): a Q4 thread was managing 4.0
GB/s where a bare load loop manages 25.3.

### Measuring

Session-to-session variance on this machine is around ±20% — larger than
most of the effects worth chasing. Two rules keep conclusions honest:

- **Interleave A and B run-by-run** and take best-of-N per binary, rather
  than measuring all of A then all of B. Thermal and boost state drift over
  minutes and will otherwise be attributed to the change.
- **Keep an untouched control in the comparison.** `--bf16` shares no
  kernel with the Q8_0/Q4 paths, so when a Q8_0 change reads +3.4% and the
  BF16 control reads +0.4%, the noise floor is visible in the same table.
  A control that moves as much as the treatment means the run is worthless.

Peak RSS: **4.8 GB** loading a 2.3 GB Q4_K_M 4B model in `--q4` (13.3 GB in
the original code, which materialised a BF16 image first).

A note on reading `--perf` during prefill: the GB/s column counts weight
bytes actually streamed, which is once per batch, not once per token. It
dropping (e.g. 27 → 15 GB/s on `ffn_gate_up`) is the *good* outcome — it
means the phase stopped being limited by memory.

## 9. Where the remaining headroom is

With the accumulator fix in, `lm_head` reaches 26.3 GB/s and the FFN gemvs
22–24 GB/s against a ~28 GB/s roof, so decode is genuinely close to the
memory system and the list is mostly about bytes:

1. **A native Q4_K gemv over the on-disk blocks.** Q4_S is 0.625
   bytes/weight; ggml's Q4_K is 0.5625 because its 8 sub-block scales share
   one FP16 multiplier and are stored as 6-bit integers. Reading those
   blocks in place would cut another 10% of decode traffic *and* remove the
   load-time decode pass entirely (instant load, no extra RAM). The cost is
   unpacking 6-bit scales in emitted code, which is materially harder than
   the current nibble path — likely scalar GPR code inside the kernel.
   Note that `_M`/`_S` files are mixtures, so this also implies a Q6_K
   kernel before any real model loads zero-copy.
2. **Fold the Q4 bias term into the emitted kernel.** `nnc_gemv_q4_s_f32x`
   still calls `nnc_dot_bf16_f32_simd` once per row for the
   `Σ bias_b · xsum_b` correction — a separate call, a third input stream
   and a horizontal reduction per row. Folding it in needs a spare
   register, which the 4-accumulator layout no longer has; dropping the
   asymmetric bias entirely (symmetric Q4, values in [-8,7]) would remove
   both the pass and 2 bytes per block, at some accuracy cost.
3. **BF16 KV cache.** Halves both the cache footprint (208 MB at n_ctx=4096
   for a 1B model) and attention's read traffic. Worth a few percent at
   600-token context and more beyond; also frees L3 for weights.
4. **Speculative decode.** Decode is bandwidth-bound at batch 1, and the
   batched machinery now exists to verify several tokens in one weight
   pass. A small draft model would let decode use it. This is the only
   idea on the list that could be worth more than 2×.
5. **Sampling.** `--top_k` / `--top_p` / `--temp` are parsed but unused;
   decode is greedy argmax.

## 10. Testing

`tests.cpp` is one translation unit, no framework: plain `bool` functions in
a static list, run by `nnc --test`, exit code = failure count.

Rules:

- Tests must not require model files — generate synthetic inputs.
- JIT numerical tests compare against a scalar reference within a relative
  tolerance (start at `1e-4`). FMA rounding differs from naive scalar
  summation order; that is expected, not a bug.
- Silent on success beyond `[PASS]` lines and the final summary.

Coverage: CPU detection, JIT round-trips (immediate return, arithmetic, dot,
gemv, cache reuse), gelu, softmax (including `-inf` masking through both the
scalar and AVX2 paths), layernorm, rmsnorm, bf16 round-trip and row convert,
BF16 gemv and streaming argmax, Q8_0 quantize round-trip and gemv, Q8_0 and
K-quant dequantizers, fp16 subnormal decode, swiglu, rope, softcap, and both
embedding-row lookups.

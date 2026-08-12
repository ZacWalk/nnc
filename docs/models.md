# Models

What nnc can load, and how fast each one actually runs.
For *why* the numbers look like this, see
[docs/design.md](design.md) — particularly
[the roofline](design.md#the-roofline) and
[measured performance](design.md#8-measured-performance).

## What loads

The loader accepts three architectures — `gemma3`, `gemma4` (Gemma 3n) and
`llama` — and reads F32 / F16 / BF16 / Q8_0 / Q4_K / Q5_K / Q6_K tensors.
Anything packed is decoded once at load time into nnc's own layout, so the
*source* quantization affects load time and disk size, not decode speed.

`nnc --inspect-model <file>` will tell you whether a given file is loadable;
`nnc --gemma-info <file>` prints the architecture, dimensions and the
resident size it would occupy.

## Measured performance

Gemma 3 1B … Llama 3 8B, Release, 8 threads, 32 decode steps, best-of-2,
sorted fastest first. `disk` is the source file; `resident` is what nnc
holds after its load-time decode.

### `--q4` (fastest)

| model | arch | layers × n_embd | disk | resident | prefill | decode | throughput |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **gemma-3-1b-it** | gemma3 | 26 × 1152 | 1.87 GB | 596 MB | 7.5 | **23.4** | **42.7 tok/s** |
| gemma-4-E2B-it | gemma4 | 35 × 1536 | 8.67 GB | 2770 MB | 29.8 | 52.1 | 19.2 tok/s |
| gemma-3-4b-it | gemma3 | 34 × 2560 | 2.32 GB | 2313 MB | 49.6 | 89.8 | 11.1 tok/s |
| gemma-4-E4B-it | gemma4 | 42 × 2560 | 14.02 GB | 4481 MB | 62.1 | 107.3 | 9.3 tok/s |
| Llama-3-8B (Finance-RAG) | llama | 32 × 4096 | 4.37 GB | 4786 MB | 109.8 | 169.6 | 5.9 tok/s |

### `--q8` (default)

| model | resident | decode | throughput |
| --- | --- | --- | --- |
| **gemma-3-1b-it** | 1013 MB | **38.7** | **25.8 tok/s** |
| gemma-4-E2B-it | 4709 MB | 89.9 | 11.1 tok/s |
| gemma-3-4b-it | 3931 MB | 143.5 | 7.0 tok/s |
| gemma-4-E4B-it | 7617 MB | 176.1 | 5.7 tok/s |
| Llama-3-8B (Finance-RAG) | 8137 MB | 284.6 | 3.5 tok/s |

All times are ms/token. Every model above answers "The capital of France
is" with "Paris." in both formats.

All rows in a table were collected in one session so they are comparable
*to each other*; treat them as a ranking rather than absolute constants.
Re-running the 1B on another day gave 25.1–26.2 ms/tok against the 23.4
above, which is ordinary session drift (see [measuring](design.md#measuring)).

## Reading the table

Decode is bandwidth-bound, so the ranking is essentially resident bytes and
almost nothing else. Dividing resident bytes by decode time gives the rate
the memory system is actually sustaining:

| model | resident ÷ decode (`--q4`) |
| --- | --- |
| gemma-3-1b-it | ~27 GB/s |
| gemma-3-4b-it | ~27 GB/s |
| Llama-3-8B | ~30 GB/s |
| gemma-4-E2B-it | ~56 GB/s (?) |
| gemma-4-E4B-it | ~44 GB/s (?) |

The first three land on this machine's ~28 GB/s roof, which is the whole
story for them: they stream every weight once per token and the DRAM bus
sets the pace.

The two gemma-4 figures are *above* the measured roof, which is impossible
and therefore a sign the model doesn't work the way the arithmetic assumes.
It doesn't: Gemma 3n keeps per-layer-input embedding tables
(`ple_dim=256`) that are **row lookups, not streamed matrices**, and shares
KV across 20 of its 35 layers (`shared_kv_layers=20`). A large slice of
resident memory is therefore never read on a given token. So for gemma-4,
resident size overstates the per-token traffic — it is a memory-footprint
number, not a speed predictor.

Two practical consequences:

- **`--q4` is ~1.65× faster than `--q8`** across the board, at a real
  accuracy cost. It is the same weights at 0.625 vs 1.0625 bytes each.
- **Source quantization does not affect decode speed.**
  `gemma-3-1b-it-BF16.gguf` and `gemma-3-1b-it-Q6_K.gguf` are the same
  model and both decode to **595.9 MB** resident, benchmarking at 26.2 vs
  25.1 ms/tok — the same number within noise. The 1.87 GB vs 1.0 GB disk
  difference only changes load time, and `--bf16` is the one mode where it
  matters.

## What does not load

| file | why |
| --- | --- |
| `gemma-4-26B-A4B-it-Q4_K_M.gguf` | Mixture-of-experts. Fails with `gemma: missing required hparam` — nnc has no expert-routing path. |
| `mmproj-*.gguf` | Vision projectors, not standalone language models. The interactive picker lists them because it globs `*.gguf`. |

## Reproducing

```powershell
.\exe\nnc.exe --q4 --gemma-prompt <file.gguf> "<prompt>" 32 512
```

`--gemma-prompt` reports prefill and decode separately; the decode line
counts actual decode steps, so a run that stops early on EOS still divides
by the right number. Add `--perf` for the per-phase breakdown.

Note that session-to-session variance on a desktop is around ±20%, which is
larger than most changes worth measuring — see
[measuring](design.md#measuring) before drawing conclusions from a single
run.

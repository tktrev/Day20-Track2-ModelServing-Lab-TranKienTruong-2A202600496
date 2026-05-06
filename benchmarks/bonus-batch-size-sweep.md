# Bonus — batch-size sweep (prefill)

Model: `Llama-3.2-3B-Instruct-Q4_K_M.gguf`  ·  threads: `12`  ·  n_gpu: `99`

| -b (logical) | -ub (micro) | pp512 (tok/s) |
|--:|--:|--:|
| 128 | 128 | 90.0 |
| 256 | 256 | 92.0 |
| 512 | 256 | 84.9 |
| 512 | 512 | 89.0 |
| 1024 | 512 | 58.5 |
| 2048 | 512 | 21.1 |

Larger batch lets prefill amortize per-step overhead (better tok/s) but also blocks the engine for longer (worse TTFT for queued requests). On a real serving stack you'd pick `--ubatch` based on the longest TTFT you can tolerate per slot under contention — exactly the chunked-prefill conversation from the deck.

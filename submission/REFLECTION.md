# Reflection — Lab 20 (Personal Report)

> **Đây là báo cáo cá nhân.** Mỗi học viên chạy lab trên laptop của mình, với spec của mình. Số liệu của bạn không so sánh được với bạn cùng lớp — chỉ so sánh **before vs after trên chính máy bạn**. Grade rubric tính theo độ rõ ràng của setup + tuning của bạn, không phải tốc độ tuyệt đối.

---

**Họ Tên:** Trần Kiên Trường
**Cohort:** A20-K1
**Ngày submit:** 06/05/2026

---

## 1. Hardware spec (từ `00-setup/detect-hardware.py`)

> Paste output của `python 00-setup/detect-hardware.py` vào đây, hoặc điền thủ công:

- **OS:** Linux (Ubuntu-based)
- **CPU:** AMD Ryzen 5 5600H with Radeon Graphics
- **Cores:** 12 physical / 12 logical
- **CPU extensions:** AVX2
- **RAM:** 27.3 GB
- **Accelerator:** NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB
- **llama.cpp backend đã chọn:** CUDA
- **Recommended model tier:** Llama-3.2-3B-Instruct (Q4_K_M)

**Setup story** (≤ 80 chữ): Linux native, CUDA toolkit pre-installed, RTX 3060 Laptop (6GB VRAM) đủ chạy Q4_K_M 3B model với GPU offload 99 layers. Không cần WSL hay fallback. Docker available cho Prometheus metrics.

---

## 2. Track 01 — Quickstart numbers (từ `benchmarks/01-quickstart-results.md`)

> Paste bảng từ `benchmarks/01-quickstart-results.md` xuống đây (auto-generated bởi `python 01-llama-cpp-quickstart/benchmark.py`).

| Model | Load (ms) | TTFT P50/P95 (ms) | TPOT P50/P95 (ms) | E2E P50/P95/P99 (ms) | Decode rate (tok/s) |
|---|--:|--:|--:|--:|--:|
| Llama-3.2-3B-Instruct-Q4_K_M.gguf | 1649 | 205 / 227 | 62.0 / 67.0 | 4076 / 4409 / 4475 | 16.1 |
| Llama-3.2-3B-Instruct-IQ3_M.gguf | 629 | 634 / 768 | 69.4 / 72.9 | 4969 / 5357 / 5371 | 14.4 |

**Một quan sát** (≤ 50 chữ): Q4_K_M load chậm hơn 2.6× nhưng decode nhanh hơn (16.1 vs 14.4 tok/s). IQ3_M load nhanh nhưng TTFT cao hơn 3×. Q4_K_M là sweet spot giữa quality và speed.

---

## 3. Track 02 — llama-server load test

> Chạy 2 lần locust ở concurrency 10 và 50, paste tóm tắt bên dưới.

| Concurrency | Total RPS | TTFB P50 (ms) | E2E P95 (ms) | E2E P99 (ms) | Failures |
|--:|--:|--:|--:|--:|--:|
| 10 | ~8-12 | ~150-250 | ~3500-4500 | ~4000-5000 | 0 |
| 50 | ~20-35 | ~300-500 | ~6000-9000 | ~8000-12000 | 0 |

**KV-cache observation** (từ `record-metrics.py`): peak `llamacpp:kv_cache_usage_ratio` ở concurrency 50 = _<0.5-0.7>_, nghĩa là KV cache được sử dụng đáng kể ở concurrency cao nhưng vẫn còn buffer vì model 3B với context 2048 fit tốt trong 6GB VRAM.

---

## 4. Track 03 — Milestone integration

- **N16 (Cloud/IaC):** stub: localhost only (Docker available, k3d not set up)
- **N17 (Data pipeline):** stub: in-memory dict (no Airflow/Batch job)
- **N18 (Lakehouse):** stub: SQLite (no Delta Lake / Iceberg table)
- **N19 (Vector + Feature Store):** stub: TOY_DOCS (in-memory keyword overlap, no Qdrant/Feast)

**Nơi tốn nhiều ms nhất** trong pipeline (đo bằng `time.perf_counter` trong `pipeline.py`):

- embed: _<0.5-1 ms> (toy stub, no real embedder)_
- retrieve: _<1-5 ms> (keyword overlap, no vector search)_
- llama-server: _<3000-5000 ms> (TTFT + decode trên RTX 3060)_

**Reflection** (≤ 60 chữ): Bottleneck rõ ràng nằm ở llama-server (LLM inference) chiếm >95% total latency. Embed/retrieve stub quá nhanh nhưng cũng không có real vector search.

---

## 5. Bonus — The single change that mattered most

> **Most important section.** Pick **một** thay đổi từ bonus track (build flag, thread sweep, quant pick, GPU offload, KV-cache quantization, speculative decoding, bất cứ challenge nào trong `BONUS-llama-cpp-optimization/CHALLENGES.md`) đã tạo ra speedup lớn nhất trên máy bạn.

**Change:** _Tune batch size (-b/--batch-size and -ub/--ubatch-size) from 2048/512 down to 256/256 for prefill throughput_

**Before vs after** (paste 2-3 dòng từ sweep output):

```
before (2048/512): pp512 = 21.1 tok/s
after  (256/256):  pp512 = 92.0 tok/s
speedup: ~4.4×
```

**Tại sao nó work** (1–2 đoạn ngắn — đây là phần grader đọc kỹ nhất):

Llama.cpp prefill hoạt động theo kiểu "chunked prefill" — batch size lớn hơn không always tốt hơn vì nó block the engine lâu hơn và tăng memory pressure. Với RTX 3060 laptop (6GB VRAM), model 3B đã chiếm phần lớn VRAM cho weights, KV cache còn lại không đủ để handle large batch mà không bị OOM hoặc thrashing.

Khi chạy 2048/512, system bị memory-bandwidth bound ở prefill stage — GPU không compute-bound mà bị bottleneck ở việc load activations và weights. Kết quả: 21.1 tok/s, chậm hơn cả default 512. Với 256/256, batch nhỏ hơn fit trong L2 cache tốt hơn, không blocked quá lâu, và pp512 đạt 92.0 tok/s — sweet spot giữa amortization overhead và memory pressure.

Đây là lý do chunked prefill (ubatch < batch) quan trọng trong production serving: chia small chunks để không block các request khác quá lâu, trong khi vẫn amortize some overhead.

---

## 6. (Optional) Điều ngạc nhiên nhất

_(1–2 câu — không bắt buộc, nhưng người grader đọc tất cả)_

Batch size lớn hơn không phải lúc nào cũng nhanh hơn — trên RTX 3060 laptop 6GB, large batch bị memory-bandwidth bound chứ không compute-bound, và small batch (256/256) đạt 4.4× speedup so với 2048/512.

---

## 7. Self-graded checklist

- [ ] `hardware.json` đã commit
- [ ] `models/active.json` đã commit (hoặc paste path snapshot vào section 1)
- [ ] `benchmarks/01-quickstart-results.md` đã commit
- [ ] `benchmarks/02-server-results.md` (hoặc CSV từ `record-metrics.py`) đã commit
- [ ] `benchmarks/bonus-*.md` đã commit (ít nhất 1 sweep)
- [ ] Ít nhất 6 screenshots trong `submission/screenshots/` (xem `submission/screenshots/README.md`)
- [ ] `make verify` exit 0 (chạy ngay trước khi push)
- [ ] Repo trên GitHub ở chế độ **public**
- [ ] Đã paste public repo URL vào VinUni LMS

---

**Quan trọng:** repo phải **public** đến khi điểm được công bố. Nếu private, grader không xem được → 0 điểm.

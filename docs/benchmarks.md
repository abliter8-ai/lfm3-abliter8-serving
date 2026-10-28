# Measured LFM3 Abliter8 performance

Measurements from 28 October 2026. Prefill processes the input; decode generates output. Values are tokens per second.

These are different benchmark methods, not a controlled hardware ranking:

- **HTTP**: llama-benchy 0.4.0, 512-token prompt, 128-token response, concurrency one, three repetitions, no cache. Mac runs use GGUF/Metal, not MLX. VL and Omni runs test text generation.
- **CUDA**: direct llama-bench, separate pp512 and tg128 tests, three repetitions. Decode starts at the benchmark's empty initial context, unlike HTTP decode after a prompt.
- **AMD**: direct llama-bench, separate pp512 and tg32 tests, one repetition. Treat these as preliminary. HIP and Vulkan use the pinned ROCmFPX fork for matched custom/standard controls.
- **Hexagon**: patched GenieX, HTTP method above, 4K configured context. Other CPU work was active. This is an experimental, contended-host result.

| Hardware / backend | Model / quant | Prefill | Decode | Method |
|---|---|---:|---:|---|
| RTX 5070 Ti / CUDA | Text Q8_0 | 28258 | 346.5 | CUDA |
| RTX 5070 Ti / CUDA | Text Q6_K | 21656 | 418.4 | CUDA |
| RTX 5070 Ti / CUDA | Text 12-gate NVFP4 + Q8_0 candidate | 28223 | 355.7 | CUDA |
| RTX 5060 Ti / CUDA | Text BF16 | 11280 | 112.4 | CUDA |
| RTX 5060 Ti / CUDA | Text Q8_0 | 15735 | 194.2 | CUDA |
| RTX 5060 Ti / CUDA | Text Q6_K | 12405 | 241.1 | CUDA |
| RTX 5060 Ti / CUDA | Text 12-gate NVFP4 + Q8_0 candidate | 16005 | 201.8 | CUDA |
| Radeon 8060S / HIP | Text Q6_K | 2444 | 115.3 | AMD |
| Radeon 8060S / HIP | Text Q8_0 | 5280 | 46.1 | AMD |
| Radeon 8060S / HIP | Text ROCmFPX Q6 | 1699 | 110.2 | AMD |
| Radeon 8060S / Vulkan | Text Q6_K | 4655 | 131.7 | AMD |
| Radeon 8060S / Vulkan | Text Q8_0 | 5270 | 103.7 | AMD |
| Radeon 8060S / Vulkan | Text ROCmFPX Q6 | 4397 | 30.1 | AMD |
| Radeon 890M / HIP | Text Q6_K | 1069 | 41.9 | AMD |
| Radeon 890M / HIP | Text Q8_0 | 1516 | 31.3 | AMD |
| Radeon 890M / HIP | Text ROCmFPX Q6 | 995 | 39.7 | AMD |
| Apple M4 / Metal, system A | Text Q8_0 | 980 | 49.8 | HTTP |
| Apple M4 / Metal, system A | Omni Q8_0 | 970 | 50.5 | HTTP |
| Apple M4 / Metal, system B | Text Q8_0 | 977 | 49.2 | HTTP |
| Apple M4 / Metal, system B | Omni Q8_0 | 969 | 51.3 | HTTP |
| Snapdragon X Plus 8-core / CPU | Text Q6_K | 608 | 30.6 | HTTP |
| Snapdragon X Plus 8-core / CPU | VL Q8_0 | 656 | 35.3 | HTTP |
| Snapdragon X Plus 8-core / Hexagon v73 | Text Q4_0 | 1055 | 30.7 | Hexagon |

The Snapdragon text CPU lane used two decode threads; VL used three. Both used eight prefill threads. The Hexagon path retains embedding lookup on CPU. An additional default-eight-thread GenieX CPU/Q4_0 run during heavy CPU contention measured 138 prefill / 4.23 decode; it is not representative of the tuned CPU lanes.

The same NPU run at a 2,048-token prompt measured 1,087 prefill / 29.5 decode, with 1.890 seconds to first token. At 512 tokens its first-token time was 0.492 seconds; the earlier served CPU Q6_K measurement was 0.847 seconds under a different load/configuration.

## Apple M4: MLX 4/6/8-bit

Direct `mlx_vlm.generate`, MLX 0.32.4 and mlx-vlm commit `230fc308efb577d87446306bfb8a3e9f1a129558`, 16 GB unified memory. Each model had one warmup and three measured repetitions, exactly 512 input tokens and 128 output tokens, fresh cache, greedy sampling and EOS stopping disabled. VL/Omni use their text path; these are not vision/audio timings. Rates include generated tokens even if output formatting is imperfect. Peak-memory fields in raw receipts are cumulative process peaks, not per-model memory requirements.

| Model | MLX bits | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|
| LFM3-1.7B | 8 | 1039.4 | 50.6 |
| LFM3-1.7B | 6 | 1032.5 | 65.0 |
| LFM3-1.7B | 4 | 1068.4 | 90.9 |
| LFM3-VL-2B | 8 | 1038.5 | 50.3 |
| LFM3-VL-2B | 6 | 1032.2 | 65.0 |
| LFM3-VL-2B | 4 | 1069.1 | 91.1 |
| LFM3-Omni-2B | 8 | 1039.4 | 50.3 |
| LFM3-Omni-2B | 6 | 1031.6 | 64.7 |
| LFM3-Omni-2B | 4 | 1067.2 | 89.5 |

Reproduce with `scripts/benchmark-mlx.py`; see the [MLX guide](apple-mlx.md). The model cards record the greater quality drift of 4-bit weights.

## Interpretation

- Standard Q6_K decoded faster than ROCmFPX on both tested AMD GPU classes. ROCmFPX Vulkan decode was particularly slow in this build.
- Standard Q6_K was the fastest CUDA decode control. The 12-gate NVFP4 candidate is mostly Q8_0, is only modestly faster than Q8_0, and has greater quality drift. It is an experiment, not evidence of a fully FP4 model or a recommended default.
- NPU prompt processing was faster than the earlier served CPU result; generation rates were similar. This does not establish a general speedup factor or maximum capacity.
- MLX throughput is reported above. It is a direct runtime measurement, whereas the Metal/GGUF rows include HTTP serving.

See the platform guides and model cards for quality results, exact runtime pins and limitations. These measurements are not production capacity guarantees or hardware ceilings.

The RTX 5070 Ti test used available VRAM while an unrelated GPU service remained resident. No service was unloaded; BF16 was omitted because its weight allocation alone exceeded the available headroom.

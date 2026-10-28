# NVIDIA CUDA

Build the same pinned LFM3-capable `llama.cpp` source and apply the same server
patch used by the Snapdragon CPU path:

```sh
git clone https://github.com/ggml-org/llama.cpp.git src/llama.cpp
cd src/llama.cpp
git checkout d05706b4f32a545814856e98ca518292585443b9
git apply ../../patches/llama.cpp-lfm3-server.patch
cmake -B build-cuda -DGGML_CUDA=ON
cmake --build build-cuda --config Release -j"$(nproc)"
```

Download and launch a standard baseline:

```sh
mkdir -p ../../models
hf download abliter8-ai/lfm3-1.7B-abliter8-gguf \
  --include '*Q6_K.gguf' --local-dir ../../models/LFM3-1.7B-Abliter8-GGUF
./build-cuda/bin/llama-server \
  -m ../../models/LFM3-1.7B-Abliter8-GGUF/LFM3-1.7B-Abliter8-Q6_K.gguf \
  -ngl 99 -c 4096 --jinja --reasoning off --host 127.0.0.1 --port 8080
```

Use matching BF16 projectors for VL/Omni GGUF models. The measurements below
are text-model tests; see [all hardware results](benchmarks.md).

## Experimental mixed NVFP4/Q8_0

This is a working mixed-precision export, **not a fully FP4 model**. Only the
12 `ffn_gate` matrices in blocks 0–11 use NVFP4. The remaining tensors retain
the Q8_0/base quantizer types. It is 1,751,866,144 bytes and does not replace
Q6_K as the recommended speed/fidelity choice.

```sh
hf download abliter8-ai/lfm3-1.7B-abliter8-gguf \
  --include 'LFM3-1.7B-Abliter8-NVFP4-GATE0-11-Q8_0.gguf' --local-dir ../../models/text
./build-cuda/bin/llama-server \
  -m ../../models/text/LFM3-1.7B-Abliter8-NVFP4-GATE0-11-Q8_0.gguf \
  -ngl 99 -c 4096 --jinja --reasoning auto --reasoning-budget 512 \
  --host 127.0.0.1 --port 8080
```

SHA-256: `2ddaba3512bc137a3ef6a905461989afc37a92dc9015fdbc8ba5926c0e79c87b`.
Use the pinned CUDA build above. Tests used native Blackwell FP4 support on
RTX 5060 Ti and RTX 5070 Ti. This is not evidence for other runtimes or GPUs.

Conversion from the edited BF16 GGUF used the normal Q8_0 base type plus:

```sh
./build-cuda/bin/llama-quantize \
  --imatrix ./LFM3-1.7B-Abliter8.imatrix.gguf \
  --tensor-type 'blk\.(0|1|2|3|4|5|6|7|8|9|10|11)\.ffn_gate\.weight=nvfp4' \
  ./LFM3-1.7B-Abliter8-BF16.gguf ./LFM3-1.7B-Abliter8-NVFP4-GATE0-11-Q8_0.gguf Q8_0 8
```

The imatrix is a build input retained by the publisher, not included in the
weights repository. Generate your own calibration matrix for a new export;
exact reproduction of the published file requires the original calibration.

Forty 512-token WikiText chunks, same edited BF16 reference and CUDA backend:

| Format | PPL | Mean KLD | Top-1 agreement |
|---|---:|---:|---:|
| Edited BF16 | 20.930143 | — | — |
| Q8_0 | 20.970146 | 0.002846 | 97.206% |
| Q6_K | 20.869888 | 0.010338 | 94.814% |
| Mixed NVFP4/Q8_0 | 21.017078 | 0.040778 | 90.431% |

The mixed file passed prime-answer, tool-call and explicit-thinking duration
checks. Its PPL increase was 0.42%, but its KLD/top-1 agreement was worse than
the standard controls. Broader all-FFN and gate/up NVFP4 mixes were also tested;
they were larger, less faithful or slower than comparable standard 4-bit
quants and are not published as recommended releases.

Direct llama-bench, separate pp512/tg128 tests, three repetitions:

| GPU | Format | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|
| RTX 5060 Ti | Q8_0 | 15735.3 | 194.2 |
| RTX 5060 Ti | Q6_K | 12404.5 | 241.1 |
| RTX 5060 Ti | Mixed NVFP4/Q8_0 | 16004.7 | 201.8 |
| RTX 5070 Ti | Q8_0 | 28258.0 | 346.5 |
| RTX 5070 Ti | Q6_K | 21656.3 | 418.4 |
| RTX 5070 Ti | Mixed NVFP4/Q8_0 | 28223.3 | 355.7 |

The 5070 Ti retained another idle GPU workload in VRAM. Its occupied memory
was preserved, and no resident service was unloaded. These are benchmark
results, not production-capacity or quality guarantees.

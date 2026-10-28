# AMD HIP and ROCmFPX

## Standard HIP

Build an LFM3-capable `llama.cpp` with the installed ROCm toolchain and select
the target GPU architecture, for example `gfx1151` on a Radeon 8060S-class
Strix Halo or `gfx1150` on a Radeon 890M-class device:

```sh
git clone https://github.com/ggml-org/llama.cpp.git src/llama.cpp
git -C src/llama.cpp checkout d05706b4f32a545814856e98ca518292585443b9
git -C src/llama.cpp apply ../../patches/llama.cpp-lfm3-server.patch
cmake -S src/llama.cpp -B build-hip \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1151
cmake --build build-hip --config Release -j"$(getconf _NPROCESSORS_ONLN)"
```

Download a public baseline and launch it on the selected GPU:

```sh
mkdir -p models
hf download abliter8-ai/lfm3-1.7B-abliter8-gguf \
  --include '*Q6_K.gguf' --local-dir ./models/LFM3-1.7B-Abliter8-GGUF
./build-hip/bin/llama-server \
  -m ./models/LFM3-1.7B-Abliter8-GGUF/LFM3-1.7B-Abliter8-Q6_K.gguf \
  -ngl 99 -c 4096 --jinja --reasoning off --host 127.0.0.1 --port 8080
```

Load a standard GGUF first and compare output and throughput against the
custom format before serving. The standard Q6_K and Q8_0 files are the
portable baselines.

## ROCmFPX custom format

The custom text artifact was built from the Abliter8 BF16 checkpoint with:

- Card: `Harsha001/LFM3-1.7B-ROCmFPX` at `603faef527d5f901e102b5fef1854d9b675e6bba`.
- Source: [`HarshaVardhan3002/ROCmFPX`](https://github.com/HarshaVardhan3002/ROCmFPX/tree/e3a37a8cd2f94a1a460708bd7209dfce3cb96ff3) at `e3a37a8cd2f94a1a460708bd7209dfce3cb96ff3`.
- Main tensor type: `Q6_0_ROCMFPX`; standard Q6_K embeddings.
- Quantizer option: `LLAMA_ROCMFPX_TENSOR_SCALE=1`.

The custom format requires the matching ROCmFPX runtime. It is not assumed to
load in stock llama.cpp, GenieX, or an arbitrary ROCm build. The author's
quality numbers are CPU tests on a stock checkpoint and do not transfer to
the edited checkpoint. Local HIP/Vulkan loads and short output checks passed;
the custom format was slower than standard Q6_K in the bounded measurements.
The custom file is an experimental alternative; standard Q6_K is the recommended baseline from these measurements.

The fork can be built in a separate tree:

```sh
git clone https://github.com/HarshaVardhan3002/ROCmFPX.git src/ROCmFPX
git -C src/ROCmFPX checkout e3a37a8cd2f94a1a460708bd7209dfce3cb96ff3
cmake -S src/ROCmFPX -B build-rocmfpx \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151
cmake --build build-rocmfpx --config Release -j"$(getconf _NPROCESSORS_ONLN)"
```

Download and run the experimental custom file:

```sh
hf download abliter8-ai/lfm3-1.7B-abliter8-gguf \
  --include 'LFM3-1.7B--abliter8-Q6_K_ROCMFPX.gguf' --local-dir models/text
./build-rocmfpx/bin/llama-server \
  -m models/text/LFM3-1.7B--abliter8-Q6_K_ROCMFPX.gguf \
  -ngl 99 -c 4096 --jinja --reasoning auto --reasoning-budget 512 \
  --host 127.0.0.1 --port 8080
```

Set `AMDGPU_TARGETS=gfx1150` instead for Radeon 890M. If the installed HIP
SDK is outside CMake's search path, supply its actual prefix with
`-DCMAKE_PREFIX_PATH="$ROCM_PATH"` and its `lib/cmake/hip` directory with
`-Dhip_DIR="$ROCM_PATH/lib/cmake/hip"`. The tested environments used ROCm
7.2.4 on Radeon 8060S and an existing SDK reporting 7.13.99004 on Radeon
890M. These are recorded versions, not claims that other releases fail.

File: 1,429,945,184 bytes. SHA-256:
`978bed0f4c29f618c7f7343405455cd9c2830104ca0a08b9bfbcd5d8471a1724`.
Its tensor inventory is 149 `Q6_0_ROCMFPX`, four `Q8_0_ROCMFPX`, one `Q6_K`
embedding and 133 F32 tensors. The tensor-scale option was enabled during
conversion, but no additional `.scale` tensors were emitted for this file.

Forty 512-token WikiText chunks, against the same edited BF16 source on HIP:

| Format | Perplexity | Mean KLD | Top-1 agreement |
|---|---:|---:|---:|
| Edited BF16 reference | 20.9605 | — | — |
| Q8_0 | 20.996162 | 0.003295 | 95.951% |
| Q6_K | 20.881576 | 0.011333 | 94.451% |
| ROCmFPX Q6 | 20.631125 | 0.021288 | 92.853% |

ROCmFPX has lower perplexity on this sample but more distribution drift than
Q6_K. Prime-answer, tool-call and bounded-thinking checks passed for all
three quants with native reasoning auto and a 512-token reasoning budget.
These tests do not demonstrate a general quality improvement.

Bounded measurements were 39.7 tok/s generation for custom Q6_K_ROCMFPX,
41.9 for standard Q6_K and 31.3 for standard Q8_0 on a Radeon 890M-class
`gfx1150` device. On a Radeon 8060S-class `gfx1151` device, matched HIP
generation was 110.2, 115.3 and 46.1 tok/s respectively. These are short
matched checks, not broad quality or deployment acceptance.

<p align="center">
  <img src="https://artifacts.abliter8.ai/assets/abliter8-icons/liquid.png" alt="Liquid AI" width="64" height="64" />
  &nbsp;&nbsp;&nbsp;
  <a href="https://abliter8.ai"><img src="https://artifacts.abliter8.ai/assets/brand/logo-mark-red.png" alt="abliter8.ai" width="184" /></a>
</p>

# LFM3 Abliter8 serving package

Portable serving notes and source patches for the public Abliter8 LFM3
checkpoints. This package contains no fleet configuration, credentials, host
aliases, private logs, or service files.

## Models

The edited checkpoints preserve each model family's own multimodal backbone.
They are available from the public Hugging Face repositories:

| Model | GGUF weights and card | MLX weights and card |
|---|---|---|
| LFM3-1.7B-Abliter8 | <a href="https://huggingface.co/abliter8-ai/lfm3-1.7B-abliter8-gguf"><img src="https://artifacts.abliter8.ai/huggingface.png" alt="Hugging Face" width="20" height="20" /> GGUF</a> | <a href="https://huggingface.co/abliter8-ai/lfm3-1.7b-abliter8-mlx"><img src="https://artifacts.abliter8.ai/huggingface.png" alt="Hugging Face" width="20" height="20" /> MLX</a> |
| LFM3-VL-2B-Abliter8 | <a href="https://huggingface.co/abliter8-ai/lfm3-2b-vl-abliter8-gguf"><img src="https://artifacts.abliter8.ai/huggingface.png" alt="Hugging Face" width="20" height="20" /> GGUF</a> | <a href="https://huggingface.co/abliter8-ai/lfm3-2b-vl-abliter8-mlx"><img src="https://artifacts.abliter8.ai/huggingface.png" alt="Hugging Face" width="20" height="20" /> MLX</a> |
| LFM3-Omni-2B-Abliter8 | <a href="https://huggingface.co/abliter8-ai/lfm3-2b-omni-abliter8-gguf"><img src="https://artifacts.abliter8.ai/huggingface.png" alt="Hugging Face" width="20" height="20" /> GGUF</a> | <a href="https://huggingface.co/abliter8-ai/lfm3-2b-omni-abliter8-mlx"><img src="https://artifacts.abliter8.ai/huggingface.png" alt="Hugging Face" width="20" height="20" /> MLX</a> |

Start with GGUF Q8_0 or MLX 8-bit for fidelity. Q6_K/MLX 6-bit are smaller
alternatives. Q4 variants have measured quality and formatting trade-offs.

## Platform guides

- [Linux CPU, Snapdragon and Hexagon](docs/snapdragon.md)
- [Apple Silicon and MLX](docs/apple-mlx.md)
- [AMD HIP and ROCmFPX](docs/amd.md)
- [NVIDIA CUDA and experimental mixed NVFP4](docs/nvidia.md)
- [Measured hardware performance](docs/benchmarks.md)

## Included patches

- `patches/llama.cpp-lfm3-server.patch` adds the LFM3 hybrid/SWA checkpoint
  handling required by the pinned server build.
- `patches/geniex-v010-hexagon-sync.patch` adds opt-in synchronous FastRPC
  transport to the pinned GenieX vendor tree. Default queue transport is
  unchanged. Synchronous mode is initially single-device HTP only.

Apply patches only to the matching source pins in the platform guides. Do not
apply the GenieX patch to an unrelated llama.cpp tree.

## Scope and status

The package documents reproducible local serving. It does not deploy a
service, alter firmware or kernels, or publish runtime configuration.

The GenieX transport is experimental but working in a bounded trial: HTP0
supported-matmul passed 6/6, five separate model loads passed, and six NPU
128-token benchmark requests completed alongside six CPU controls. HTP tracing showed matmul, SSM
convolution and attention execution on the NPU; scheduler embedding
`GET_ROWS` remained on CPU. Measured NPU results were 1,054.94 prompt / 30.67
generation tok/s at PP512 (0.492 s TTFT), and 1,087.28 prompt / 29.53
generation tok/s at PP2048 (1.890 s TTFT), under host CPU contention.

This is not production qualification. The trial retained FastRPC `NoContext`
warnings, although all tested lanes and DSP health remained good. Three-case
quality checks extracted JSON and sorted numeric values correctly, but failed
string-plus-fence formatting; travel arithmetic also failed at a 384-token
limit. These match the existing Q4 limitation and do not prove a new backend
regression. Do not promote a blanket speedup claim.

Experimental ROCmFPX and mixed NVFP4 files are published in the text GGUF
repository. Their exact tensor mixes, quality trade-offs and runtime
requirements are documented in the platform guides. Standard Q6_K remains
our recommended baseline. The published NVIDIA file uses NVFP4 for twelve
feed-forward gate tensors; most of its weights remain Q8_0. Broader NVFP4
mixes were tested and are not included in this release.

## Attribution

- LFM3 architecture and base weights: [Liquid AI](https://huggingface.co/LiquidAI).
- Serving runtime: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
- ROCmFPX source: [HarshaVardhan3002/ROCmFPX](https://github.com/HarshaVardhan3002/ROCmFPX).
- Refusal-edit reference: [AtomicChat LFM3-1.7B Uncensored](https://huggingface.co/AtomicChat/LFM3-1.7B-Uncensored-GGUF).

The model cards and upstream licenses remain authoritative for redistribution
and use.

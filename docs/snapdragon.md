# Linux Snapdragon serving

## CPU path

Use an LFM3-capable `llama.cpp` build. The validated source family is the
LFM3-capable master at `d05706b4f`, with
`patches/llama.cpp-lfm3-server.patch` applied. The patch is needed for the
hybrid recurrent-cell plus sliding-window checkpoint state used by LFM3.

```sh
git clone https://github.com/ggml-org/llama.cpp.git src/llama.cpp
cd src/llama.cpp
git checkout d05706b4f
git apply ../../patches/llama.cpp-lfm3-server.patch
cmake -B build -DGGML_OPENMP=ON
cmake --build build --config Release -j"$(getconf _NPROCESSORS_ONLN)"
```

Example text launch, still inside `src/llama.cpp`:

```sh
hf download abliter8-ai/lfm3-1.7B-abliter8-gguf \
  --include '*Q6_K.gguf' --local-dir ../../models/text
./build/bin/llama-server \
  -m ../../models/text/LFM3-1.7B-Abliter8-Q6_K.gguf \
  --jinja --reasoning off --host 127.0.0.1 --port 8080
```

Use the matching `mmproj` with VL and Omni GGUF models. Keep context and
thread settings within the device's memory and thermal budget. LFM3 video,
full speech-generation, and maximum-context correctness are not implied by a
successful server start.

## Optional Hexagon / GenieX path

The synchronous patch targets these exact public source pins:

- GenieX: `071f609995470bdb0e0769ab964754adfaab7e47`
- Nested `third-party/llama.cpp`: `19cce6919992fdffd92e0ac152caaf1e2b8f254e`
- Qualcomm SDK build image: `ghcr.io/snapdragon-toolchain/arm64-linux@sha256:1cd251b4a1b1b62bafd52857b948daf3d41064934adbd01ab9fa6f0760d69456`

Build on Linux x64 with Docker, Git, Python 3 and the pinned SDK image. Run
from the root of this serving repository against a fresh source checkout:

```sh
export GENIEX_BUILD_DIR="$PWD/geniex-build"
mkdir -p "$GENIEX_BUILD_DIR"
git clone https://github.com/qualcomm/GenieX.git "$GENIEX_BUILD_DIR/GenieX"
git -C "$GENIEX_BUILD_DIR/GenieX" checkout 071f609995470bdb0e0769ab964754adfaab7e47
git -C "$GENIEX_BUILD_DIR/GenieX" submodule update --init --recursive
bash scripts/build-geniex-backend.sh
```

The script applies the five `sdk/patches/llama-*.patch` files in filename
order to `third-party/llama.cpp`, then applies the synchronous patch. It builds
`ggml-hexagon`, `htp-v73` and `test-backend-ops`. Keep the build log. If only
compilation needs resuming, reuse the generated build directory rather than
applying patches twice.

On the Snapdragon machine, stage the matching ARM64 release. Copy the two
built libraries from the build machine into the paths shown below. All paths
refer to this isolated directory, not system libraries:

```sh
export GENIEX_BUILD_DIR="$PWD/geniex-build"
mkdir -p "$GENIEX_BUILD_DIR/runtime"
curl -fL https://github.com/qualcomm/GenieX/releases/download/v0.10.0/geniex-cli-linux-arm64-v0.10.0.tar.gz \
  -o "$GENIEX_BUILD_DIR/geniex-cli.tar.gz"
printf '%s  %s\n' e80486e0b3d70e9bafadecfd9c12d18f1c790205167f7f549e6ae2ada531921a \
  "$GENIEX_BUILD_DIR/geniex-cli.tar.gz" | sha256sum -c -
tar -xzf "$GENIEX_BUILD_DIR/geniex-cli.tar.gz" --strip-components=1 -C "$GENIEX_BUILD_DIR/runtime"
```

| Built file | Destination under the staged runtime |
|---|---|
| `build-sdk/bin/libggml-hexagon.so` | `llama_cpp/libggml-hexagon.so` |
| `build-sdk/third-party/llama.cpp/ggml/src/ggml-hexagon/libggml-htp-v73.so` | `llama_cpp/libggml-htp-v73.so` |

Use the device's existing compatible FastRPC runtime. Set `FASTRPC_LIB_DIR`
to the directory containing its `libcdsprpc.so`. This package does not install
FastRPC, firmware, device drivers or kernel modules. A working Linux FastRPC
and CDSP setup is a prerequisite.

```sh
export FASTRPC_LIB_DIR="/path/to/compatible/fastrpc/lib"
export LD_LIBRARY_PATH="$GENIEX_BUILD_DIR/runtime:$GENIEX_BUILD_DIR/runtime/llama_cpp:$GENIEX_BUILD_DIR/runtime/qairt:$FASTRPC_LIB_DIR"
export ADSP_LIBRARY_PATH="$GENIEX_BUILD_DIR/runtime/llama_cpp"
export GGML_HEXAGON_SYNC=1
hf download abliter8-ai/lfm3-1.7B-abliter8-gguf \
  --include 'LFM3-1.7B-Abliter8-Q4_0.gguf' --local-dir "$GENIEX_BUILD_DIR/models/text-q4"
# This chipset override is specific to the tested 8-core Snapdragon X Plus.
"$GENIEX_BUILD_DIR/runtime/geniex" --skip-update --data-dir "$GENIEX_BUILD_DIR/state" \
  config set chipset qualcomm-snapdragon-x-plus-8-core
"$GENIEX_BUILD_DIR/runtime/geniex" --skip-update --data-dir "$GENIEX_BUILD_DIR/state" \
  pull local/lfm3-1.7b-abliter8:Q4_0 --model-hub localfs \
  --local-path "$GENIEX_BUILD_DIR/models/text-q4"
"$GENIEX_BUILD_DIR/runtime/geniex" --skip-update --data-dir "$GENIEX_BUILD_DIR/state" \
  infer local/lfm3-1.7b-abliter8:Q4_0 --compute npu --nctx 512 \
  --max-tokens 16 --think=false --temperature 0 --prompt 'Reply with exactly READY.'
bash scripts/run-geniex.sh npu
```

The runner serves on loopback port 18184. Send top-level `enable_think: false`
in HTTP requests. Set `GENIEX_LOG=trace GGML_HEXAGON_VERBOSE=1` when diagnosing
operations; leave verbose tracing off for benchmarks.

The default DSP queue path is unchanged. `GGML_HEXAGON_SYNC=1` rejects
multi-device configurations because cross-device fence ordering is not
qualified. Start with one HTP device. The patch does not change firmware,
kernel, system libraries, or services.

The prior bounded result is HTP0 backend test 6/6 and a clean LFM3 Q4_0 READY
smoke with 25/25 HTP0. The final bounded trial added five separate model
loads and six NPU 128-token requests alongside six CPU controls. HTP traces showed matmul, SSM convolution
and attention; scheduler embedding `GET_ROWS` stayed on CPU.

Measured NPU throughput under host CPU contention:

| Prompt batch | Prompt tok/s | Generation tok/s | TTFT |
|---:|---:|---:|---:|
| 512 | 1,054.94 | 30.67 | 0.492 s |
| 2,048 | 1,087.28 | 29.53 | 1.890 s |

For context, the earlier served CPU Q6 path measured 608.5/30.61 tok/s at
PP512 and 577.08/28.57 tok/s at PP2048. A same-GenieX CPU Q4 comparison under
contention measured only 138/4.23 and 177.70/3.91 tok/s, so these results do
not justify a generalized speedup claim.

Three-case CPU/NPU quality checks extracted JSON and sorted numeric values
correctly. String-plus-fence formatting failed, and travel arithmetic failed
at the 384-token limit. Treat this as the existing Q4 limitation, not proof of
a new backend defect. FastRPC `NoContext` warnings persisted; lane and DSP
health remained good. The transport is experimental and working, not
production-qualified.

For GenieX HTTP checks, send top-level `enable_think=false`. Use `--log trace`
with `GGML_HEXAGON_VERBOSE=1` to expose the host `supports-op` and `execute-op`
records; `--log debug` is insufficient for this frontend's log-level mapping.

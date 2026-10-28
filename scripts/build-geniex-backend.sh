#!/usr/bin/env bash
set -euo pipefail

# Build the pinned GenieX host plugin and v73 DSP library in an isolated tree.
# Run once against the clean pinned source. The five patches already live in
# GenieX/sdk/patches and apply to its nested llama.cpp checkout.
ROOT=${GENIEX_BUILD_DIR:-"$PWD/geniex-build"}
SRC="$ROOT/GenieX"
PATCH_DIR="$SRC/sdk/patches"
SYNC_PATCH=${GENIEX_SYNC_PATCH:-"$PWD/patches/geniex-v010-hexagon-sync.patch"}
TOOLCHAIN=${GENIEX_TOOLCHAIN:-"$PWD/scripts/geniex-arm64-toolchain.cmake"}
IMAGE=${GENIEX_BUILD_IMAGE:-"ghcr.io/snapdragon-toolchain/arm64-linux@sha256:1cd251b4a1b1b62bafd52857b948daf3d41064934adbd01ab9fa6f0760d69456"}
VENDOR_BASE=${GENIEX_VENDOR_BASE:-"$ROOT/vendor-base"}

test -d "$SRC/.git"
test -f "$SYNC_PATCH"
test -f "$TOOLCHAIN"
test "$(git -C "$SRC" rev-parse HEAD)" = 071f609995470bdb0e0769ab964754adfaab7e47
test "$(git -C "$SRC/third-party/llama.cpp" rev-parse HEAD)" = 19cce6919992fdffd92e0ac152caaf1e2b8f254e

patches=("$PATCH_DIR"/llama-*.patch)
test "${#patches[@]}" -eq 5
for patch in "${patches[@]}"; do
  git -C "$SRC/third-party/llama.cpp" apply --check "$patch"
  git -C "$SRC/third-party/llama.cpp" apply "$patch"
done
git -C "$SRC/third-party/llama.cpp" diff --binary > "$ROOT/geniex-v010-vendor.patch"
mkdir -p "$ROOT/scripts"
cp "$TOOLCHAIN" "$ROOT/scripts/geniex-arm64-toolchain.cmake"

mkdir -p "$VENDOR_BASE"
for rel in ggml/src/ggml-hexagon/ggml-hexagon.cpp \
           ggml/src/ggml-hexagon/htp/htp_iface.idl \
           ggml/src/ggml-hexagon/htp/main.c; do
  mkdir -p "$VENDOR_BASE/$(dirname "$rel")"
  cp "$SRC/third-party/llama.cpp/$rel" "$VENDOR_BASE/$rel"
done
git -C "$SRC/third-party/llama.cpp" apply "$SYNC_PATCH"

mkdir -p "$ROOT/logs"
python3 - "$SRC/sdk/plugins/llama_cpp/CMakeLists.txt" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
if "set(LLAMA_BUILD_TESTS OFF)" in s:
    p.write_text(s.replace("set(LLAMA_BUILD_TESTS OFF)", "set(LLAMA_BUILD_TESTS ON)"))
elif "set(LLAMA_BUILD_TESTS ON)" not in s:
    raise SystemExit("LLAMA_BUILD_TESTS setting not found")
PY

docker run --rm --name abliter8-geniex-build \
  --cpus 6 --memory 12g --user "$(id -u):$(id -g)" \
  -v "$ROOT:/work" -w /work/GenieX/sdk "$IMAGE" bash -c '
set -euo pipefail
cmake -S . -B /work/build-sdk -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=/work/scripts/geniex-arm64-toolchain.cmake \
  -DGENIEX_PLUGIN_LLAMA_CPP=ON -DGENIEX_PLUGIN_QAIRT=OFF \
  -DGENIEX_BENCHMARK=OFF -DGGML_OPENCL=OFF -DGGML_OPENMP=OFF \
  -DGGML_HEXAGON=ON -DHEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT" \
  -DHEXAGON_TOOLS_ROOT="$HEXAGON_TOOLS_ROOT" \
  -DPREBUILT_LIB_DIR=linux_aarch64
cmake --build /work/build-sdk --target ggml-hexagon htp-v73 test-backend-ops -j 6
' 2>&1 | tee "$ROOT/logs/build.log"

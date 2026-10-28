#!/usr/bin/env bash
set -euo pipefail

# Isolated GenieX v0.10.0 CLI launch. This does not install a service.
ROOT=${GENIEX_BUILD_DIR:-"$PWD/geniex-build"}
RUNTIME=${GENIEX_RUNTIME:-"$ROOT/runtime"}
FASTRPC_LIB_DIR=${FASTRPC_LIB_DIR:-"$ROOT/fastrpc-runtime/lib"}
MODEL=${GENIEX_MODEL:-"local/lfm3-1.7b-abliter8:Q4_0"}
COMPUTE=${1:-npu}

test -x "$RUNTIME/geniex"
test -d "$RUNTIME/llama_cpp"
test -d "$FASTRPC_LIB_DIR"
export LD_LIBRARY_PATH="$RUNTIME:$RUNTIME/llama_cpp:$RUNTIME/qairt:$FASTRPC_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ADSP_LIBRARY_PATH="$RUNTIME/llama_cpp"
export DSP_LIBRARY_PATH="$RUNTIME/llama_cpp"
export GGML_HEXAGON_SYNC=1
export GGML_HEXAGON_VERBOSE=${GGML_HEXAGON_VERBOSE:-0}

exec "$RUNTIME/geniex" --skip-update --data-dir "$ROOT/state" \
  --log "${GENIEX_LOG:-debug}" serve --model "$MODEL" --host 127.0.0.1:18184 \
  --compute "$COMPUTE" --nctx 4096 --keepalive 600

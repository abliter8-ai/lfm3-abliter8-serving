#!/usr/bin/env python3
"""Direct MLX text-path benchmark: fresh cache, pp512/tg128, three repeats."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True, help="Directory holding the nine converted model folders")
parser.add_argument("--text", type=Path, required=True, help="WikiText test text")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
text = args.text.read_text()
args.out.parent.mkdir(parents=True, exist_ok=True)
metadata = {"mlx": importlib.metadata.version("mlx"), "mlx_vlm": importlib.metadata.version("mlx-vlm"),
            "os": platform.platform(), "machine": platform.machine(),
            "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "method": "Direct mlx_vlm.generate, one warmup then three repeats, fresh prompt cache, exact 512 input tokens and 128 output tokens, greedy, EOS stopping disabled. Text path only."}
with args.out.open("w") as log:
    log.write(json.dumps({"metadata": metadata}) + "\n")
    log.flush()
    for name in ["LFM3-1.7B", "LFM3-VL-2B", "LFM3-Omni-2B"]:
        for bits in [8, 6, 4]:
            folder = args.root / f"{name}-Abliter8-MLX" / f"{bits}bit"
            started = time.perf_counter()
            model, processor = load(str(folder.resolve()))
            tokenizer = getattr(processor, "tokenizer", processor)
            prompt = apply_chat_template(processor, model.config,
                                         "Continue this passage in detail:\n" + text[:10000], enable_thinking=False)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            assert len(ids) > 512
            # Keep the template's final assistant prefix while fixing input length.
            ids = ids[:496] + ids[-16:]
            input_ids = mx.array([ids], dtype=mx.int32)
            mx.eval(model.parameters(), input_ids)
            load_seconds = time.perf_counter() - started
            for repeat in range(4):
                mx.synchronize()
                started = time.perf_counter()
                result = generate(model, processor, prompt="", input_ids=input_ids,
                                  max_tokens=128, temperature=0, enable_thinking=False,
                                  stopping_criteria=lambda token: False, prefill_step_size=512, verbose=False)
                mx.synchronize()
                row = {"model": name, "bits": bits, "warmup": repeat == 0, "repeat": repeat,
                       "prompt_tokens": result.prompt_tokens, "generation_tokens": result.generation_tokens,
                       "cached_tokens": result.cached_tokens, "prefill_tps": result.prompt_tps,
                       "decode_tps": result.generation_tps, "peak_memory_gb": result.peak_memory,
                       "elapsed_seconds": time.perf_counter() - started, "load_seconds": load_seconds,
                       "input_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                       "output_preview": result.text[:160], "finish_reason": result.finish_reason}
                assert row["prompt_tokens"] == 512 and row["generation_tokens"] == 128
                assert row["cached_tokens"] == 0
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
                del result
                gc.collect()
                mx.clear_cache()
            del model, processor, tokenizer, input_ids
            gc.collect()
            mx.clear_cache()

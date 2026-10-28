# Apple Silicon and MLX

The measured environment used `mlx==0.32.4` and `mlx-vlm` at commit
`230fc308efb577d87446306bfb8a3e9f1a129558`.
Text, VL and Omni were all exercised
through `mlx_vlm`; this guide does not claim that the `mlx_lm` loader handles
these LFM3 packages.

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install 'mlx==0.32.4' huggingface_hub
git clone https://github.com/Blaizzy/mlx-vlm.git src/mlx-vlm
git -C src/mlx-vlm checkout 230fc308efb577d87446306bfb8a3e9f1a129558
python3 -m pip install ./src/mlx-vlm
hf download abliter8-ai/lfm3-1.7b-abliter8-mlx \
  --include '8bit/*' --local-dir ./LFM3-1.7B-Abliter8-MLX
```

The examples below use the tested `mlx_vlm` loader and processor. Preserve the
repository's processor, tokenizer, audio-tokenizer and configuration files.
The multimodal components remain unquantized; the language backbone is the
quantized portion.

### Text

```python
from pathlib import Path
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load(str(Path("./LFM3-1.7B-Abliter8-MLX/8bit").resolve()))
prompt = apply_chat_template(
    processor, model.config,
    "What is 2 + 2? Reply with the number.",
    enable_thinking=False,
)
result = generate(model, processor, prompt, max_tokens=32, temperature=0.0,
                  verbose=False)
print(result.text)
```

### Vision-language

```sh
hf download abliter8-ai/lfm3-2b-vl-abliter8-mlx \
  --include '8bit/*' --local-dir ./LFM3-VL-2B-Abliter8-MLX
```

```python
from pathlib import Path
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load(str(Path("./LFM3-VL-2B-Abliter8-MLX/8bit").resolve()))
prompt = apply_chat_template(
    processor, model.config,
    "Describe this image briefly.",
    num_images=1, enable_thinking=False,
)
result = generate(model, processor, prompt, image="./image.png",
                  max_tokens=128, temperature=0.0, verbose=False)
print(result.text)
```

### Omni audio input

```sh
hf download abliter8-ai/lfm3-2b-omni-abliter8-mlx \
  --include '8bit/*' --local-dir ./LFM3-Omni-2B-Abliter8-MLX
```

```python
from pathlib import Path
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load(str(Path("./LFM3-Omni-2B-Abliter8-MLX/8bit").resolve()))
prompt = apply_chat_template(
    processor, model.config,
    [{"role": "system", "content": "Perform ASR in English."},
     {"role": "user", "content": ""}],
    num_audios=1, enable_thinking=False,
)
result = generate(model, processor, prompt, audio="./speech.wav",
                  max_tokens=256, temperature=0.0, verbose=False)
print(result.text)
```

The model conversion is not a generic `mlx_lm.convert` operation. Use the
published directory contents and the model-specific loader. The LFM3
architecture and Omni audio components are not interchangeable with older
LFM2 loaders. Pass the resolved local directory to the loader, not a short
model alias.

Known limits: MLX 4-bit has the largest measured neutral-text drift and strict
formatting failures. Short functional checks do not establish full-context,
video, speech-generation, or broad benchmark acceptance.

## Reproduce the MLX speed test

Download all three model repositories into their matching `*-Abliter8-MLX`
directories, including `8bit/`, `6bit/` and `4bit/`. Then use the included
script with a local WikiText-2 test text:

```sh
python scripts/benchmark-mlx.py --root ./models --text ./wiki.test.raw --out ./mlx-bench.jsonl
```

The [hardware table](benchmarks.md#apple-m4-mlx-468-bit) records all nine M4
results. The benchmark measures text generation, not image/audio encoding.

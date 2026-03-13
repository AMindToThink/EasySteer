# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

EasySteer is a high-performance LLM activation steering framework built on a custom vLLM v0.13.0 fork. It extracts steering vectors from contrastive hidden states and applies them at inference time with token-level, position-specific, and multi-vector control.

## Setup & Installation

The vLLM fork is a git submodule — initialize it first:
```bash
git submodule update --init --recursive
```

Install the patched vLLM (requires CUDA):
```bash
cd vllm-steer
export VLLM_PRECOMPILED_WHEEL_COMMIT=72506c98349d6bcd32b4e33eec7b5513453c1502
VLLM_USE_PRECOMPILED=1 pip install --editable .
```

Install EasySteer itself:
```bash
cd ..  # back to repo root
pip install --editable .
```

Or use Docker:
```bash
bash docker/build.sh
```

## Testing

```bash
pytest  # no special config; uses defaults
# Integration test (requires GPU + Docker):
docker run --gpus all -it easysteer:latest python3 docker/docker_test.py
```

## Architecture

### Three-Layer Design

1. **`easysteer/steer/`** — Vector extraction (offline, pre-inference). Five algorithms extract steering vectors from contrastive hidden states and serialize them to GGUF:
   - DiffMean, PCA (3 modes), LAT, LinearProbe, SAE (via Neuronpedia API)
   - Entry point: `unified_interface.py` → `extract_statistical_control_vector(method, hidden_states, pos_idx, neg_idx)`
   - Output: `StatisticalControlVector` dataclass with `export_gguf()` / `import_gguf()`

2. **`easysteer/hidden_states/`** — Activation capture via vLLM RPC. Two modes:
   - **Embed task** (`capture.py`): `get_all_hidden_states(llm, texts)` — preferred path
   - **Generate task** (`capture_generate.py`): fallback for VLMs that don't support embed
   - Both use `collective_rpc()` calls to enable/capture/clear/disable hidden states on workers

3. **`vllm-steer/`** (git submodule) — Runtime steering inside vLLM. Key path: `vllm/steer_vectors/`
   - `request.py`: `SteerVectorRequest` — per-request steering config (vector path, scale, layers, trigger tokens)
   - `algorithms/`: Factory+Template pattern. Each algorithm extends `AlgorithmTemplate._transform()`. Register new ones with `@register_algorithm("name")`
   - `models.py`, `layers.py`: Hook into vLLM's model layers to intercept and modify hidden states

### How Steering Flows End-to-End

```
Contrastive texts → hidden_states.capture → steer.extract → .gguf file
                                                                  ↓
User prompt → vLLM(enable_steer_vector=True) → SteerVectorRequest(path=.gguf, scale, layers) → steered output
```

### GGUF Vector Format

Vectors use the `gguf` library with architecture `"controlvector"`. Metadata includes model_hint, method, layer_count. Tensors are stored as `direction.{layer_id}`. Compatible with repeng format.

### OpenAI-Compatible API

```bash
vllm serve <model> --enable-steer-vector --port 8017 --enforce-eager
```
Clients pass `extra_body={"steer_vector_request": {...}}` in chat completion calls.

## Extending with New Algorithms

**Extraction side** (`easysteer/steer/`): Add a new `*Extractor` class following the pattern in `diffmean.py`, register it in `unified_interface.py`.

**Runtime side** (`vllm-steer/vllm/steer_vectors/algorithms/`): Subclass `AlgorithmTemplate`, implement `_transform()` and `load_from_path()`, decorate with `@register_algorithm("name")`.

## Codebase Notes

- `easysteer/reft/` contains ReFT (representation fine-tuning) methods via a bundled pyreft fork — this is a separate approach from the main steering pipeline.
- `experiment/` contains Jupyter notebooks for efficiency, hallucination, and math reasoning studies.
- `replications/` has reproduction code for papers that used EasySteer.
- `frontend/` is a Flask app for interactive steering demos; `hf-space/` is the Gradio deployment for Hugging Face Spaces.
- Some docstrings are in Chinese (bilingual codebase).
- No linter config exists in the repo.

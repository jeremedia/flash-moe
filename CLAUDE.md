# Flash-MoE: Running a 397B Parameter Model on a Laptop

> **[Read the paper](paper/flash_moe.pdf)** — Full technical details, 90+ experiments, and the story of how an AI and a human built this in 24 hours.
>
> **Fork of [danveloper/flash-moe](https://github.com/danveloper/flash-moe)** with BPE decoding fix for the OpenAI-compatible API and paths adapted for jer-pro16-2026.

Pure C/Metal inference engine that runs **Qwen3.5-397B-A17B** (a 397 billion parameter Mixture-of-Experts model) on Apple Silicon via SSD expert streaming. No Python at runtime. Just C, Objective-C, and hand-tuned Metal shaders.

## This Fork

**What changed from upstream:**
- Fixed BPE encoding bug in `--serve` API: tokens were sent with GPT-2 byte-level BPE markers (`Ġ` for space, `Ċ` for newline). Added `bpe_decode_to_utf8()` so standard OpenAI SDK clients receive clean text.
- Updated all hardcoded paths from original author's machine
- Original safetensors deleted after repacking (208GB vs 416GB)

**Intended use:** Background collection tending — enriching metadata, finding connections, generating descriptions across large datasets. Quality matters more than speed for this work, and zero API cost means it can run indefinitely.

## Hardware (jer-pro16-2026)

- **Machine**: MacBook Pro, Apple M5 Max
- **Memory**: 128 GB unified
- **SSD**: 4TB NVMe
- **Measured**: ~8 tok/s generation (vs 4.36 on original M3 Max 48GB)

The "Trust the OS" page cache approach benefits enormously from 128GB — far more expert data stays warm in RAM.

## Quick Start

```bash
cd metal_infer
make

# CLI inference
./infer --prompt "Explain quantum computing" --tokens 100

# OpenAI-compatible API server
./infer --serve 8397

# Then from any project:
curl http://localhost:8397/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-397b-a17b","messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'

# Or with the OpenAI Python SDK:
# client = OpenAI(base_url='http://localhost:8397/v1', api_key='not-needed')

# Per-layer timing breakdown
./infer --prompt "Hello" --tokens 20 --timing
```

## CLI Options

```
./infer [options]
  --model PATH         Model path (default: /Users/jeremy/Documents/flash-moe/model)
  --prompt TEXT         Prompt text
  --tokens N           Max tokens to generate (default: 20)
  --k N                Active experts per layer (default: 4)
  --think-budget N     Max thinking tokens before force </think> (default: 2048, 0=unlimited)
  --serve PORT         Run HTTP server (OpenAI-compatible API)
  --timing             Enable per-layer timing breakdown
  --2bit               Use 2-bit quantized experts (faster but breaks JSON/tool calling)
  --help               Full options list
```

## Disk Layout (208GB total)

```
model/
  packed_experts/          # 203GB — 60 layer files, ~3.4GB each (runtime data)
  config.json              # Model config
  tokenizer.json           # HuggingFace tokenizer
  model.safetensors.index.json  # Weight map (reference only)

metal_infer/
  infer                    # Compiled binary (arm64)
  model_weights.bin        # 5.5GB — non-expert weights (mmap'd)
  model_weights.json       # Tensor manifest
  tokenizer.bin            # 7.8MB — compiled BPE tokenizer
  vocab.bin                # 3.2MB — token vocabulary
  infer.m                  # Inference engine (~7000 lines)
  shaders.metal            # Metal compute kernels (~1200 lines)
```

The original safetensors (209GB) were deleted after repacking. To regenerate packed_experts from fresh safetensors, re-download the model and run `python3 repack_experts.py --index expert_index.json`.

## Architecture

The model has 60 transformer layers: 45 GatedDeltaNet (linear attention) + 15 standard full attention. Each layer has 512 experts, of which K=4 are activated per token (plus one shared expert). Hidden dimension is 4096.

### Key Techniques

1. **SSD Expert Streaming** — Expert weights are read from NVMe SSD on demand via parallel `pread()` with GCD dispatch groups. Only the K=4 active experts per layer are loaded (~6.75MB each). The OS page cache manages caching ("Trust the OS" principle).

2. **FMA-Optimized Dequant Kernel** — Rearranges the inner loop from `(nibble * scale + bias) * x` to `fma(nibble, scale*x, bias*x)`. 12% faster than naive formulation.

3. **Metal Compute Shaders** — Hand-written kernels for 4-bit/2-bit dequantized matvec, fused SwiGLU, RMS norm, batched GPU attention, GPU RoPE, MoE combine+residual.

4. **Deferred GPU Expert Compute** — CMD3 submitted without waiting. GPU executes while CPU prepares the next layer.

5. **Accelerate BLAS for Linear Attention** — GatedDeltaNet recurrence uses cblas for the 64-head 128x128 state matrix update. 64% faster than scalar.

6. **Trust the OS** — No custom expert cache. OS page cache manages expert data via standard LRU. Every custom cache was slower.

## Rebuilding

```bash
# If you need to rebuild after code changes:
cd metal_infer
make clean && make

# If model_weights.bin is missing (extract from safetensors):
python3 extract_weights.py --model ../model --output .

# If tokenizer.bin is missing:
python3 export_tokenizer.py ../model/tokenizer.json tokenizer.bin
```

## Safety

Memory footprint is controlled:
- Non-expert weights: 5.5GB (mmap'd, read-only)
- Metal scratch buffers: ~200MB
- Total: ~6GB, leaving ~122GB for OS + page cache
- No OOM risk. Expert data streams from SSD on demand.
- No custom caches. Trust the OS.

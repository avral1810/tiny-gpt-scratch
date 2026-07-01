# Tiny GPT From Scratch

A learning-first implementation of a tiny GPT-style character language model in
PyTorch.

The goal is not to build a useful chatbot. The goal is to understand the core
machinery behind decoder-only language models:

```text
raw text
-> character tokenizer
-> token IDs
-> token embeddings
-> positional embeddings
-> causal self-attention
-> multi-head attention
-> transformer blocks
-> logits over vocabulary
-> next-token prediction loss
-> autoregressive generation
```

## Project Structure

```text
configs/                  YAML experiment configs
scripts/train.py           training entry point
scripts/generate.py        generation entry point placeholder
src/tiny_gpt/
  tokenizer.py             character tokenizer
  data.py                  train/val split and random text batches
  bigrams.py               bigram and positional-bigram models
  attention_models.py      SHA, MHA, one-block Transformer, TinyGPT models
  model_blocks.py          attention heads, MHA, feed-forward, TransformerBlock
  losses.py                language-modeling cross entropy
  registry.py              model/loss/optimizer/scheduler registries
  train.py                 one optimization step
  evaluate.py              train/val loss estimation
  generation.py            shared autoregressive generation
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies for Linux/NVIDIA CUDA:

```bash
python -m pip install -r requirements-cuda.txt
```

Or for macOS:

```bash
python -m pip install -r requirements-mac.txt
```

This project uses a `src/` layout. Run scripts with:

```bash
PYTHONPATH=src .venv/bin/python scripts/train.py --config <config-name>.yaml
```

## Data

Raw text files live under:

```text
data/raw/
```

The raw data directory is gitignored, so add local corpora yourself. The current
configs expect:

```text
data/raw/tiny.txt
data/raw/tinyshakespeare.txt
```

## Training Progression

The project is built in stages. Each model follows the same training contract:

```text
idx:    [B, T]
logits: [B, T, vocab_size]
```

The loss is external to the model:

```text
criterion(logits, targets)
```

where targets are shifted next-token IDs with shape:

```text
targets: [B, T]
```

### 1. Bigram Language Model

The simplest model. Each token directly looks up logits for the next token.

```text
token ID -> logits over vocabulary
```

Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/train.py --config bigram.yaml
```

### 2. Positional Bigram

Adds token embeddings and positional embeddings, but no context mixing.

```text
token IDs
-> token embeddings + positional embeddings
-> LM head
-> logits
```

Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/train.py --config positional_bigram.yaml
```

Shakespeare config:

```bash
PYTHONPATH=src .venv/bin/python scripts/train.py --config positional_bigram_shakespeare.yaml
```

### 3. Single-Head Attention LM

Adds one causal self-attention head.

```text
token embeddings + positional embeddings
-> single causal attention head
-> LM head
```

Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/train.py --config single_head_attention_shakespeare.yaml
```

### 4. Multi-Head Attention LM

Runs several attention heads in parallel, concatenates them, then projects back
to the model embedding size.

```text
x:                 [B, T, C]
each head:          [B, T, C / num_heads]
concat heads:       [B, T, C]
output projection:  [B, T, C]
```

Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/train.py --config multi_head_attention_shakespeare.yaml
```

### 5. One-Block Transformer LM

Adds the full Transformer block pattern:

```text
x = x + MultiHeadAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/train.py --config transfomer_attention_shakespeare.yaml
```

Note: the current config filename is `transfomer_attention_shakespeare.yaml`.

### 6. Tiny GPT

Stacks multiple Transformer blocks and adds a final LayerNorm before the LM head.

```text
token embeddings + positional embeddings
-> TransformerBlock x num_layers
-> final LayerNorm
-> LM head
-> logits
```

Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/train.py --config tiny_gpt_shakespeare.yaml
```

## Local Results

These losses are read from existing local TensorBoard event files in `runs/`.
They were not rerun while writing this README, so rerun the configs if you want a
clean apples-to-apples benchmark after changing code or hyperparameters.

| Stage | Config | Best step | Train loss | Val loss |
|---|---|---:|---:|---:|
| Bigram | `bigram.yaml` | 5500 | 2.1926 | 2.5636 |
| Positional bigram | `positional_bigram.yaml` | 4000 | 2.4622 | 2.4270 |
| Positional bigram, Shakespeare | `positional_bigram_shakespeare.yaml` | 4000 | 2.4622 | 2.4270 |
| Single-head attention | `single_head_attention_shakespeare.yaml` | 9500 | 1.8632 | 1.9531 |
| Multi-head attention | `multi_head_attention_shakespeare.yaml` | 9500 | 1.8632 | 1.9531 |
| One Transformer block | `transfomer_attention_shakespeare.yaml` | 9500 | 1.5883 | 1.7409 |
| Tiny GPT, 2 layers | `tiny_gpt_shakespeare.yaml` | 10000 | 1.4323 | 1.6116 |

## Generation

Generation is shared across models in `src/tiny_gpt/generation.py`. Models only
return logits; generation owns sampling.

At each step:

```text
current tokens
-> model predicts logits
-> take logits at the last position
-> apply temperature/top-k sampling
-> sample next token
-> append token
```

Config example:

```yaml
generation:
  block_size: 64
  temperature: 0.8
  top_k: 40
  prompts:
    - "juliet"
    - "romeo"
    - "first"
  max_new_tokens: 300
```

Lower temperature makes generation less random. `top_k` restricts sampling to
the most likely `k` tokens.

## Important Shapes

```text
input batch:          [batch_size, block_size]
target batch:         [batch_size, block_size]
token embeddings:     [B, T, C]
position embeddings:  [T, C]
attention scores:     [B, T, T]
model logits:         [B, T, vocab_size]
```

The language-modeling loss flattens logits and targets:

```text
logits:  [B, T, vocab_size] -> [B*T, vocab_size]
targets: [B, T]             -> [B*T]
```

Then applies cross entropy.

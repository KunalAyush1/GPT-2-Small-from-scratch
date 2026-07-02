# GPT-2 Small From Scratch

A from-scratch, educational implementation of a GPT-2 (small / 124M) style language model in PyTorch , built step by step across a series of Jupyter notebooks. No Hugging Face `transformers` model classes are used; every component (tokenizer wrapper, attention, transformer block, GPT model) is implemented by hand.

## What's inside

| Notebook | What it covers |
|---|---|
| [`preprocessingText.ipynb`](./preprocessingText.ipynb) | Text preprocessing & tokenization: regex-based custom tokenizer (`SimpleTokenizerV1`/`V2`), special tokens (`<\|unk\|>`, `<\|endoftext\|>`), byte-pair encoding with `tiktoken`, sliding-window input/target pair generation, and a PyTorch `Dataset`/`DataLoader` (`GPTDatasetV1`) for batching training data. Also covers token + positional embeddings. |
| [`attention.ipynb`](./attention.ipynb) | The attention mechanism, built up incrementally: simplified self-attention → trainable self-attention (`SelfAttention_v1`/`V2`) → causal (masked) attention with dropout → multi-head attention (`MultiHeadAttentionWrapper` and the more efficient single-matrix `MultiHeadAttention`). |
| [`architecture.ipynb`](./architecture.ipynb) | Assembling the full GPT-2 small architecture: model config (`GPT_CONFIG_124M`), custom `LayerNorm`, `GELU` activation, position-wise `FeedForward` network, a full `TransformerBlock` (attention + feed-forward + residual connections), and the final `GPTModel`. Includes parameter counting/memory footprint analysis, GPT-2 medium/large/XL config exercises, and a simple greedy `generate_text_simple` text-generation loop. |

The notebooks are meant to be worked through in order: **preprocessing → attention → architecture**. `architecture.ipynb` imports the `MultiHeadAttention` class directly from `attention.ipynb` using `ipynb.fs.defs`.

## Model config (GPT-2 small / 124M)

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # BPE vocabulary size (GPT-2 tokenizer)
    "context_length": 1024,  # max sequence length for positional embeddings
    "emb_dim": 768,          # embedding dimension
    "n_heads": 12,           # number of attention heads
    "n_layers": 12,          # number of transformer blocks
    "drop_rate": 0.1,        # dropout rate
    "qkv_bias": False        # query/key/value bias
}
```

## Data

[`the-verdict.txt`](./the-verdict.txt) — a short public-domain short story ("The Verdict" by Edith Wharton) used as toy training/demo text for tokenization and dataloader examples. It's downloaded automatically by the first cell of `preprocessingText.ipynb` if not already present.

## Getting started

### Requirements

- Python 3.9+
- [PyTorch](https://pytorch.org/)
- [tiktoken](https://github.com/openai/tiktoken)
- [ipynb](https://pypi.org/project/ipynb/) (used to import code from one notebook into another)
- `requests`, `numpy`, `matplotlib`
- Jupyter (to run the notebooks)

Install dependencies:

```bash
pip install torch tiktoken ipynb requests numpy matplotlib jupyter
```

### Running

```bash
git clone https://github.com/KunalAyush1/GPT-2-Small-from-scratch.git
cd GPT-2-Small-from-scratch
jupyter notebook
```

Open the notebooks in order (`preprocessingText.ipynb` → `attention.ipynb` → `architecture.ipynb`) and run the cells top to bottom.

### Quick sanity check

`architecture.ipynb` ends with a minimal greedy text-generation demo:

```python
model.eval()
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
print(tokenizer.decode(out.squeeze(0).tolist()))
```

Note: the model is **randomly initialized** (no pretrained weights are loaded), so generated text will be gibberish — this repo focuses on architecture and mechanics, not on producing a usable pretrained model.

## Status / Roadmap

- [x] Custom tokenizer + BPE tokenization
- [x] Dataset/DataLoader for input-target batches
- [x] Token + positional embeddings
- [x] Self-attention, causal attention, multi-head attention
- [x] LayerNorm, GELU, FeedForward
- [x] Full Transformer block & GPT model assembly
- [x] Parameter counting & GPT-2 medium/large/XL config exercises
- [x] Greedy text generation loop
- [ ] Pretraining loop / loss computation
- [ ] Loading official GPT-2 pretrained weights
- [ ] Fine-tuning examples

## Acknowledgements

This project follows the structure and exercises popularized by Sebastian Raschka's *Build a Large Language Model (From Scratch)* book/repo, implemented here independently for learning purposes.

## License

No license specified yet — add one (e.g. MIT) if you intend others to reuse this code.

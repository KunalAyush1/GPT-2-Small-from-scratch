<img width="800" height="436" alt="1ytpiPFSiw9FzLgC7J4JNQg" src="https://github.com/user-attachments/assets/1b1995d4-adf3-4c0a-961b-32eeeb912230" />

# GPT-2 Small From Scratch

A from-scratch, educational implementation of a GPT-2 (small / 124M) style language model in PyTorch , built step by step across a series of Jupyter notebooks. No Hugging Face `transformers` model classes are used; every component (tokenizer wrapper, attention, transformer block, GPT model) is implemented by hand.

## What's inside

| Notebook | What it covers |
|---|---|
| [`preprocessingText.ipynb`](./preprocessingText.ipynb) | Text preprocessing & tokenization: regex-based custom tokenizer (`SimpleTokenizerV1`/`V2`), special tokens (`<\|unk\|>`, `<\|endoftext\|>`), byte-pair encoding with `tiktoken`, sliding-window input/target pair generation, and a PyTorch `Dataset`/`DataLoader` (`GPTDatasetV1`) for batching training data. Also covers token + positional embeddings. |
| [`attention.ipynb`](./attention.ipynb) | The attention mechanism, built up incrementally: simplified self-attention → trainable self-attention (`SelfAttention_v1`/`V2`) → causal (masked) attention with dropout → multi-head attention (`MultiHeadAttentionWrapper` and the more efficient single-matrix `MultiHeadAttention`). |
| [`architecture.ipynb`](./architecture.ipynb) | Assembling the full GPT-2 small architecture: model config (`GPT_CONFIG_124M`), custom `LayerNorm`, `GELU` activation, position-wise `FeedForward` network, a full `TransformerBlock` (attention + feed-forward + residual connections), and the final `GPTModel`. Includes parameter counting/memory footprint analysis, GPT-2 medium/large/XL config exercises, and a simple greedy `generate_text_simple` text-generation loop. |
| [`pretraining.ipynb`](./pretraining.ipynb) | The full training pipeline: loading the from-scratch `GPTModel`, computing cross-entropy loss over batches (`calc_loss_batch`/`calc_loss_loader`), the training loop (`train_model_simple`) with periodic evaluation and sample text generation, loss curve plotting, and saving trained weights (`model.pth`). Trained end-to-end on **`the-verdict.txt`** as a small-scale proof of concept. |

The notebooks are meant to be worked through in order: **preprocessing → attention → architecture → pretraining**. `architecture.ipynb` imports the `MultiHeadAttention` class from `attention.ipynb`, and `pretraining.ipynb` imports `GPTModel`/`generate_text_simple` from `architecture.ipynb` and `create_dataloader_v1` from `preprocessingText.ipynb`, all via `ipynb.fs.defs`.

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

## Training

The model in `pretraining.ipynb` was trained on a single **NVIDIA T4 GPU via a [Lightning AI](https://lightning.ai/) Studio**. It's trained on the small `the-verdict.txt` corpus purely as a proof of concept for the training loop (loss computation, optimization, evaluation, and text generation during training) — not as a path to a strong general-purpose language model. Trained weights are saved to `model.pth`.

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
pip install -r requirements.txt
pip install torch   # not pinned in requirements.txt — install the build matching your CUDA/CPU setup
```

### Running

```bash
git clone https://github.com/KunalAyush1/GPT-2-Small-from-scratch.git
cd GPT-2-Small-from-scratch
jupyter notebook
```

Open the notebooks in order (`preprocessingText.ipynb` → `attention.ipynb` → `architecture.ipynb` → `pretraining.ipynb`) and run the cells top to bottom. GPU acceleration (e.g. a Lightning AI Studio with a T4) is recommended for `pretraining.ipynb`, though it will also run on CPU for the small demo dataset.

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

Note: `architecture.ipynb` on its own uses a **randomly initialized** model, so its generated text is gibberish — that notebook focuses purely on architecture and mechanics. `pretraining.ipynb` actually trains the model (see [Training](#training) below) and saves the resulting weights to `model.pth`, which can be reloaded with `model.load_state_dict(torch.load("model.pth"))`.

## Status / Roadmap

- [x] Custom tokenizer + BPE tokenization
- [x] Dataset/DataLoader for input-target batches
- [x] Token + positional embeddings
- [x] Self-attention, causal attention, multi-head attention
- [x] LayerNorm, GELU, FeedForward
- [x] Full Transformer block & GPT model assembly
- [x] Parameter counting & GPT-2 medium/large/XL config exercises
- [x] Greedy text generation loop
- [x] Pretraining loop / loss computation (trained on T4 GPU via Lightning AI)
- [ ] Loading official GPT-2 pretrained weights
- [ ] Fine-tuning examples

## Acknowledgements

This project follows the structure and exercises popularized by Sebastian Raschka's *Build a Large Language Model (From Scratch)* book/repo, implemented here independently for learning purposes.



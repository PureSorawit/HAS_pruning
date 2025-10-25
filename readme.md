# 🧩 HAS Pruning — Hessian-Aware Structured Pruning for Vision Transformers

This repository implements **Hessian-Aware Structured (HAS) Pruning** for Vision Transformers (ViT).  
It supports pruning **Feed-Forward Network (FFN) neurons**, **attention heads**, and **QKV dimensions**,  
while preserving layer consistency and exporting importance scores and masks for analysis.

---

## 1. Setup

Clone this repository and install dependencies:

```bash
git clone https://github.com/PureSorawit/HAS_pruning.git
cd HAS_pruning
```

## 2. Download the model

python download_model.py

this should create "checkpoints/vit_base_cifar100.pth"

## 3. run the pruning

Example CLI: 
python has_pruning.py \
  --ckpt ./checkpoints/vit_base_cifar100.pth \
  --model vit_base_patch16_224 \
  --mlp 0.30 --heads 0.15 --qkv 0.10 \
  --max-warmup-batches 64 --score-stride 1 \
  --amp \
  --save vit_base_pruned_has.pth \
  --export-prefix ./exports/vitb_cifar100_has



| Argument               | Type    | Default                | Required  | Description                                                              |
| ---------------------- | ------- | ---------------------- | --------- | ------------------------------------------------------------------------ |
| `--ckpt`               | `str`   | —                      | ✅ **Yes** | Path to pretrained checkpoint (`.pth`).                                  |
| `--model`              | `str`   | `vit_base_patch16_224` | ❌         | Model name from [timm](https://huggingface.co/docs/timm/).               |
| `--amp`                | flag    | `False`                | ❌         | Enable mixed precision (AMP) for faster evaluation.                      |
| `--max-warmup-batches` | `int`   | `64`                   | ❌         | Number of batches for importance scoring.                                |
| `--score-stride`       | `int`   | `1`                    | ❌         | Use every *n-th* batch for scoring to reduce cost.                       |
| `--mlp`                | `float` | `0.0`                  | ❌         | Fraction of FFN hidden units to prune globally (e.g. `0.30` = 30%).      |
| `--heads`              | `float` | `0.0`                  | ❌         | Fraction of attention heads to mask globally.                            |
| `--qkv`                | `float` | `0.0`                  | ❌         | Fraction of per-head QKV dimensions to mask per block.                   |
| `--chunk`              | `int`   | `1`                    | ❌         | FFN group size for structured pruning (e.g. `4` = prune in groups of 4). |
| `--save`               | `str`   | `""`                   | ❌         | Path to save the pruned model weights.                                   |
| `--export-prefix`      | `str`   | `""`                   | ❌         | Prefix for exported score/mask files (`.json` + `.npy`).                 |

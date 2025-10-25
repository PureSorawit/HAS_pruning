| Argument               | Type    | Default                | Required  | Description                                                                                          |
| ---------------------- | ------- | ---------------------- | --------- | ---------------------------------------------------------------------------------------------------- |
| `--ckpt`               | `str`   | —                      | ✅ **Yes** | Path to the pretrained checkpoint (`.pth` file).                                                     |
| `--model`              | `str`   | `vit_base_patch16_224` | ❌         | Model name from [timm](https://huggingface.co/docs/timm/).                                           |
| `--amp`                | flag    | `False`                | ❌         | Enable automatic mixed precision (AMP) for evaluation.                                               |
| `--max-warmup-batches` | `int`   | `64`                   | ❌         | Number of batches to sample for importance scoring (Taylor/Hessian-aware warm-up).                   |
| `--score-stride`       | `int`   | `1`                    | ❌         | Use every *n-th* batch during importance collection (for faster scoring).                            |
| `--mlp`                | `float` | `0.0`                  | ❌         | Fraction of **FFN hidden units** to prune globally (e.g., `0.30` removes 30%).                       |
| `--heads`              | `float` | `0.0`                  | ❌         | Fraction of **attention heads** to prune globally (e.g., `0.15` removes 15%).                        |
| `--qkv`                | `float` | `0.0`                  | ❌         | Fraction of **per-head QKV dimensions** to prune *per block* (e.g., `0.10` removes 10%).             |
| `--chunk`              | `int`   | `1`                    | ❌         | FFN group size for structured pruning (e.g., `4` removes neurons in groups of 4).                    |
| `--save`               | `str`   | `""`                   | ❌         | Output path to save the pruned model’s `state_dict`.                                                 |
| `--export-prefix`      | `str`   | `""`                   | ❌         | Prefix for exporting importance **scores** (`_scores.json/.npy`) and **masks** (`_masks.json/.npy`). |

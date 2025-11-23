import argparse, time, json, os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import timm

# -------------------------------------------------
# Utils
# -------------------------------------------------
def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

@torch.no_grad()
def evaluate(model, loader, device, use_amp=False):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
        pred = logits.argmax(1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)
        loss_sum += loss.item() * targets.size(0)
    return loss_sum/total, correct/total

def build_loaders(img_size=224, batch_size=128, workers=2):
    norm = transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
    tf_train = transforms.Compose([transforms.Resize((img_size,img_size)),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), norm])
    tf_val   = transforms.Compose([transforms.Resize((img_size,img_size)),
                                   transforms.ToTensor(), norm])
    train = datasets.CIFAR100("./data", train=True, transform=tf_train, download=True)
    val   = datasets.CIFAR100("./data", train=False, transform=tf_val, download=True)
    tr = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True)
    va = torch.utils.data.DataLoader(val,   batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return tr, va

def total_mlp_units(model: nn.Module) -> int:
    return sum(b.mlp.fc1.out_features for b in model.blocks)

def total_heads(model: nn.Module) -> int:
    return sum(b.attn.num_heads for b in model.blocks)

# -------------------------------------------------
# Importance (first-order Hessian-aware/Taylor: sum (w*grad) over group, squared)
# -------------------------------------------------
def collect_importance_taylor(
    model: nn.Module,
    loader,
    device,
    max_warmup_batches: int = 200,
    score_stride: int = 16,
) -> Tuple[Dict[Tuple[int,int], float],
           Dict[Tuple[int,int], float],
           Dict[Tuple[int,int], float]]:
    """
    Return dicts:
      ffn_imp:  {(block, unit) -> score}
      head_imp: {(block, head) -> score}
      qkv_dim_imp: {(block, dimpos) -> score}
    Score per group g: (sum_{w in g} (w * grad_w))^2, accumulated over sampled batches.
    AMP is disabled here to keep grads numerically stable/consistent.
    """
    model.eval()
    ffn_imp: Dict[Tuple[int,int], float] = {}
    head_imp: Dict[Tuple[int,int], float] = {}
    qkv_dim_imp: Dict[Tuple[int,int], float] = {}

    scored_batches = 0
    for b_idx, (images, targets) in enumerate(loader):
        if scored_batches >= max_warmup_batches:
            break
        score_this = (b_idx % score_stride) == 0

        images, targets = images.to(device), targets.to(device)
        # Forward
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        with torch.amp.autocast("cuda", enabled=False):
            logits = model(images)
            loss = F.cross_entropy(logits, targets)

        # Backward for grads
        loss.backward()

        if score_this:
            # iterate blocks and aggregate per-structure Taylor scores
            for l, blk in enumerate(model.blocks):
                # ----- FFN per hidden unit: fc1 row i + fc2 column i
                fc1, fc2 = blk.mlp.fc1, blk.mlp.fc2
                g1 = fc1.weight.grad       # [hidden, in]
                g2 = fc2.weight.grad       # [out, hidden]
                if (g1 is not None) and (g2 is not None):
                    s1 = (fc1.weight * g1).sum(dim=1)      # [hidden]
                    s2 = (fc2.weight * g2).sum(dim=0)      # [hidden]
                    s = s1 + s2
                    # square as in Eq.(6)
                    s = s * s
                    for i in range(s.numel()):
                        key = (l, int(i))
                        ffn_imp[key] = ffn_imp.get(key, 0.0) + float(s[i].item())

                # ----- Attention heads: group = {qkv slice of head h} U {proj columns of head h}
                attn = blk.attn
                nh   = attn.num_heads
                qkv, proj = attn.qkv, attn.proj
                gqkv = qkv.weight.grad
                gprj = proj.weight.grad
                if (gqkv is not None) and (gprj is not None) and nh > 0:
                    E_in = qkv.weight.shape[1]
                    hd = proj.in_features // nh
                    try:
                        Wqkv = qkv.weight.view(3, nh, hd, E_in)
                        Gqkv = gqkv.view(3, nh, hd, E_in)
                    except Exception:
                        Wqkv = None
                    if Wqkv is not None:
                        for h in range(nh):
                            # qkv slice for head h
                            t_qkv = (Wqkv[:, h] * Gqkv[:, h]).sum()
                            # proj columns for head h
                            cols = slice(h*hd, (h+1)*hd)
                            t_prj = (proj.weight[:, cols] * gprj[:, cols]).sum()
                            t = t_qkv + t_prj
                            score = float((t * t).item())
                            key = (l, int(h))
                            head_imp[key] = head_imp.get(key, 0.0) + score

                        # per-dimension inside head (dimpos d, shared across heads)
                        # combine qkv at dim d across all heads + corresponding proj columns
                        for d in range(hd):
                            t_qkv_d = (Wqkv[:, :, d, :] * Gqkv[:, :, d, :]).sum()
                            # proj: collect column d in each head
                            acc = 0.0
                            for h in range(nh):
                                c = h*hd + d
                                acc += float(((proj.weight[:, c] * gprj[:, c]).sum()).item())
                            t_dim = float(t_qkv_d.item()) + acc
                            score = t_dim * t_dim
                            key = (l, int(d))
                            qkv_dim_imp[key] = qkv_dim_imp.get(key, 0.0) + score

            scored_batches += 1

        # clear grads for next step
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None

    return ffn_imp, head_imp, qkv_dim_imp

# -------------------------------------------------
# Pruning helpers
# -------------------------------------------------
def _new_linear_like(old: nn.Linear, in_f: int, out_f: int) -> nn.Linear:
    dev, dt = old.weight.device, old.weight.dtype
    new = nn.Linear(in_f, out_f, bias=(old.bias is not None))
    return new.to(device=dev, dtype=dt)

@torch.no_grad()
def _prune_linear_out(old: nn.Linear, keep_idx: List[int]) -> nn.Linear:
    W = old.weight[keep_idx, :].clone()
    B = old.bias[keep_idx].clone() if old.bias is not None else None
    new = _new_linear_like(old, old.in_features, len(keep_idx))
    new.weight.copy_(W)
    if B is not None: new.bias.copy_(B)
    return new

@torch.no_grad()
def _prune_linear_in(old: nn.Linear, keep_idx: List[int]) -> nn.Linear:
    W = old.weight[:, keep_idx].clone()
    B = old.bias.clone() if old.bias is not None else None
    new = _new_linear_like(old, len(keep_idx), old.out_features)
    new.weight.copy_(W)
    if B is not None: new.bias.copy_(B)
    return new

def _unit_count_per_block(model: nn.Module) -> List[int]:
    return [blk.mlp.fc1.out_features for blk in model.blocks]

def pick_ffn_units(ffn_imp: Dict[Tuple[int,int], float], model: nn.Module,
                   target_frac: float, chunk: int = 1) -> Dict[int, List[int]]:
    Hs = _unit_count_per_block(model)
    total = sum(Hs)
    K = int(target_frac * total + 1e-6)
    if K <= 0: return {}
    items = []
    for l, H in enumerate(Hs):
        for i in range(H):
            items.append((ffn_imp.get((l,i), 0.0), l, i))
    items.sort(key=lambda t: t[0])  # lowest first

    chosen, marked = {}, set()
    if chunk <= 1:
        for _, l, i in items[:K]:
            chosen.setdefault(l, []).append(i)
    else:
        left = K
        for _, l, i in items:
            if left <= 0: break
            base = (i // chunk) * chunk
            grp = list(range(base, min(base+chunk, Hs[l])))
            if any((l, j) in marked for j in grp): continue
            for j in grp:
                chosen.setdefault(l, []).append(j)
                marked.add((l, j))
            left -= len(grp)

    for l in list(chosen.keys()):
        chosen[l] = [i for i in sorted(set(chosen[l])) if i < Hs[l]]
    return chosen

def apply_ffn_pruning(model: nn.Module, to_remove: Dict[int, List[int]]):
    for l, rm in to_remove.items():
        blk = model.blocks[l]
        H = blk.mlp.fc1.out_features
        keep = sorted(set(range(H)) - set(rm))
        if len(keep) == H: continue
        blk.mlp.fc1 = _prune_linear_out(blk.mlp.fc1, keep)
        blk.mlp.fc2 = _prune_linear_in(blk.mlp.fc2, keep)
        if hasattr(blk.mlp, "hidden_features"):
            blk.mlp.hidden_features = len(keep)

def pick_heads(head_imp: Dict[Tuple[int,int], float], model: nn.Module,
               target_frac: float) -> Dict[int, List[int]]:
    T = total_heads(model)
    K = int(target_frac * T + 1e-6)
    if K <= 0: return {}
    items = []
    for l, blk in enumerate(model.blocks):
        for h in range(blk.attn.num_heads):
            items.append((head_imp.get((l,h), 0.0), l, h))
    items.sort(key=lambda t: t[0])
    chosen, remain = {}, {l: model.blocks[l].attn.num_heads for l in range(len(model.blocks))}
    for _, l, h in items:
        if K <= 0: break
        if remain[l] <= 1: continue
        chosen.setdefault(l, []).append(h)
        remain[l] -= 1
        K -= 1
    return chosen

@torch.no_grad()
def _new_attn_linear_like(old: nn.Linear, in_f: int, out_f: int) -> nn.Linear:
    dev, dt = old.weight.device, old.weight.dtype
    new = nn.Linear(in_f, out_f, bias=(old.bias is not None))
    return new.to(device=dev, dtype=dt)

@torch.no_grad()
def apply_head_pruning(model: nn.Module, to_remove: Dict[int, List[int]]):
    """
    Mask-only attention head pruning compatible with timm Attention that
    reshapes to embed_dim. We keep shapes/metadata unchanged and just zero:
      • all Q/K/V slices for the heads to prune
      • the corresponding column blocks in the output projection
    """
    import torch
    for l, rm_heads in to_remove.items():
        if not rm_heads:
            continue
        attn = model.blocks[l].attn
        nh = attn.num_heads
        # current per-head dim from proj input width
        hd = attn.proj.in_features // nh

        # sanitize indices
        rm = sorted({h for h in rm_heads if 0 <= h < nh})
        if not rm:
            continue

        # ---- Zero out Q/K/V rows for those heads
        # qkv.weight: [3*nh*hd, in_features] -> [3, nh, hd, in_features]
        Wqkv = attn.qkv.weight          # (no shape change)
        Bqkv = attn.qkv.bias if attn.qkv.bias is not None else None
        E_in = Wqkv.shape[1]
        try:
            Wqkv_r = Wqkv.view(3, nh, hd, E_in)
        except RuntimeError:
            # unexpected layout; skip this block safely
            continue

        # zero entire head slices across all dims for Q/K/V
        Wqkv_r[:, rm, :, :].zero_()
        if Bqkv is not None:
            Bqkv_r = Bqkv.view(3, nh, hd)
            Bqkv_r[:, rm, :].zero_()

        # ---- Zero corresponding column blocks in proj
        # proj.weight: [embed_dim, nh*hd]; columns for head h are [h*hd : (h+1)*hd]
        Wproj = attn.proj.weight
        for h in rm:
            cols = slice(h * hd, (h + 1) * hd)
            Wproj[:, cols].zero_()


def pick_qkv_dims(qkv_dim_imp: Dict[Tuple[int,int], float], model: nn.Module,
                  target_frac: float) -> Dict[int, List[int]]:
    if target_frac <= 0.0: return {}
    out = {}
    for l, blk in enumerate(model.blocks):
        nh = blk.attn.num_heads
        hd = blk.attn.proj.in_features // nh
        if hd <= 1: continue
        k = int(target_frac * hd + 1e-6)
        if k <= 0: continue
        items = [(qkv_dim_imp.get((l,d), 0.0), d) for d in range(hd)]
        items.sort(key=lambda t: t[0])
        out[l] = sorted({d for _, d in items[:k]})
    return out

@torch.no_grad()
def apply_qkv_dim_pruning(model: nn.Module, to_remove: Dict[int, List[int]]):
    """
    Mask-only QKV per-head-dim pruning to stay compatible with timm Attention
    that reshapes to a fixed embed_dim. We do NOT change num_heads/head_dim or
    any Linear shapes. Instead, we zero out:
      • the selected per-head dims in Q, K, V (for all heads), and
      • the corresponding columns in the output projection.
    """
    for l, rm_dims in to_remove.items():
        if not rm_dims:
            continue

        attn = model.blocks[l].attn
        nh = attn.num_heads
        hd = attn.proj.in_features // nh  # current per-head dim (unchanged structurally)

        rm = sorted({d for d in rm_dims if 0 <= d < hd})
        if not rm:
            continue

        # ----- zero Q/K/V slices at those dims (all heads)
        Wqkv = attn.qkv.weight         # [3*nh*hd, in_features]
        Bqkv = attn.qkv.bias if attn.qkv.bias is not None else None
        E_in = Wqkv.shape[1]

        try:
            Wqkv_r = Wqkv.view(3, nh, hd, E_in)  # [3, nh, hd, in_features]
        except RuntimeError:
            # in case ran into unexpected shape; skip this block
            continue

        Wqkv_r[:, :, rm, :].zero_()
        if Bqkv is not None:
            Bqkv_r = Bqkv.view(3, nh, hd)
            Bqkv_r[:, :, rm].zero_()

        # proj.weight: [embed_dim, nh*hd]; column for (head h, dim d) is base=h*hd + d
        Wproj = attn.proj.weight
        for h in range(nh):
            base = h * hd
            for d in rm:
                Wproj[:, base + d].zero_()

# -------------------------------------------------
# Exports
# -------------------------------------------------
def export_scores_and_masks(prefix: str,
                            ffn_imp, head_imp, qkv_dim_imp,
                            ffn_mask, head_mask, qkv_dim_mask):
    os.makedirs(os.path.dirname(prefix) if os.path.dirname(prefix) else ".", exist_ok=True)

    scores_json = {
        "ffn": {f"{l}:{i}": float(s) for (l,i), s in ffn_imp.items()},
        "heads": {f"{l}:{h}": float(s) for (l,h), s in head_imp.items()},
        "qkv_dim": {f"{l}:{d}": float(s) for (l,d), s in qkv_dim_imp.items()},
    }
    with open(prefix + "_scores.json", "w") as f:
        json.dump(scores_json, f, indent=2)

    # pack masks as lists per block
    with open(prefix + "_masks.json", "w") as f:
        json.dump({
            "ffn": {str(l): m for l, m in ffn_mask.items()},
            "heads": {str(l): m for l, m in head_mask.items()},
            "qkv_dim": {str(l): m for l, m in qkv_dim_mask.items()},
        }, f, indent=2)

    np.save(prefix + "_ffn_scores.npy",
            np.array([[l, i, s] for (l,i), s in ffn_imp.items()], dtype=np.float64))
    np.save(prefix + "_head_scores.npy",
            np.array([[l, h, s] for (l,h), s in head_imp.items()], dtype=np.float64))
    np.save(prefix + "_qkvdim_scores.npy",
            np.array([[l, d, s] for (l,d), s in qkv_dim_imp.items()], dtype=np.float64))

    # masks as object arrays of per-block vectors
    np.save(prefix + "_ffn_masks.npy", np.array([ffn_mask[k] for k in sorted(ffn_mask.keys())], dtype=object))
    np.save(prefix + "_head_masks.npy", np.array([head_mask[k] for k in sorted(head_mask.keys())], dtype=object))
    np.save(prefix + "_qkvdim_masks.npy", np.array([qkv_dim_mask[k] for k in sorted(qkv_dim_mask.keys())], dtype=object))

# -------------------------------------------------
# Main
# -------------------------------------------------
def run(args):
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and torch.cuda.is_available()
    print(f"CUDA: {torch.cuda.is_available()} | AMP: {use_amp}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = timm.create_model(args.model, pretrained=False, num_classes=100).to(device)

    sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] loaded (missing={len(missing)}, unexpected={len(unexpected)}).")
    print(f"Params: {count_params(model)/1e6:.2f}M | "
          f"FFN units={total_mlp_units(model)} | Heads={total_heads(model)}")

    train_loader, val_loader = build_loaders(img_size=224, batch_size=128, workers=2)

    t0 = time.time()
    b_loss, b_acc = evaluate(model, val_loader, device, use_amp)
    print(f"[Baseline] Val loss {b_loss:.4f}  Acc {b_acc*100:.2f}%  (t={time.time()-t0:.1f}s)")

    print(f"[Warmup] Collecting importance (max_batches={args.max_warmup_batches}, "
          f"stride={args.score_stride}) ...")
    t1 = time.time()
    ffn_imp, head_imp, qkv_dim_imp = collect_importance_taylor(
        model, train_loader, device,
        max_warmup_batches=args.max_warmup_batches,
        score_stride=args.score_stride,
    )
    print(f"[Warmup] Done in {time.time()-t1:.1f}s | scored: "
          f"FFN={len(ffn_imp)} Heads={len(head_imp)} QKV-dims={len(qkv_dim_imp)}")

    to_remove_units, to_remove_heads, to_remove_dims = {}, {}, {}

    if args.mlp > 0:
        total_before = total_mlp_units(model)
        to_remove_units = pick_ffn_units(ffn_imp, model, args.mlp, chunk=args.chunk)
        removed = sum(len(v) for v in to_remove_units.values())
        print(f"[FFN] Planned removal: {removed}/{total_before} "
              f"(~{100.0*removed/total_before:.1f}%).")

    if args.heads > 0:
        total_before = total_heads(model)
        to_remove_heads = pick_heads(head_imp, model, args.heads)
        removed = sum(len(v) for v in to_remove_heads.values())
        print(f"[HEADS] Planned removal: {removed}/{total_before} "
              f"(~{100.0*removed/total_before:.1f}%).")

    if args.qkv > 0:
        to_remove_dims = pick_qkv_dims(qkv_dim_imp, model, args.qkv)
        per_block_rm = {l: len(v) for l, v in to_remove_dims.items()}
        print(f"[QKV] Planned per-head dim removals per block: {per_block_rm}")

    ffn_mask = {}
    for l, blk in enumerate(model.blocks):
        H_orig = blk.mlp.fc1.out_features
        rm = set(to_remove_units.get(l, []))
        keep = sorted(set(range(H_orig)) - rm)
        m = np.zeros(H_orig, dtype=int)
        m[keep] = 1
        ffn_mask[l] = m.tolist()

    head_mask = {}
    for l, blk in enumerate(model.blocks):
        nh_orig = blk.attn.num_heads
        rm = set(to_remove_heads.get(l, []))
        keep = sorted(set(range(nh_orig)) - rm)
        m = np.zeros(nh_orig, dtype=int)
        m[keep] = 1
        head_mask[l] = m.tolist()

    qkv_dim_mask = {}
    for l, blk in enumerate(model.blocks):
        nh = blk.attn.num_heads
        hd_orig = blk.attn.proj.in_features // nh
        rm = set(to_remove_dims.get(l, []))
        keep = sorted(set(range(hd_orig)) - rm)
        m = np.zeros(hd_orig, dtype=int)
        m[keep] = 1
        qkv_dim_mask[l] = m.tolist()

    if args.export_prefix:
        export_scores_and_masks(
            args.export_prefix,
            ffn_imp, head_imp, qkv_dim_imp,
            ffn_mask, head_mask, qkv_dim_mask
        )
        print(f"[Export] wrote {args.export_prefix}_scores.json/.npy and "
              f"{args.export_prefix}_masks.json/.npy (PRE-PRUNE)")

    if args.mlp > 0:
        p0 = count_params(model)
        apply_ffn_pruning(model, to_remove_units)
        p1 = count_params(model)
        print(f"[FFN] Applied pruning. Params {p0/1e6:.2f}M -> {p1/1e6:.2f}M")

    if args.heads > 0:
        total_before = total_heads(model) + sum(len(v) for v in to_remove_heads.values())
        apply_head_pruning(model, to_remove_heads)
        print(f"[HEADS] Applied pruning. Heads per block now: "
              f"{[b.attn.num_heads for b in model.blocks]}")

    if args.qkv > 0:
        before = []
        for blk in model.blocks:
            nh = blk.attn.num_heads
            before.append(blk.attn.proj.in_features // nh)
        apply_qkv_dim_pruning(model, to_remove_dims)
        after = []
        for blk in model.blocks:
            nh = blk.attn.num_heads
            after.append(blk.attn.proj.in_features // nh)
        print(f"[QKV] per-head dims (before -> after per block): "
              f"{list(zip(before, after))}")

    t2 = time.time()
    a_loss, a_acc = evaluate(model, val_loader, device, use_amp)
    print(f"[After]  Val loss {a_loss:.4f}  Acc {a_acc*100:.2f}%  "
          f"(Δt={time.time()-t2:.1f}s)")

    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"[Save] {args.save}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--model", type=str, default="vit_base_patch16_224")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--max-warmup-batches", type=int, default=64)
    p.add_argument("--score-stride", type=int, default=1)
    p.add_argument("--mlp", type=float, default=0.0, help="fraction of FFN hidden units to prune globally")
    p.add_argument("--heads", type=float, default=0.0, help="fraction of attention heads to prune globally")
    p.add_argument("--qkv", type=float, default=0.0, help="fraction of per-head dims to prune per block")
    p.add_argument("--chunk", type=int, default=1, help="FFN group size (e.g., 4) for structured unit pruning")
    p.add_argument("--save", type=str, default="", help="path to save pruned model state_dict")
    p.add_argument("--export-prefix", type=str, default="", help="prefix path for exporting scores/masks")
    args = p.parse_args()
    run(args)

if __name__ == "__main__":
    main()

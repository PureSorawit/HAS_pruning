import os, urllib.request, torch, timm

# Choose one of: "old-ti", "old-b", "new-ti", "new-b"
CHECKPOINT = "old-ti"
NUM_CLASSES = 100

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ------------------------------------------------------------
# HuggingFace fine-tuned checkpoints (CIFAR-100)
# ------------------------------------------------------------
OLD_URL_B  = (
    "https://huggingface.co/edadaltocg/"
    "vit_base_patch16_224_in21k_ft_cifar100/resolve/main/pytorch_model.bin"
)
OLD_URL_TI = (
    "https://huggingface.co/edadaltocg/"
    "vit_tiny_patch16_224_in21k_ft_cifar100/resolve/main/pytorch_model.bin"
)

# ------------------------------------------------------------
# AugReg from Google’s ViT collection
# ------------------------------------------------------------
INDEX_URL  = "https://storage.googleapis.com/vit_models/augreg/index.csv"
INDEX_PATH = "data/index.csv"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _strip_prefixes(sd, prefixes=("model.", "module.", "net.")):
    out = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
                break
        out[nk] = v
    return out

def _load_state_any(source):
    if source.startswith(("http://", "https://")):
        return torch.hub.load_state_dict_from_url(source, map_location="cpu")
    return torch.load(source, map_location="cpu")

def _save(model, path):
    torch.save(model.state_dict(), path)
    print(f"[✓] Saved -> {path}")

def _ensure_index():
    if not os.path.isfile(INDEX_PATH):
        urllib.request.urlretrieve(INDEX_URL, INDEX_PATH)


# ------------------------------------------------------------
# Main: 4 explicit specs
# ------------------------------------------------------------
if CHECKPOINT == "old-ti":
    # NOTE: this HF checkpoint is actually 768-dim (ViT-Base spec),
    # so we *explicitly* use vit_base_patch16_224 here.
    MODEL = "vit_base_patch16_224"
    SAVE  = "checkpoints/vitb_old_ti_cifar100.pth"

    sd = _strip_prefixes(_load_state_any(OLD_URL_TI))
    m = timm.create_model(MODEL, pretrained=False, num_classes=NUM_CLASSES)
    m.load_state_dict(sd, strict=False)
    _save(m, SAVE)

elif CHECKPOINT == "old-b":
    # HuggingFace base fine-tuned model, 768-dim ViT-Base spec
    MODEL = "vit_base_patch16_224"
    SAVE  = "checkpoints/vitb_old_cifar100.pth"

    sd = _strip_prefixes(_load_state_any(OLD_URL_B))
    m = timm.create_model(MODEL, pretrained=False, num_classes=NUM_CLASSES)
    m.load_state_dict(sd, strict=False)
    _save(m, SAVE)

elif CHECKPOINT == "new-ti":
    # AugReg ViT-Tiny (192-dim) in21k, then finetuned to CIFAR-100
    _ensure_index()
    MODEL = "vit_tiny_patch16_224.augreg_in21k"
    SAVE  = "checkpoints/vitt_new_cifar100.pth"

    m = timm.create_model(MODEL, pretrained=True, num_classes=NUM_CLASSES)
    _save(m, SAVE)

elif CHECKPOINT == "new-b":
    # AugReg ViT-Base (768-dim) in21k, then finetuned to CIFAR-100
    _ensure_index()
    MODEL = "vit_base_patch16_224.augreg_in21k"
    SAVE  = "checkpoints/vitb_new_cifar100.pth"

    m = timm.create_model(MODEL, pretrained=True, num_classes=NUM_CLASSES)
    _save(m, SAVE)

else:
    raise ValueError("Choose one of: old-ti, old-b, new-ti, new-b")

print("Done.")

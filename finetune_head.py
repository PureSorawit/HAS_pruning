import argparse, torch, timm, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_tiny_patch16_224")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--data", default="./data")        # CIFAR-100 root
    ap.add_argument("--save", default="./checkpoints/vitt_tiny_cifar100_ft.pth")
    args = ap.parse_args()

    tfm_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)),
    ])
    tfm_val = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)),
    ])

    tr = datasets.CIFAR100(args.data, train=True, download=True, transform=tfm_train)
    va = datasets.CIFAR100(args.data, train=False, download=True, transform=tfm_val)
    trl = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
    val = DataLoader(va, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(args.model, pretrained=True, num_classes=100).to(device)

    for n,p in model.named_parameters():
        p.requires_grad = ("head." in n)  # head-only finetune

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss()

    def eval_acc():
        model.eval(); corr=tot=0
        with torch.no_grad():
            for x,y in val:
                x,y = x.to(device), y.to(device)
                logits = model(x)
                corr += (logits.argmax(1)==y).sum().item()
                tot += y.numel()
        return corr/tot*100

    for e in range(1, args.epochs+1):
        model.train()
        for x,y in trl:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
        sched.step()
        print(f"epoch {e}/{args.epochs} | val acc: {eval_acc():.2f}%")

    torch.save(model.state_dict(), args.save)
    print(f"[save] {args.save}")

if __name__ == "__main__":
    main()
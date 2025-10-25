import os
import timm
import torch

def get_vit_base_cifar100(save_path: str = "./checkpoints/vit_base_cifar100.pth"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create the model
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=100)
    torch.save(model.state_dict(), save_path)

    print(f"Pretrained ViT-B/16 saved at: {save_path}")
    return save_path


if __name__ == "__main__":
    get_vit_base_cifar100()

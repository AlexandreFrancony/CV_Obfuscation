import torch
from torchvision.models import resnet18, ResNet18_Weights
from typing import List, Tuple


def load_pretrained_resnet18(device: str | None = None) -> Tuple[torch.nn.Module, torch.nn.Module, str, List[str]]:
    """
    Load a pretrained ResNet-18 model on ImageNet and its associated preprocessing transforms.
    Returns:
        model: pretrained ResNet-18 in eval mode on the chosen device.
        preprocess: torchvision transform pipeline matching the weights.
        device: device string ("cpu" or "cuda").
        imagenet_categories: list of 1000 ImageNet class names.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    model.to(device)

    preprocess = weights.transforms()
    imagenet_categories: List[str] = weights.meta["categories"]

    return model, preprocess, device, imagenet_categories

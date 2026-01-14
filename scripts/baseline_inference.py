import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Imagenette

from src.models.model_loader import load_pretrained_resnet18


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def get_dataloader(preprocess, batch_size: int = 32) -> DataLoader:
    data_root = Path("data") / "imagenette"
    dataset = Imagenette(
        root=str(data_root),
        split="val",      # "train" or "val"
        download=True,
        transform=preprocess,
    )
    logging.info(f"Loaded Imagenette val split with {len(dataset)} samples.")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def evaluate_baseline():
    setup_logging()
    logging.info("Loading pretrained ResNet-18...")

    model, preprocess, device = load_pretrained_resnet18()
    logging.info(f"Model loaded on device: {device}")

    dataloader = get_dataloader(preprocess)

    correct = 0
    total = 0

    logging.info("Starting baseline evaluation on Imagenette (val)...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (i + 1) % 10 == 0:
                batch_acc = (preds == labels).float().mean().item()
                logging.info(
                    f"Batch {i+1:03d} | Batch accuracy: {batch_acc:.3f}"
                )

    accuracy = correct / total if total > 0 else 0.0
    logging.info(f"Baseline top-1 accuracy on Imagenette (val): {accuracy:.3f}")


if __name__ == "__main__":
    evaluate_baseline()

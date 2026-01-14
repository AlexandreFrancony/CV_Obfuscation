import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Imagenette

from src.models.model_loader import load_pretrained_resnet18
from src.attacks.fgsm import fgsm_attack


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def get_dataloader(preprocess, batch_size: int = 32) -> DataLoader:
    data_root = Path("data") / "imagenette"
    dataset = Imagenette(
        root=str(data_root),
        split="val",
        download=False,  # déjà téléchargé par le baseline
        transform=preprocess,
    )
    logging.info(f"Loaded Imagenette val split with {len(dataset)} samples.")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def evaluate_fgsm(epsilon: float = 0.01):
    setup_logging()
    logging.info("Loading pretrained ResNet-18...")

    model, preprocess, device = load_pretrained_resnet18()
    logging.info(f"Model loaded on device: {device}")

    dataloader = get_dataloader(preprocess)

    clean_correct = 0
    adv_correct = 0
    total = 0

    logging.info(f"Starting FGSM evaluation with epsilon={epsilon}...")
    model.eval()

    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Clean predictions
        with torch.no_grad():
            outputs_clean = model(images)
            _, preds_clean = torch.max(outputs_clean, dim=1)
        clean_correct += (preds_clean == labels).sum().item()

        # Adversarial images
        images_adv = fgsm_attack(model, images, labels, epsilon=epsilon)

        # Predictions on adversarial images
        with torch.no_grad():
            outputs_adv = model(images_adv)
            _, preds_adv = torch.max(outputs_adv, dim=1)
        adv_correct += (preds_adv == labels).sum().item()

        total += labels.size(0)

        if (i + 1) % 10 == 0:
            clean_acc_batch = (preds_clean == labels).float().mean().item()
            adv_acc_batch = (preds_adv == labels).float().mean().item()
            logging.info(
                f"Batch {i+1:03d} | "
                f"Clean acc: {clean_acc_batch:.3f} | "
                f"Adv acc: {adv_acc_batch:.3f}"
            )

    clean_acc = clean_correct / total if total > 0 else 0.0
    adv_acc = adv_correct / total if total > 0 else 0.0

    logging.info(f"Final clean accuracy: {clean_acc:.3f}")
    logging.info(f"Final adversarial accuracy (FGSM, eps={epsilon}): {adv_acc:.3f}")


if __name__ == "__main__":
    # Tu pourras ajuster epsilon plus tard
    evaluate_fgsm(epsilon=0.01)

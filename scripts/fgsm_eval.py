import logging
from pathlib import Path
from typing import Dict, List, Sequence, Union, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import Imagenette
import torchvision.transforms as T

from src.metrics.image_quality import compute_psnr_batch, compute_ssim_batch
from src.models.model_loader import load_pretrained_resnet18
from src.attacks.fgsm import fgsm_attack


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def get_base_transform() -> T.Compose:
    """
    Base transform: resize to 224x224 and convert to tensor in [0, 1],
    without normalization.
    """
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),  # -> [0,1]
    ])

def denormalize_imagenet(t: torch.Tensor) -> torch.Tensor:
    """
    Undo ImageNet normalization, returning images back to approx [0, 1].
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(1, 3, 1, 1)
    return (t * std + mean).clamp(0.0, 1.0)


def get_dataset_and_dataloader(
    base_transform: T.Compose,
    batch_size: int = 32,
) -> Tuple[Imagenette, DataLoader]:
    """
    Load the Imagenette validation split using a base transform that
    keeps images in [0,1] (no normalization).
    """
    data_root = Path("data") / "imagenette"
    dataset = Imagenette(
        root=str(data_root),
        split="val",
        download=False,
        transform=base_transform,
    )
    logging.info(f"Loaded Imagenette val split with {len(dataset)} samples.")
    logging.info(f"Imagenette classes (dataset.classes): {dataset.classes}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataset, dataloader


def _normalize_imagenette_class_name(
    cls_entry: Union[str, Sequence[str]]
) -> str:
    if isinstance(cls_entry, str):
        return cls_entry
    if isinstance(cls_entry, (list, tuple)) and len(cls_entry) > 0:
        return cls_entry[0]
    raise ValueError(f"Unsupported class entry format: {cls_entry}")


def build_imagenette_to_imagenet_mapping(
    imagenette_dataset: Imagenette,
    imagenet_categories: List[str],
) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    imagenette_classes = imagenette_dataset.classes

    logging.info("Building Imagenette -> ImageNet mapping for FGSM eval...")
    logging.info(f"Raw Imagenette classes: {imagenette_classes}")

    for imgnt_idx, cls_entry in enumerate(imagenette_classes):
        canonical_name = _normalize_imagenette_class_name(cls_entry)

        try:
            imnet_idx = imagenet_categories.index(canonical_name)
        except ValueError as e:
            raise ValueError(
                f"Could not find Imagenette class '{canonical_name}' "
                f"(from {cls_entry}) in ImageNet categories list."
            ) from e

        mapping[imgnt_idx] = imnet_idx
        logging.info(
            f"Mapping Imagenette idx {imgnt_idx} ('{canonical_name}') "
            f"-> ImageNet idx {imnet_idx} ('{imagenet_categories[imnet_idx]}')"
        )

    return mapping


def evaluate_fgsm(epsilon: float = 0.01, batch_size: int = 32) -> None:
    """
    Evaluate the impact of an untargeted FGSM attack on ResNet-18
    restricted to the 10 Imagenette classes, and compute PSNR / SSIM
    between clean and adversarial images.
    """
    setup_logging()
    logging.info("Loading pretrained ResNet-18...")

    model, preprocess, device, imagenet_categories = load_pretrained_resnet18()
    logging.info(f"Model loaded on device: {device}")

    base_transform = get_base_transform()
    dataset, dataloader = get_dataset_and_dataloader(base_transform, batch_size=batch_size)

    # Build mapping Imagenette (0-9) -> ImageNet (0-999)
    mapping = build_imagenette_to_imagenet_mapping(dataset, imagenet_categories)

    # Indices des 10 classes ImageNet correspondantes
    imnet_indices = torch.tensor(
        [mapping[i] for i in range(len(mapping))],
        device=device,
        dtype=torch.long,
    )
    logging.info(f"ImageNet indices used for restriction: {imnet_indices.tolist()}")
    logging.info(f"Running FGSM with epsilon={epsilon}...")

    clean_correct = 0
    adv_correct = 0
    total = 0

    loss_fn = nn.CrossEntropyLoss()
    psnr_values: List[float] = []
    ssim_values: List[float] = []

    model.eval()

    # Forward restreint aux 10 classes (B, 10), aligné avec labels 0-9
    def forward_restricted(x: torch.Tensor) -> torch.Tensor:
        outputs = model(x)  # (B, 1000)
        logits_subset = outputs.index_select(dim=1, index=imnet_indices)  # (B, 10)
        return logits_subset

    for i, (images, labels) in enumerate(dataloader):
        # images: [0,1] from base_transform
        images = images.to(device)
        labels = labels.to(device)

        # Normalisation ImageNet via preprocess (applique resize + ToTensor + normalize).
        # Les images sont déjà resized + ToTensor, mais preprocess peut les accepter comme Tensor.
        images_norm = preprocess(images)  # type: ignore[arg-type]

        # --- Clean predictions ---
        with torch.no_grad():
            logits_clean_subset = forward_restricted(images_norm)  # (B, 10)
            _, preds_clean = torch.max(logits_clean_subset, dim=1)

        clean_correct += (preds_clean == labels).sum().item()

        # --- Adversarial examples (FGSM sur logits restreints) ---
        images_adv_norm = fgsm_attack(
            model=None,
            images=images_norm,
            labels=labels,
            epsilon=epsilon,
            loss_fn=loss_fn,
            forward_fn=forward_restricted,
        )

        # Pour metrics: revenir dans l'espace [0,1]
        images_clean_01 = denormalize_imagenet(images_norm)
        images_adv_01 = denormalize_imagenet(images_adv_norm)

        psnr_batch = compute_psnr_batch(images_clean_01, images_adv_01)
        ssim_batch = compute_ssim_batch(images_clean_01, images_adv_01)
        psnr_values.append(psnr_batch)
        ssim_values.append(ssim_batch)

        # --- Prédictions adversariales ---
        with torch.no_grad():
            logits_adv_subset = forward_restricted(images_adv_norm)  # (B, 10)
            _, preds_adv = torch.max(logits_adv_subset, dim=1)

        adv_correct += (preds_adv == labels).sum().item()
        total += labels.size(0)

        if (i + 1) % 10 == 0:
            clean_acc_batch = (preds_clean == labels).float().mean().item()
            adv_acc_batch = (preds_adv == labels).float().mean().item()
            logging.info(
                f"Batch {i+1:03d} | Clean acc: {clean_acc_batch:.3f} | "
                f"Adv acc: {adv_acc_batch:.3f} | "
                f"PSNR: {psnr_batch:.2f} dB | SSIM: {ssim_batch:.4f}"
            )

    clean_acc = clean_correct / total if total > 0 else 0.0
    adv_acc = adv_correct / total if total > 0 else 0.0
    mean_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0.0
    mean_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0.0

    logging.info(f"Final clean accuracy (restricted): {clean_acc:.3f}")
    logging.info(f"Final adversarial accuracy (FGSM, eps={epsilon}): {adv_acc:.3f}")
    logging.info(f"Mean PSNR (clean vs adv, eps={epsilon}): {mean_psnr:.2f} dB")
    logging.info(f"Mean SSIM (clean vs adv, eps={epsilon}): {mean_ssim:.4f}")


if __name__ == "__main__":
    for eps in [0.001, 0.005, 0.01, 0.02, 0.05]:
        print("\n" + "=" * 60)
        print(f"Running FGSM eval with epsilon={eps}")
        evaluate_fgsm(epsilon=eps)

from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T

from src.models.model_loader import load_pretrained_resnet18
from src.attacks.fgsm import fgsm_attack


class FGSMFilter:
    """
    High-level adversarial filter that:
    - loads a pretrained ResNet-18,
    - applies an untargeted FGSM attack on Imagenette classes,
    - returns a perturbed image tensor.
    """

    def __init__(self, epsilon: float = 0.01, device: str | None = None) -> None:
        model, preprocess, model_device, imagenet_categories = load_pretrained_resnet18(device=device)
        self.model = model
        self.preprocess = preprocess
        self.device = model_device
        self.epsilon = epsilon

        # Same Imagenette mapping logic as in baseline / fgsm_eval
        from torchvision.datasets import Imagenette
        from torch.utils.data import DataLoader

        # Dummy dataset just to recover classes and mapping
        dummy_transform = T.ToTensor()
        dummy_dataset = Imagenette(
            root=str(Path("data") / "imagenette"),
            split="val",
            download=False,
            transform=dummy_transform,
        )
        imagenette_classes = dummy_dataset.classes

        mapping = self._build_imagenette_to_imagenet_mapping(
            imagenette_classes, imagenet_categories
        )
        imnet_indices = torch.tensor(
            [mapping[i] for i in range(len(mapping))],
            device=self.device,
            dtype=torch.long,
        )
        self.imnet_indices = imnet_indices
        self.loss_fn = nn.CrossEntropyLoss()
        self.model.eval()

    @staticmethod
    def _normalize_imagenette_class_name(
        cls_entry,
    ) -> str:
        if isinstance(cls_entry, str):
            return cls_entry
        if isinstance(cls_entry, (list, tuple)) and len(cls_entry) > 0:
            return cls_entry[0]
        raise ValueError(f"Unsupported class entry format: {cls_entry}")

    def _build_imagenette_to_imagenet_mapping(
        self,
        imagenette_classes,
        imagenet_categories: List[str],
    ) -> dict[int, int]:
        mapping: dict[int, int] = {}
        for imgnt_idx, cls_entry in enumerate(imagenette_classes):
            canonical_name = self._normalize_imagenette_class_name(cls_entry)
            try:
                imnet_idx = imagenet_categories.index(canonical_name)
            except ValueError as e:
                raise ValueError(
                    f"Could not find Imagenette class '{canonical_name}' "
                    f"(from {cls_entry}) in ImageNet categories list."
                ) from e
            mapping[imgnt_idx] = imnet_idx
        return mapping

    def _forward_restricted(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)  # (B, 1000)
        logits_subset = outputs.index_select(dim=1, index=self.imnet_indices)  # (B, 10)
        return logits_subset

    def apply_to_tensor(
        self,
        image_tensor: torch.Tensor,
        label: int,
    ) -> torch.Tensor:
        """
        Apply the FGSM filter to a single image tensor in [0, 1] (C, H, W).
        `label` is the Imagenette class index (0-9).
        Returns the adversarial image tensor in [0, 1].
        """
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)  # (1, C, H, W)

        image_tensor = image_tensor.to(self.device)

        # Apply the same preprocess as during training/eval
        # (resize, center crop, normalize) – preprocess accepte un batch
        images_norm = self.preprocess(image_tensor)  # (1, C, H, W) normalized

        labels = torch.tensor([label], device=self.device)

        images_adv_norm = fgsm_attack(
            model=None,
            images=images_norm,
            labels=labels,
            epsilon=self.epsilon,
            loss_fn=self.loss_fn,
            forward_fn=self._forward_restricted,
        )

        # Denormaliser pour revenir en [0,1]
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        images_adv_01 = (images_adv_norm * std + mean).clamp(0.0, 1.0)

        return images_adv_01.squeeze(0).detach().cpu()

    def apply_to_pil(
        self,
        image_pil: Image.Image,
        label: int,
    ) -> Image.Image:
        """
        Apply the FGSM filter to a PIL image and return a PIL image.
        """
        to_tensor = T.ToTensor()
        to_pil = T.ToPILImage()

        img_tensor = to_tensor(image_pil)  # [0,1], (C,H,W)
        adv_tensor = self.apply_to_tensor(img_tensor, label)
        adv_pil = to_pil(adv_tensor)

        return adv_pil

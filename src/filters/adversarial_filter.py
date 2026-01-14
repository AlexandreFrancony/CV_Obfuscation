from typing import Literal, Optional

import torch
import torch.nn as nn

from src.attacks.fgsm import fgsm_attack


AttackType = Literal["fgsm"]  # plus tard: "pgd", "cw", etc.


class AdversarialFilter:
    """
    High-level adversarial filter wrapper that can be applied to images
    before they are passed to a vision model.
    """

    def __init__(
        self,
        model: nn.Module,
        attack_type: AttackType = "fgsm",
        epsilon: float = 0.01,
    ):
        self.model = model
        self.attack_type = attack_type
        self.epsilon = epsilon

    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.attack_type == "fgsm":
            return fgsm_attack(self.model, images, labels, epsilon=self.epsilon)
        else:
            raise ValueError(f"Unsupported attack_type: {self.attack_type}")

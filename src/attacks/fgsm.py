import torch
import torch.nn as nn
from typing import Optional


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    loss_fn: Optional[nn.Module] = None,
) -> torch.Tensor:
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    images_adv = images.clone().detach().requires_grad_(True)

    outputs = model(images_adv)
    loss = loss_fn(outputs, labels)

    model.zero_grad()
    loss.backward()

    grad_sign = images_adv.grad.data.sign()
    images_adv = images_adv + epsilon * grad_sign

    # Clamp à une plage raisonnable pour des inputs normalisés
    images_adv = torch.clamp(images_adv, -3.0, 3.0).detach()

    return images_adv

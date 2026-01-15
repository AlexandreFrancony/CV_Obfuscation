import torch
import torch.nn as nn
from typing import Optional, Callable


def fgsm_attack(
    model: Optional[nn.Module],
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    loss_fn: Optional[nn.Module] = None,
    forward_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Perform an untargeted FGSM attack on a batch of images.

    Either `model` or `forward_fn` must be provided.
    - If `model` is provided, forward_fn(x) = model(x).
    - If `forward_fn` is provided, it should return logits aligned with `labels`.
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    if forward_fn is None:
        if model is None:
            raise ValueError("Either `model` or `forward_fn` must be provided.")
        forward_fn = model  # type: ignore[assignment]

    images_adv = images.clone().detach().requires_grad_(True)

    outputs = forward_fn(images_adv)  # logits alignés avec labels
    loss = loss_fn(outputs, labels)

    if model is not None:
        model.zero_grad()
    else:
        # Si model est None, on assume que forward_fn ne garde pas de gradients internes
        pass

    loss.backward()

    grad_sign = images_adv.grad.data.sign()
    images_adv = images_adv + epsilon * grad_sign

    images_adv = torch.clamp(images_adv, -3.0, 3.0).detach()

    return images_adv

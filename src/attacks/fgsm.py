import torch
import torch.nn as nn


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    loss_fn: nn.Module | None = None,
) -> torch.Tensor:
    """
    Perform an untargeted FGSM attack on a batch of images.

    Args:
        model: PyTorch model in eval() mode.
        images: Input images (batch, C, H, W), already preprocessed.
        labels: Ground-truth labels (batch,).
        epsilon: Perturbation magnitude (in normalized space).
        loss_fn: Loss function (default: nn.CrossEntropyLoss).

    Returns:
        adversarial_images: Perturbed images (clipped to valid range).
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    # Ensure we don't modify the original tensor
    images_adv = images.clone().detach().requires_grad_(True)

    outputs = model(images_adv)
    loss = loss_fn(outputs, labels)

    # Compute gradients w.r.t. input
    model.zero_grad()
    loss.backward()

    # FGSM step: x_adv = x + epsilon * sign(grad_x)
    grad_sign = images_adv.grad.data.sign()
    images_adv = images_adv + epsilon * grad_sign

    # Clamp to valid normalized range (assuming inputs are roughly in [-1, 1])
    images_adv = torch.clamp(images_adv, -3.0, 3.0).detach()

    return images_adv

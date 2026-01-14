import torch
from torch import Tensor
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _to_numpy_image_batch(t: Tensor):
    """
    Convert a batch of images (B, C, H, W) in [0, 1] to numpy (B, H, W, C).
    """
    t = t.detach().cpu().clamp(0.0, 1.0)
    return t.permute(0, 2, 3, 1).numpy()  # (B, H, W, C)


def compute_psnr_batch(original: Tensor, perturbed: Tensor) -> float:
    """
    Compute mean PSNR over a batch of images assumed to be in [0, 1].
    """
    x = _to_numpy_image_batch(original)
    y = _to_numpy_image_batch(perturbed)

    psnr_values = []
    for i in range(x.shape[0]):
        psnr = peak_signal_noise_ratio(x[i], y[i], data_range=1.0)
        psnr_values.append(psnr)

    return float(sum(psnr_values) / len(psnr_values)) if psnr_values else 0.0


def compute_ssim_batch(original: Tensor, perturbed: Tensor) -> float:
    """
    Compute mean SSIM over a batch of images assumed to be in [0, 1].
    """
    x = _to_numpy_image_batch(original)
    y = _to_numpy_image_batch(perturbed)

    ssim_values = []
    for i in range(x.shape[0]):
        ssim = structural_similarity(
            x[i], y[i], channel_axis=-1, data_range=1.0
        )
        ssim_values.append(ssim)

    return float(sum(ssim_values) / len(ssim_values)) if ssim_values else 0.0

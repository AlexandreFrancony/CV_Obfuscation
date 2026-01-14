import torch
from torch import Tensor
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(original: Tensor, perturbed: Tensor) -> float:
    """
    Compute PSNR between two batches of images (in [0, 1] space).
    """
    # original, perturbed: (B, C, H, W)
    original_np = original.detach().cpu().numpy()
    perturbed_np = perturbed.detach().cpu().numpy()

    psnr_values = []
    for i in range(original_np.shape[0]):
        # skimage expects (H, W, C)
        x = original_np[i].transpose(1, 2, 0)
        y = perturbed_np[i].transpose(1, 2, 0)
        psnr_values.append(peak_signal_noise_ratio(x, y, data_range=1.0))

    return float(sum(psnr_values) / len(psnr_values))


def compute_ssim(original: Tensor, perturbed: Tensor) -> float:
    """
    Compute SSIM between two batches of images (in [0, 1] space).
    """
    original_np = original.detach().cpu().numpy()
    perturbed_np = perturbed.detach().cpu().numpy()

    ssim_values = []
    for i in range(original_np.shape[0]):
        x = original_np[i].transpose(1, 2, 0)
        y = perturbed_np[i].transpose(1, 2, 0)
        ssim = structural_similarity(x, y, channel_axis=-1, data_range=1.0)
        ssim_values.append(ssim)

    return float(sum(ssim_values) / len(ssim_values))

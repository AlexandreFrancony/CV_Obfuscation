import argparse
from pathlib import Path

from PIL import Image
from torchvision.utils import save_image
import torch
import torchvision.transforms as T

from src.filters.fgsm_filter import FGSMFilter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply FGSM adversarial filter to all images in a folder."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Folder containing input images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Folder where filtered images will be saved.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="FGSM epsilon (perturbation magnitude).",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=0,
        help="Imagenette class index (0-9) to use as target label for the filter.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fgsm_filter = FGSMFilter(epsilon=args.epsilon)

    to_tensor = T.ToTensor()

    image_paths = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        img_tensor = to_tensor(img)  # [0,1], (C,H,W)

        adv_tensor = fgsm_filter.apply_to_tensor(img_tensor, label=args.label)

        out_path = output_dir / img_path.name
        save_image(adv_tensor, str(out_path))

        print(f"Saved filtered image to {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()

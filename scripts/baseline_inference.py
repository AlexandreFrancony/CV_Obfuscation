import logging
from pathlib import Path
from typing import Dict, List, Sequence, Union

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Imagenette

from src.models.model_loader import load_pretrained_resnet18


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


from typing import Tuple
from pathlib import Path
import logging

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Imagenette


def get_dataset_and_dataloader(
    preprocess, batch_size: int = 32
) -> Tuple[Imagenette, DataLoader]:
    """
    Load the Imagenette validation split and return both the dataset
    (for class names) and the dataloader (for evaluation).

    If the dataset is not found on disk, a clear log message is emitted
    explaining how to enable download.
    """
    data_root = Path("data") / "imagenette"

    try:
        dataset = Imagenette(
            root=str(data_root),
            split="val",
            download=False,  # set to True on a new machine to auto-download
            transform=preprocess,
        )
    except RuntimeError as e:
        logging.error(
            "Imagenette dataset not found at '%s'.\n"
            "If you are running this project on a new machine, "
            "edit the function `get_dataset_and_dataloader` in `scripts/baseline_inference.py` "
            "and set `download=True` in the `Imagenette(...)` call (line 44), "
            "then rerun the script.",
            data_root,
        )
        raise e

    logging.info(f"Loaded Imagenette val split with {len(dataset)} samples.")
    logging.info(f"Imagenette classes (dataset.classes): {dataset.classes}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataset, dataloader

def _normalize_imagenette_class_name(
    cls_entry: Union[str, Sequence[str]]
) -> str:
    """
    Convert the Imagenette class entry (tuple of synonyms or single string)
    to a canonical string we can match against ImageNet categories.

    Strategy:
    - If it's a tuple/list, use the first element (e.g. 'tench').
    - If it's already a string, return as is.
    """
    if isinstance(cls_entry, str):
        return cls_entry
    if isinstance(cls_entry, (list, tuple)) and len(cls_entry) > 0:
        return cls_entry[0]
    raise ValueError(f"Unsupported class entry format: {cls_entry}")


def build_imagenette_to_imagenet_mapping(
    imagenette_dataset: Imagenette,
    imagenet_categories: List[str],
) -> Dict[int, int]:
    """
    Build a mapping from Imagenette class index (0-9) to ImageNet class index (0-999).

    Imagenette exposes classes as tuples of synonyms, e.g.:
    ('tench', 'Tinca tinca'), ('English springer', 'English springer spaniel'), ...

    ImageNet categories from the pretrained weights meta are single strings.
    We match using the first element of each Imagenette tuple.
    """
    mapping: Dict[int, int] = {}
    imagenette_classes = imagenette_dataset.classes

    logging.info(f"Building Imagenette -> ImageNet mapping...")
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


def evaluate_baseline_restricted(batch_size: int = 32) -> None:
    """
    Evaluate a pretrained ResNet-18 on Imagenette, restricting predictions
    to the 10 Imagenette classes by mapping them to the corresponding
    ImageNet indices.
    """
    setup_logging()
    logging.info("Loading pretrained ResNet-18...")

    model, preprocess, device, imagenet_categories = load_pretrained_resnet18()
    logging.info(f"Model loaded on device: {device}")

    dataset, dataloader = get_dataset_and_dataloader(preprocess, batch_size=batch_size)

    # Build mapping: Imagenette label (0-9) -> ImageNet index (0-999)
    mapping = build_imagenette_to_imagenet_mapping(dataset, imagenet_categories)

    # List of ImageNet indices corresponding to Imagenette labels 0..9
    imnet_indices = torch.tensor(
        [mapping[i] for i in range(len(mapping))],
        device=device,
        dtype=torch.long,
    )
    logging.info(f"ImageNet indices used for restriction: {imnet_indices.tolist()}")

    correct = 0
    total = 0

    logging.info("Starting restricted baseline evaluation on Imagenette (val)...")
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)  # shape: (B, 1000)

            # Restrict logits to the 10 relevant ImageNet classes
            logits_subset = outputs.index_select(dim=1, index=imnet_indices)  # (B, 10)

            _, preds = torch.max(logits_subset, dim=1)  # preds in [0..9]

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (i + 1) % 10 == 0:
                batch_acc = (preds == labels).float().mean().item()
                logging.info(f"Batch {i+1:03d} | Batch accuracy: {batch_acc:.3f}")

    accuracy = correct / total if total > 0 else 0.0
    logging.info(f"Restricted top-1 accuracy on Imagenette (val): {accuracy:.3f}")


if __name__ == "__main__":
    evaluate_baseline_restricted()

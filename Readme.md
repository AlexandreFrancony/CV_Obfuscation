# AI-Resistant Image Filtering through Adversarial Perturbations

This repository contains the code for a computer vision project aiming to design **imperceptible adversarial filters** that degrade model performance while preserving visual quality. The primary goal is to study the trade-off between invisibility and effectiveness of such perturbations on pre-trained vision models and, later, on generative models.

## 1. Project structure

The repository is currently organized as follows:

- `requirements.txt` – Python dependencies for the project.
- `.gitignore` – Git ignore rules (virtual environment, caches, data, models, etc.).
- `README.md` – This file.
- `Reports/` – Project proposal and state-of-the-art documents.
- `scripts/` – Python entrypoints for experiments (baseline, attacks, etc.).
- `src/` – Source code (models, attacks, filters, metrics).
- `data/` – Local datasets (e.g., Imagenette) downloaded by scripts (not committed).
- `.venv/` – Local Python virtual environment (not tracked by Git).

The `src/` folder is organized as:

- `src/models/` – Model loading utilities (e.g., pretrained ResNet-18).
- `src/attacks/` – Adversarial attack implementations (e.g., FGSM).
- `src/filters/` – High-level adversarial filters to be applied on images.
- `src/metrics/` – Image quality and evaluation metrics.

## 2. Python virtual environment

### 2.1 Create the virtual environment

From the project root:

```bash
python -m venv .venv
```

### 2.2 Activate the virtual environment

- **Linux / macOS**

```bash
source .venv/bin/activate
```

- **Windows (PowerShell)**

```bash
.venv\Scripts\Activate.ps1
```

To deactivate:

```bash
deactivate
```

## 3. Install dependencies

Upgrade `pip` and install all dependencies listed in `requirements.txt`:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3.1 Optional: GPU-enabled PyTorch (CUDA)

On machines with an NVIDIA GPU (e.g., RTX 3070), you can install a CUDA-enabled build of PyTorch before installing the rest of the requirements. Refer to the official PyTorch “Get Started” page for the recommended command for your system. [web:104]

Typical example for Windows + CUDA 11.8 (adapt if needed):

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

You can verify that CUDA is available with:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 4. Baseline experiment (ResNet-18 + Imagenette)

The initial baseline uses:

- A **ResNet-18** model pretrained on ImageNet, loaded via `torchvision.models` with `ResNet18_Weights.DEFAULT`. [web:18]  
- The **Imagenette** dataset (10-class subset of ImageNet) from `torchvision.datasets.Imagenette`. [web:61][web:71]

The baseline script:

- Restricts the 1000-class ImageNet logits to the 10 Imagenette classes using a mapping between class names.
- Evaluates top-1 accuracy on the validation split.

To run the baseline (and trigger the first Imagenette download):

```bash
python -m scripts.baseline_inference
```

The script will:
- Download Imagenette into `data/imagenette` if not present.
- Log per-batch accuracy and a final restricted top-1 accuracy.

## 5. FGSM adversarial evaluation

The current adversarial baseline implements an **untargeted FGSM attack** on the Imagenette classification task:

- Attacks ResNet-18 via gradients computed on the logits restricted to the 10 Imagenette classes.
- Evaluates both clean accuracy and adversarial accuracy for different values of `epsilon`.

To run the FGSM evaluation:

```bash
python -m scripts.fgsm_eval
```

This will:

- Reuse the same pretrained ResNet-18 and Imagenette mapping as the baseline.
- Log per-batch clean / adversarial accuracy.
- Report the final clean and adversarial accuracies for each tested `epsilon`.

## 6. Run the FGSM demo on custom images

From the project root:

```bash
python -m scripts.apply_fgsm_filter \
  --input-dir data/demo_input \
  --output-dir data/demo_output \
  --epsilon 0.01 \
  --label 3
```

This command applies the imperceptible FGSM filter to all images in `data/demo_input/` and writes the filtered versions to `data/demo_output/`.


## 7. Reproduce Imagenette results

```bash
python -m scripts.evaluate_imagenette \
  --epsilon 0.01 \
  --batch-size 64
```

This script prints clean and adversarial accuracy, as well as PSNR and SSIM, corresponding to the values reported in `report.md`.

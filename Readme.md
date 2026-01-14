# AI-Resistant Image Filtering through Adversarial Perturbations

This repository contains the code for a computer vision project aiming to design **imperceptible adversarial filters** that degrade model performance while preserving visual quality. The primary goal is to study the trade-off between invisibility and effectiveness of such perturbations on pre-trained vision models and, later, on generative models.

## 1. Project structure (initial)

For now, the repository is organized as follows:

- `requirements.txt` – Python dependencies for the project.
- `.gitignore` – Git ignore rules (virtual environment, caches, data, models, etc.).
- `README.md` – This file.
- `.venv/` – Local Python virtual environment (not tracked by Git).

Additional folders such as `src/`, `notebooks/`, `data/`, and `models/` will be added as the project evolves.

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
pip install --upgrade pip
pip install -r requirements.txt
```

If you need a specific CPU/GPU build of PyTorch, refer to the official installation commands and adjust the `torch`, `torchvision`, and `torchaudio` lines in `requirements.txt` accordingly. [web:31][web:33]

## 4. Jupyter kernel (optional)

If you plan to use Jupyter notebooks for experiments, create a dedicated kernel:

```bash
python -m ipykernel install --user --name ai-resistant-filter --display-name "AI Resistant Filter"
```

You can then select the **AI Resistant Filter** kernel from your Jupyter / VS Code environment.

---

Further sections (baseline model, dataset description, experiment pipeline, and evaluation protocol) will be added once the core training and adversarial filtering scripts are implemented.

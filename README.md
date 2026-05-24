# Computer Vision Projects

---

## Project Title and Overview

This repository contains a collection of focused computer vision projects and experiments covering classical stereo vision, feature matching and panorama assembly, and deep-learning-based image classification (facial expression / emotion recognition using transfer learning). The codebase is organized into four project folders; each folder is a self-contained assignment or experiment with notebooks, reference scripts, trained model weights, and example datasets.

Primary use cases:
- Reproduce training and inference for emotion classification using transfer learning (DenseNet121, ResNet50, MobileNet/Inception variants).
- Explore classical stereo vision methods (block matching and dynamic programming for disparity estimation) with visual results.
- Run feature matching and simple image stitching / AR demos for panorama and augmented-reality style outputs.

---

## Core Architecture and Features

This workspace contains three main technical tracks:

- Transfer-learning-based Image Classification (Project 4)
  - Architectures: `DenseNet121`, `ResNet50`, MobileNet + Inception variants. Training and transfer-learning experiments are provided as Jupyter notebooks and include exported model weights (`.pth`). See [Project 4/DenseNet121.ipynb](Project%204/DenseNet121.ipynb) and [Project 4/TransferLearning.ipynb](Project%204/TransferLearning.ipynb).
  - Dataset organization: class folders under `Project 4/dataset/` (Angry, happy, Other, Sad). Augmented images are present (filenames prefixed with `aug-...`) to support data-augmentation-based training.
  - Pipeline steps (high level): dataset loading and augmentation -> model selection and backbone freezing -> classifier head adaptation -> training loop with validation -> checkpointing and export (`.pth`) -> inference & confusion-matrix visualization.

- Classical Stereo Vision (Project 3)
  - Algorithms: block matching and dynamic programming for disparity map estimation, with visual results stored under `Project 3/results/`.
  - Notebooks: [Project 3/block_matching.ipynb](Project%203/block_matching.ipynb) and [Project 3/dynamic_programming.ipynb](Project%203/dynamic_programming.ipynb) provide algorithm descriptions, parameter sweeps (window sizes, search ranges), and result visualizations.

- Feature Matching and Panorama / AR (Project 2 and Project 1)
  - Feature detectors / descriptors and matching approaches implemented in assignment notebooks and scripts (e.g., `Project 2/part1.py`, `Project 2/part2.py`, `Project 1/partB.py`).
  - Outputs: stitched panoramas and example AR overlays (see `Project 2/ar_output.mp4`, and `Project 2/assignment_2_materials/`).

Mathematical background and evaluation
- Classification: standard cross-entropy loss and accuracy metrics; confusion matrices used for per-class analysis.
- Stereo: disparity estimation via sum-of-absolute-differences (SAD) and sum-of-squared-differences (SSD) objective functions; dynamic programming for global optimization across scanlines.

---

## Repository Structure

Top-level layout (key files and directories):

- Project 1/
  - partA.ipynb, partB.ipynb, partB.py — coursework on basic feature detection and matching.
  - part1-images/, part2-images/ — example input images used by the notebooks and scripts.

- Project 2/
  - part1.ipynb, part1.py, part2.ipynb, part2.py — panorama and AR assignment code.
  - assignment_2_materials/ — source images and assets for assignment exercises.
  - ar_output.mp4 — example output video for AR demo.

- Project 3/
  - block_matching.ipynb, dynamic_programming.ipynb — stereo vision algorithms and analysis.
  - results/ — generated disparity and visualization images.
  - stereo_materials/ — left/right image pairs used for experiments.

- Project 4/
  - DenseNet121.ipynb, ResNet.ipynb, TransferLearning.ipynb, VGG.ipynb — model training and transfer-learning experiments.
  - mobilenet_inception.ipynb, mobilenet_inception_updated.ipynb — MobileNet + Inception experiments.
  - best_model_DenseNet121.pth, best_resnet50_transfer_learning.pth, mobilenet_scratch.pth, inceptionv3_scratch.pth — exported model weights and checkpoints.
  - dataset/ — labeled image folders (Angry, happy, Other, Sad) and many augmented images used during training.
  - confusion_matrix_transfer_learning.png — example evaluation artifact.

Additional configuration:
- .vscode/settings.json — workspace editor settings.

---

## Prerequisites and Environment Setup

Recommended system configuration:
- Operating system: Windows / Linux / macOS (code tested in notebook form; path separators are platform dependent).
- Python: 3.8 or later.
- GPU (recommended for training): NVIDIA GPU with CUDA 11.x support for reasonable training times. CPU-only execution is supported for inference or small experiments but will be slower.

Python packages (high-level):
- PyTorch (torch, torchvision) — core training and inference.
- OpenCV (opencv-python) — image I/O, preprocessing, classical CV algorithms.
- NumPy, SciPy — numeric utilities.
- scikit-image, matplotlib, seaborn — visualization and image utilities.
- Jupyter / JupyterLab — notebooks.

Example commands to create a virtual environment and install common dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # CPU-only fallback
pip install opencv-python numpy scipy scikit-image matplotlib seaborn jupyter
```

If you have CUDA and want GPU-accelerated PyTorch, follow the official install instructions at https://pytorch.org to select the right wheel for your CUDA version, then install the remainder of the dependencies with `pip`.

---

## Installation and Usage

Clone the repository and start from the project folder you want to run:

```bash
git clone <repository-url> Computer-Vision-Projects
cd Computer-Vision-Projects
```

Notebooks
- Launch Jupyter and open the notebook for the project you want to run. Example:

```bash
jupyter lab    # or jupyter notebook
```

Running example scripts
- Project 1 feature matching script (example):

```bash
python "Project 1/partB.py"
```

- Project 2 panorama example (scripts provided as `part1.py`, `part2.py`):

```bash
python "Project 2/part1.py"
python "Project 2/part2.py"
```

Training and inference (Project 4)
- The experiments are primarily implemented as notebooks. A typical training flow inside a notebook follows these steps: dataset preparation -> transforms and augmentation -> model backbone selection (e.g., DenseNet121) -> head replacement -> optimizer + scheduler -> training loop -> checkpoint save. Pretrained / best checkpoints are available in the repository. Example inference snippet (PyTorch):

```python
import torch
from torchvision import models, transforms
from PIL import Image

# example: load DenseNet121 checkpoint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.densenet121(pretrained=False)
# adapt final layer if needed (not shown, depends on notebook code)
checkpoint = torch.load('Project 4/best_model_DenseNet121.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open('Project 4/dataset/happy/001.jpg').convert('RGB')
input_tensor = preprocess(img).unsqueeze(0).to(device)
with torch.no_grad():
    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1)
    predicted = torch.argmax(probs, dim=1).item()

print('Predicted class index:', predicted)
```

Replace the checkpoint loading and final-layer adaptation to match the exact implementation in the notebook you choose to run (see [Project 4/TransferLearning.ipynb](Project%204/TransferLearning.ipynb)).

---

## Performance and Evaluation

Evaluation artifacts included in the repository:
- Confusion matrix image for transfer-learning experiments: [Project 4/confusion_matrix_transfer_learning.png](Project%204/confusion_matrix_transfer_learning.png)
- Example model checkpoints: [Project 4/best_model_DenseNet121.pth](Project%204/best_model_DenseNet121.pth), [Project 4/best_resnet50_transfer_learning.pth](Project%204/best_resnet50_transfer_learning.pth)

Recommended evaluation protocol (table template you can reproduce in a notebook):

| Model | Dataset Split | Accuracy | Precision | Recall | F1 | Inference FPS (GPU/CPU) | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| DenseNet121 (transfer) | val | 0.92 | 0.91 | 0.92 | 0.915 | 120 / 10 | example — replace with measured values |
| ResNet50 (transfer) | val | 0.90 | 0.89 | 0.90 | 0.895 | 95 / 8 | example — replace with measured values |

How to benchmark inference speed
1. Use a representative batch size and input resolution (e.g., 224x224). Warm up the model for a few iterations.
2. Measure time per forward pass with `torch.cuda.synchronize()` before/after timing for GPU, or `time.perf_counter()` for CPU.
3. Report median and 95th percentile latencies plus derived FPS.

Suggested metric collection snippet (PyTorch):

```python
import time
import torch

model.eval()
input_batch = torch.randn(32, 3, 224, 224).to(device)
with torch.no_grad():
    # warm up
    for _ in range(10):
        _ = model(input_batch)

    times = []
    for _ in range(50):
        start = time.perf_counter()
        _ = model(input_batch)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        times.append(time.perf_counter() - start)

    median = sorted(times)[len(times)//2]
    fps = len(input_batch) / median
    print(f'Median latency (s): {median:.6f}, Approx FPS: {fps:.1f}')
```

---

## Notes and Next Steps

- The notebooks are the canonical source of the implementation details for each experiment; replicate runs and parameter sweeps inside the notebooks for reproducible results.
- To convert a notebook experiment to a script, extract the data-preparation and training loop cells into a Python module and parameterize via CLI flags or a small configuration file.
- If you would like, I can: generate a `requirements.txt`, extract training scripts from notebooks into runnable `.py` files, or add CI-friendly smoke tests for key scripts.

---

## License and Attribution

This repository contains course/assignment code and trained artifacts. Review any included datasets and external libraries for their respective licenses before redistribution. If you want, I can add a specific license file.

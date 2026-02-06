# DiagCovidPNA
This is a deep learning system for COVID-19 and pneumonia diagnosis from chest X-ray images using transfer learning with Xception, Inception-v3, and Inception-ResNet-v2 architectures.
The code implements the paper: https://doi.org/10.1007/s00500-023-08915-1

## Reproduction Results
Expriments are reproduced with the exact details in the paper.

- Models 1–5 and 7-9: Successfully reproduced with results being consistent with the original reported performance.  
- Model 6 (Inception-V3 fine-tuned model): The model was unable to perform unlike the paper, likely due to differences in transfer learning based fine tuning of the Inception-V3 backbone.

  
## Overview

This project implements a hybrid diagnostic system that classifies chest X-rays into four categories:
- **COVID-19**
- **Normal**
- **Bacterial Pneumonia**
- **Viral Pneumonia**

The hybrid approach uses a two-stage classification:
1. **3-class model**: Classifies images as COVID-19, Normal, or Pneumonia
2. **2-class model**: Further classifies Pneumonia cases as Bacterial or Viral

## Reproduction Results

Evaluation results on the test set:

| Model | Architecture | Accuracy (%) | F-measure (%) | Precision (%) | Recall (%) |
|-------|--------------|--------------|---------------|---------------|------------|
| Model 1 | Xception (4-class) | 95.56 | 95.55 | 95.60 | 95.56 |
| Model 2 | Xception (4-class) | 94.78 | 94.76 | 94.86 | 94.78 |
| Model 3 | Xception (4-class) | 94.50 | 94.51 | 94.52 | 94.50 |
| Model 4 | Xception (4-class) | 94.50 | 94.47 | 94.48 | 94.50 |
| Model 5 | Xception (4-class) | **96.56** | **96.55** | **96.56** | **96.56** |
| Model 6 | Inception-v3 (4-class) | 94.72 | 94.71 | 94.71 | 94.72 |
| Model 7 | Inception-ResNet-v2 (4-class) | 95.33 | 95.31 | 95.41 | 95.33 |
| Model 8 | Xception (2-class) | 93.33 | 93.33 | 93.46 | 93.33 |
| Model 9 | Xception (3-class) | **99.17** | **99.17** | **99.17** | **99.17** |
| Model 10 | Hybrid (3-class → 2-class) | 95.94 | 95.92 | 95.96 | 95.94 |

**Best Results:**
- **Model 5 (Xception)** achieves the best 4-class accuracy at **96.56%**
- **Model 9 (3-class)** achieves the highest overall accuracy at **99.17%**
- **Model 10 (Hybrid)** combines models 8 and 9 for 4-class classification at **95.94%**

## Project Structure

```
DiagCovidPNA/
├── src/
│   ├── config.py          # Configuration and hyperparameters
│   ├── data.py            # Data loading and augmentation
│   ├── models.py          # Model architectures (Xception, Inception-v3, Inception-ResNet-v2)
│   ├── train.py           # Training script
│   ├── evaluation.py      # Model evaluation and metrics
│   └── visualize.py       # Grad-CAM visualization
├── data/
│   ├── Train/             # Training images
│   ├── Valid/             # Validation images
│   └── Test/              # Test images
├── checkpoints/           # Saved model weights and training history
├── pyproject.toml         # Project dependencies
└── README.md
```

## Setup

### 1. Install uv (Python package manager)

```bash
pip install uv
```

### 2. Clone and setup the project

```powershell
cd DiagCovidPNA
uv sync
```

This will create a virtual environment and install all dependencies including:
- PyTorch with CUDA 12.4 support
- timm (for Xception and Inception-ResNet-v2)
- torchvision
- matplotlib
- pandas
- opencv-python
- grad-cam

### 3. Prepare the data

Organize your chest X-ray images in the following structure:

```
data/
├── Train/
│   ├── Covid-19/
│   ├── Normal/
│   ├── Bacterial Pneumonia/
│   └── Viral Pneumonia/
├── Valid/
│   └── (same structure)
└── Test/
    └── (same structure)
```

## Configuration (`src/config.py`)

All hyperparameters and model configurations are centralized in `config.py`:


### Training Hyperparameters

| Setting | Single Model (4-class) | Hybrid Model (2/3-class) |
|---------|------------------------|--------------------------|
| Learning Rate | `LR_SINGLE = 0.0003` | `LR_HYBRID = 0.00003` |
| Batch Size | `BATCH_SIZE_SINGLE = 32` | `BATCH_SIZE_HYBRID = 16` |
| Epochs | `EPOCHS_SINGLE = 100` | `EPOCHS_HYBRID = 150` |

### Model Configuration Flags

Each model in `Config.MODELS` has these flags:

| Flag | Description |
|------|-------------|
| `architecture` | Base model: `'xception'`, `'inception_v3'`, or `'inception_resnet_v2'` |
| `num_classes` | Output classes: `4` (all), `3` (COVID/Normal/Pneumonia), `2` (Bacterial/Viral) |
| `use_contrast` | Enable contrast augmentation (1.0-1.5× random adjustment) |
| `use_augmentation` | Enable zoom (0.85-1.0 crop) and translation (±10% shift) |
| `use_transfer_learning` | Load ImageNet pretrained weights |

### Example Model Config
```python
'model5': {
    'architecture': 'xception',        # Use Xception backbone
    'num_classes': 4,                  # 4-class classification
    'use_contrast': True,              # Enable contrast augmentation
    'use_augmentation': True,          # Enable zoom + translation
    'use_transfer_learning': True      # Use ImageNet weights
}
```

## Usage

### Training

Train models:

```powershell
# Trains all models
uv run src/train.py
```

Or modify `src/train.py` to train different models:

```python
# Train different models
train(Config, model_id='model1', class_mode='4-class')  # Xception, no contrast
train(Config, model_id='model6', class_mode='4-class')  # Inception-v3
train(Config, model_id='model8', class_mode='2-class')  # 2-class (Bacterial vs Viral)
train(Config, model_id='model9', class_mode='3-class')  # 3-class (COVID vs Normal vs Pneumonia)
```

### Model Configurations

| Model | Architecture | Contrast | Augmentation | Transfer Learning |
|-------|--------------|----------|--------------|-------------------|
| model1 | Xception | ❌ | ✅ | ✅ |
| model2 | Xception | ✅ | ❌ | ✅ |
| model3 | Xception | ❌ | ❌ | ✅ |
| model4 | Xception | ✅ | ✅ | ❌ |
| model5 | Xception | ✅ | ✅ | ✅ |
| model6 | Inception-v3 | ✅ | ✅ | ✅ |
| model7 | Inception-ResNet-v2 | ✅ | ✅ | ✅ |
| model8 | Xception (2-class) | ✅ | ✅ | ✅ |
| model9 | Xception (3-class) | ✅ | ✅ | ✅ |

### Evaluation

Evaluate all trained models:

```powershell
uv run src/evaluation.py
```

### Visualization (Grad-CAM)

Generate Grad-CAM heatmaps to visualize model attention:

```powershell
uv run src/visualize.py
```

This creates a figure showing original images and their corresponding heatmaps for each class.

## Hyperparameters

The training automatically selects hyperparameters based on `num_classes`:

| Parameter | Single Model (4-class) | Hybrid Model (2/3-class) |
|-----------|--------------|--------------|
| Learning Rate | 0.0003 | 0.00003 |
| Batch Size | 32 | 16 |
| Epochs | 100 | 150 |
| Image Size | 150×150 | 150×150 |
| Optimizer | Adam | Adam |

## Data Augmentation

When enabled via config flags, the following augmentations are applied during training:

| Augmentation | Flag | Effect |
|--------------|------|--------|
| **Zoom** | `use_augmentation` | Random crop with scale 0.85-1.0 |
| **Translation** | `use_augmentation` | Random shift ±10% horizontal/vertical |
| **Contrast** | `use_contrast` | Random adjustment 1.0-1.5× |

## Training Optimizations

The training script includes several GPU optimizations:

- **Mixed Precision (FP16)**: Uses `torch.amp.autocast` and `GradScaler` for faster training
- **cuDNN Benchmark**: `torch.backends.cudnn.benchmark = True` for optimized convolutions
- **Non-blocking transfers**: `non_blocking=True` for async CPU→GPU data transfers
- **Gradient optimization**: `optimizer.zero_grad(set_to_none=True)` for memory efficiency

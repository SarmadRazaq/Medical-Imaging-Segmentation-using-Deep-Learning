# Medical Imaging Segmentation using Deep Learning

A deep learning project for multi-organ segmentation in CT scans using the RSNA dataset. This project implements a U-Net architecture with MobileNetV3 encoder for efficient and accurate segmentation of abdominal organs.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Training Configuration](#training-configuration)
- [Results & Evaluation](#results--evaluation)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)

## 🔬 Overview

This project performs semantic segmentation on medical CT images to identify and segment 6 different classes:
- **Class 0**: Background
- **Class 1**: Liver
- **Class 2**: Spleen
- **Class 3**: Left Kidney
- **Class 4**: Right Kidney
- **Class 5**: Bowel

## 📊 Dataset

- **Source**: RSNA (Radiological Society of North America) PNG Dataset
- **Image Size**: Resized to 512x512 pixels
- **Format**: PNG images (converted from DICOM)
- **Split**: 90% Training / 10% Validation

## 🏗️ Model Architecture

| Component | Details |
|-----------|---------|
| **Architecture** | U-Net |
| **Encoder** | MobileNetV3-Large (timm-mobilenetv3_large_100) |
| **Pre-trained Weights** | ImageNet |
| **Input Channels** | 3 (Grayscale stacked to RGB) |
| **Output Classes** | 6 |
| **Decoder** | U-Net decoder with Batch Normalization |

## 💻 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended: NVIDIA A100 or similar)
- Google Colab (optional, for cloud training)

### Dependencies

```bash
# Core Deep Learning
pip install torch torchvision

# Segmentation Models
pip install segmentation-models-pytorch

# Image Processing
pip install opencv-python
pip install albumentations>=1.4.0
pip install scikit-image>=0.23.0

# Data Science
pip install numpy pandas scipy>=1.13
pip install scikit-learn
pip install matplotlib seaborn

# DICOM Support (optional)
pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg

# Progress Bars
pip install tqdm
```

## 🚀 Usage

### Training

```python
# Mount Google Drive (if using Colab)
from google.colab import drive
drive.mount('/content/drive')

# Run the training notebook
# The model will be saved to Google Drive automatically
```

### Inference

```python
import torch
import segmentation_models_pytorch as smp

# Load Model
model = smp.Unet(
    encoder_name="timm-mobilenetv3_large_100",
    encoder_weights=None,
    in_channels=3,
    classes=6
)

# Load trained weights
model.load_state_dict(torch.load("best_mobilenet_rsna_refined.pth"))
model.eval()

# Predict
with torch.no_grad():
    output = model(input_image)
    prediction = torch.argmax(output, dim=1)
```

## ⚙️ Training Configuration

### Initial Training (512px)

| Parameter | Value |
|-----------|-------|
| Image Size | 512 x 512 |
| Batch Size (Real) | 32 |
| Virtual Batch Size | 512 |
| Gradient Accumulation Steps | 16 |
| Learning Rate | 1e-3 |
| Epochs | 30 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |

### Fine-Tuning

| Parameter | Value |
|-----------|-------|
| Learning Rate | 5e-5 |
| Epochs | 10 |
| Batch Size | 64 |

### Loss Function
- **Combined Loss**: 50% Dice Loss + 50% Focal Loss
- Dice Loss: Handles class imbalance in segmentation
- Focal Loss: Focuses on hard-to-classify pixels

### Data Augmentation

```python
# Training Augmentations
- HorizontalFlip (p=0.5)
- VerticalFlip (p=0.5)
- Rotation (limit=45°, p=0.5)
- GridDistortion (p=0.3)
- CoarseDropout (p=0.2)
```

## 📈 Results & Evaluation

### Metrics Computed
- **Pixel Accuracy**: Overall correct pixel classification
- **Dice Score**: Per-class overlap metric (2×Intersection / Sum)
- **IoU (Intersection over Union)**: Per-class segmentation quality
- **AUC-ROC**: Per-class discrimination ability

### Visualization Outputs
1. **Training Curves**: Loss and Accuracy over epochs
2. **Confusion Matrix**: Pixel-wise classification performance
3. **ROC Curves**: Multi-class AUC per organ
4. **Prediction Visualization**: Side-by-side comparison of Input, Ground Truth, and Prediction

## 🛠️ Technologies Used

### Deep Learning Framework
| Library | Purpose |
|---------|---------|
| PyTorch | Core deep learning framework |
| segmentation-models-pytorch | Pre-built segmentation architectures |
| torch.amp | Mixed precision training (FP16) |

### Image Processing
| Library | Purpose |
|---------|---------|
| OpenCV (cv2) | Image I/O and manipulation |
| Albumentations | Fast image augmentation |
| scikit-image | Image processing utilities |

### Data Science & Visualization
| Library | Purpose |
|---------|---------|
| NumPy | Numerical operations |
| Pandas | Data manipulation |
| Matplotlib | Plotting and visualization |
| Seaborn | Statistical visualizations |
| scikit-learn | Metrics, train/test split, ROC curves |

### Utilities
| Library | Purpose |
|---------|---------|
| tqdm | Progress bars |
| shutil | File operations |
| Google Colab | Cloud GPU training |

## 📁 Project Structure

```
DL segmentation project/
│
├── Medical Imaging Segmentation.ipynb    # Main training & evaluation notebook (Google Colab)
├── RSNA base with avg acuracy.ipynb      # Base training notebook (Kaggle)
├── README.md                              # Project documentation
└── models/                                # Saved model weights
    ├── best_model_512.pth                 # Initial training checkpoint
    ├── best_mobilenet_rsna_refined.pth    # Fine-tuned model (Colab)
    └── best_mobilenet.pth                 # Base model (Kaggle)
```

## 🔧 Optimizations

- **CUDA Benchmark**: `torch.backends.cudnn.benchmark = True`
- **Mixed Precision Training**: Using `torch.amp.autocast` and `GradScaler`
- **Gradient Accumulation**: Virtual batch size of 512 with real batch size of 32
- **Multi-threaded DataLoader**: 16 workers with pin_memory
- **OpenCV Threading**: Disabled for stability (`cv2.setNumThreads(0)`)

## 📝 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- RSNA (Radiological Society of North America) for the dataset
- segmentation-models-pytorch library by Pavel Yakubovskiy
- PyTorch team for the deep learning framework
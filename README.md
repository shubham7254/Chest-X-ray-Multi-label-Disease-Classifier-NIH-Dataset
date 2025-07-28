# Chest X-ray Multi-label Disease Classifier (NIH Dataset)
A complete medical AI pipeline to classify 14 common thoracic diseases from frontal-view chest X-ray images using deep learning. The model is trained and evaluated on the official NIH ChestX-ray14 dataset with rigorous preprocessing, class imbalance handling, and clinically relevant evaluation metrics.

---

## 📌 Dataset

- **Source**: NIH ChestX-ray14
- **Images**: 112,120 frontal-view X-rays from 30,805 unique patients
- **Labels**: Multi-label disease annotations (14 classes)
- **Format**: CSV file `Data_Entry_2017.csv` + 12 image folders `images_001` to `images_012`

---

## Pipeline Overview

### 1. **Data Preprocessing**
- Patient-level deduplication
- Histogram analysis of image widths and heights
- Resized all images to **512×512** to standardize input
- One-hot encoding of 14 disease labels
- Removed unused or corrupted metadata columns

### 2. **Stratified Train/Val/Test Splitting**
- Multi-label **stratified K-Fold** splitting using `MultilabelStratifiedKFold`
- **Patient-level grouping** to avoid data leakage
- Splits: 15% test, 15% validation, 70% training

### 3. **Handling Class Imbalance**
- Computed **class-wise weights** using Negative/Positive sample ratios
- Used `BCEWithLogitsLoss(pos_weight=...)` in PyTorch to apply these weights
- Plotted comparison of original class frequencies vs. loss weights

### 4. **Dataloader & Transforms**
- Custom `Dataset` class dynamically loads from 12 image subfolders
- Torchvision transforms:
  - Resize to 512×512
  - Normalize grayscale channels to [-1, 1]
- Batching with PyTorch `DataLoader`

### 5. **Model Setup**
- Model: **ResNet-18** (pretrained on ImageNet)
- Modified final FC layer for 14 disease outputs
- Multi-GPU support (`nn.DataParallel`)
- Training with **Mixed Precision (AMP)** using `torch.cuda.amp`
- Optimizer: Adam
- LR Scheduler: StepLR
- Loss: Weighted `BCEWithLogitsLoss`

### 6. **Training & Evaluation**
- Trained for 5 epochs on Kaggle GPU (NVIDIA T4 × 2)
- Metrics:
  - Training Loss
  - Validation AUC
  - Final Test AUC

---

## 📊 Medical Evaluation Metrics

calculate clinically relevant metrics per class on the **test set**, including:

| Metric | Description |
|--------|-------------|
| **Precision** | Correct positive predictions out of all positive predictions |
| **Sensitivity (Recall)** | True positive rate – detects how many real positives were identified |
| **Specificity** | True negative rate – identifies how well negatives were predicted |
| **F1 Score** | Harmonic mean of precision and recall |


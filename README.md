<p align="center">
  <h1 align="center">ConvNeXt-Based Multimodal Skin Cancer Classification<br>with Cross-Modal Attention Pooling</h1>
  <p align="center">
    <strong>Yuxin Hou</strong> · Junjie Han · Xinyue Zha · Ruijie Xu · Kuizhang Zhao · <strong>Guanfeng Wu*</strong>
  </p>
  <p align="center">
    School of Mathematics, Southwest Jiaotong University, Chengdu, China
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch" alt="PyTorch">
    <img src="https://img.shields.io/badge/Dataset-ISIC%202019-blue" alt="ISIC 2019">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
    <img src="https://img.shields.io/badge/Accuracy-89.1%25-brightgreen" alt="Accuracy">
  </p>
</p>

---

## Abstract

This study proposes a **multimodal skin cancer classification model** based on the **ConvNeXt** architecture and **cross-modal attention pooling (CMAP)** , aiming to improve the accuracy and efficiency of skin cancer diagnosis. The core innovations include:

1. Adopting a pre-trained **ConvNeXt-Tiny** as the image feature extractor
2. Introducing a **cross-modal attention pooling** mechanism that utilizes clinical metadata (age, gender, anatomical site) to guide the weight allocation of image regions
3. Transforming image features into regional vectors through **spatial pyramid projection**

Experimental results on the ISIC 2019 dataset achieve **89.1% accuracy** and **0.862 macro-F1**, outperforming traditional concatenation-based multimodal fusion methods.

---

## Architecture

```
┌─────────────────────┐     ┌──────────────────────┐
│   Dermoscopic Image │     │   Clinical Metadata   │
│      (224×224)      │     │ (age, sex, site, ...)  │
└────────┬────────────┘     └──────────┬───────────┘
         │                             │
    ┌────▼────┐                   ┌────▼────┐
    │ConvNeXt │                   │  MLP    │
    │  Tiny   │                   │ Encoder │
    └────┬────┘                   └────┬────┘
         │                             │
    ┌────▼──────────┐                  │
    │Spatial Pyramid│    768-dim       │  128-dim
    │  Projection   │ ──► 256-dim      │
    │  (4×4 grid)   │     regions      │
    └────┬──────────┘                  │
         │  16 × 256 dim              │
         └──────────┬─────────────────┘
                    │
              ┌─────▼──────┐
              │ Cross-Modal │
              │  Attention  │
              │  Pooling    │
              │  (CMAP)     │
              └─────┬──────┘
                    │  384-dim fused
               ┌────▼────┐
               │   MLP   │
               │Classifier│
               └────┬────┘
                    │
               ┌────▼────┐
               │ 8-class │
               │ Output  │
               └─────────┘
```

The **Cross-Modal Attention Pooling (CMAP)** module is the key innovation — it dynamically assigns attention weights to different image regions *conditioned on* the patient's clinical metadata, enabling the model to focus on clinically relevant visual cues.

---

## Key Features

- **Multimodal fusion**: Deep integration of dermoscopic images with clinical metadata (age, gender, anatomical site)
- **Cross-modal attention**: Metadata-guided dynamic weighting of image regions, not simple feature concatenation
- **Class imbalance handling**: Inverse-frequency weighted loss + label smoothing
- **Reproducibility**: Fixed random seeds, detailed training configuration, mixed-precision FP16 training
- **Interpretability**: Attention weight visualization reveals which image regions drive predictions

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

| Package       | Version  |
|---------------|----------|
| PyTorch       | ≥ 1.12   |
| torchvision   | ≥ 0.13   |
| timm          | ≥ 0.6    |
| numpy         | ≥ 1.21   |
| pandas        | ≥ 1.3    |
| scikit-learn  | ≥ 1.0    |
| opencv-python | ≥ 4.5    |
| matplotlib    | ≥ 3.5    |
| seaborn       | ≥ 0.11   |
| tqdm          | ≥ 4.62   |

---

## Dataset

This project uses the **ISIC 2019** dataset. Download it from the [ISIC Archive](https://challenge.isic-archive.com/data/) and organize as follows:

```
data/
├── ISIC_2019_Training_GroundTruth.csv
├── ISIC_2019_Training_Metadata.csv
└── ISIC_2019_Training_Input/
    ├── ISIC_0000000.jpg
    ├── ISIC_0000001.jpg
    └── ...
```

The dataset contains **25,331 dermoscopic images** across **8 diagnostic categories**:

| Class | Name                      | Samples |
|-------|---------------------------|---------|
| MEL   | Melanoma                  | 4,522   |
| NV    | Melanocytic Nevus         | 12,875  |
| BCC   | Basal Cell Carcinoma      | 3,323   |
| AK    | Actinic Keratosis         | 867     |
| BKL   | Benign Keratosis          | 2,624   |
| DF    | Dermatofibroma            | 239     |
| VASC  | Vascular Lesion           | 253     |
| SCC   | Squamous Cell Carcinoma   | 628     |

---

## Project Structure

```
skin-cancer-convnext-multimodal/
├── config.py                # Hyperparameters and path configuration
├── data_preprocessing.py    # Data loading, augmentation, metadata encoding
├── model.py                 # ConvNeXt backbone, CMAP module, full model
├── train_utils.py           # Training/validation loops, optimizer, scheduler
├── main.py                  # Main training script and result visualization
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
├── data/                    # (place ISIC 2019 dataset here)
│   └── .gitkeep
└── results/                 # Training outputs
    └── models/              # Saved model checkpoints
```

---

## Usage

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/skin-cancer-convnext-multimodal.git
   cd skin-cancer-convnext-multimodal
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download ISIC 2019 dataset** and place it in the `data/` directory as described above.

4. **Run training**:
   ```bash
   python main.py
   ```

The training will automatically:
- Split data into train/validation sets (80/20)
- Apply data augmentation (flips, rotation, color jitter, resized crop)
- Train with mixed-precision FP16 (if CUDA available)
- Apply early stopping (patience=8)
- Save the best model to `results/models/best_model.pth`

### Results Output

After training, the `results/` directory will contain:
- `models/best_model.pth` — Best model checkpoint
- `training_history.png` — Loss, accuracy, and attention weight evolution
- `confusion_matrix.png` — Confusion matrix on validation set
- `classification_report.csv` — Per-class precision, recall, F1
- `attention_distribution.png` — Final attention weight distribution
- `validation_auc.png` — AUC curve

---

## Experimental Results

### Backbone Comparison

| Backbone         | Accuracy | AUC    | Precision | Recall | F1     |
|-----------------|----------|--------|-----------|--------|--------|
| ResNet50        | 0.7389   | 0.9215 | 0.5725    | 0.7450 | 0.6295 |
| VGG16           | 0.8244   | 0.9331 | 0.7624    | 0.7832 | 0.7722 |
| EfficientNet-B0 | 0.8488   | 0.9213 | 0.7303    | 0.8146 | 0.7617 |
| **ConvNeXt-Tiny** | **0.8769** | 0.9321 | **0.8612** | **0.8169** | **0.8372** |

### Fusion Strategy Ablation

| Fusion Strategy      | Accuracy | AUC    | F1     |
|---------------------|----------|--------|--------|
| Simple Concatenation | 0.8810   | 0.9516 | 0.8383 |
| **CMAP (Ours)**     | **0.8909** | **0.9563** | **0.8619** |

The cross-modal attention mechanism improves recall by **6–9%** on rare classes (AK, VASC).

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{hou2026convnext,
  title     = {ConvNeXt-Based Multimodal Skin Cancer Classification Model
               with Cross-Modal Attention Pooling},
  author    = {Yuxin Hou and Junjie Han and Xinyue Zha and Ruijie Xu
               and Kuizhang Zhao and Guanfeng Wu},
  year      = {2026},
  school    = {Southwest Jiaotong University},
  note      = {Supported by the Fundamental Research Funds for the Central
               Universities (Grant No: 202510613070, 2682025ZTPY057)}
}
```

---

## Acknowledgments

This work is supported by:
- Fundamental Research Funds for the Central Universities (Grant No: 202510613070, 2682025ZTPY057)
- Sichuan Medical Innovation Association (Grant No: YCH-KY-YCZD2024-302)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

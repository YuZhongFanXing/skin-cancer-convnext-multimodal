# ConvNeXt-Based Multimodal Skin Cancer Classification with Cross-Modal Attention Pooling

**Yuxin Hou**, Junjie Han, Xinyue Zha, Ruijie Xu, Kuizhang Zhao, **Guanfeng Wu\***

School of Mathematics, Southwest Jiaotong University, Chengdu, China

---

## Abstract

This study proposes a multimodal skin cancer classification model based on the ConvNeXt architecture and cross-modal attention pooling (CMAP). The key innovations are: (1) ConvNeXt-Tiny as image feature extractor; (2) cross-modal attention pooling that uses clinical metadata (age, gender, anatomical site) to guide attention over image regions; (3) spatial pyramid projection for regional feature representation. On the ISIC 2019 dataset (25,331 images, 8 classes), the model achieves 89.1% accuracy and 0.862 macro-F1, outperforming concatenation-based fusion.

---

## Architecture

```
  Dermoscopic Image (224x224)        Clinical Metadata (age, sex, site)
         |                                      |
  ConvNeXt-Tiny (pretrained)            MLP Encoder
         |                                      |
  Spatial Pyramid Projection                   |
   4x4 grid -> 16 regions x 256-dim            |  128-dim
         |                                      |
         +------------ concat -----------------+
                        |
              Cross-Modal Attention Pooling (CMAP)
              Metadata-guided region weighting
              Weighted Sum -> FC -> BN -> GELU
                        |  384-dim
                   MLP Classifier (8 classes)
```

---

## Key Modules

### Spatial Pyramid Projection (model.py:7-24)

AdaptiveAvgPool2d compresses ConvNeXt feature maps into a 4x4 grid, then 1x1 convolution projects each cell to a 256-dim region vector. This converts dense feature maps to a set of 16 regional representations for the attention mechanism.

### Cross-Modal Attention Pooling (model.py:27-54)

The CMAP module fuses image regions with metadata through attention:

1. Expand metadata to match N image regions
2. Concatenate each region feature with metadata: C_i = [r_i ; m_exp]
3. Compute attention scores via two-layer MLP: s_i = AttentionNet(C_i)
4. Softmax normalize: w_i = exp(s_i) / sum exp(s_j)
5. Weighted sum: F = sum w_i * C_i
6. Post-process: FC -> BatchNorm -> GELU -> Dropout(0.3)

This enables metadata to dynamically guide which image regions the model attends to.

### Loss Function

Weighted cross-entropy with inverse-frequency class weights and label smoothing (epsilon=0.1) to handle severe class imbalance (e.g., 12,875 NV vs 239 DF).

---

## Dataset: ISIC 2019

| Class | Name                    | Samples |
|-------|-------------------------|---------|
| MEL   | Melanoma                | 4,522   |
| NV    | Melanocytic Nevus       | 12,875  |
| BCC   | Basal Cell Carcinoma    | 3,323   |
| AK    | Actinic Keratosis       | 867     |
| BKL   | Benign Keratosis        | 2,624   |
| DF    | Dermatofibroma          | 239     |
| VASC  | Vascular Lesion         | 253     |
| SCC   | Squamous Cell Carcinoma | 628     |

Download from [ISIC Archive](https://challenge.isic-archive.com/data/) and place in `data/`:

```
data/
├── ISIC_2019_Training_GroundTruth.csv
├── ISIC_2019_Training_Metadata.csv
└── ISIC_2019_Training_Input/
    └── *.jpg
```

---

## Quick Start

```bash
git clone https://github.com/YuZhongFanXing/skin-cancer-convnext-multimodal.git
cd skin-cancer-convnext-multimodal
pip install -r requirements.txt
# Download ISIC 2019 dataset to data/
python main.py
```

### Training Configuration

| Parameter       | Value                     |
|-----------------|---------------------------|
| Image Size      | 224 x 224                 |
| Batch Size      | 64                        |
| Epochs          | 30 (early stop patience=8)|
| Optimizer       | AdamW (wd=3e-4)           |
| Learning Rate   | 1e-4, cosine annealing    |
| Mixed Precision | FP16                      |

---

## Results

### Backbone Comparison

| Backbone          | Accuracy | AUC    | Precision | Recall | F1     |
|-------------------|----------|--------|-----------|--------|--------|
| ResNet50          | 0.7389   | 0.9215 | 0.5725    | 0.7450 | 0.6295 |
| VGG16             | 0.8244   | 0.9331 | 0.7624    | 0.7832 | 0.7722 |
| EfficientNet-B0   | 0.8488   | 0.9213 | 0.7303    | 0.8146 | 0.7617 |
| **ConvNeXt-Tiny** | **0.8769** | **0.9321** | **0.8612** | **0.8169** | **0.8372** |

### Fusion Strategy Ablation

| Strategy              | Accuracy | AUC    | F1     |
|-----------------------|----------|--------|--------|
| Simple Concatenation  | 0.8810   | 0.9516 | 0.8383 |
| **CMAP (This work)**  | **0.8909** | **0.9563** | **0.8619** |

Rare class (AK, VASC) recall improved by 6-9%.

---

## Project Structure

```
├── config.py              # Hyperparameters & paths
├── data_preprocessing.py  # ISIC loader, augmentation, metadata encoding
├── model.py               # ConvNeXt + SpatialPyramid + CMAP + Classifier
├── train_utils.py         # AdamW, cosine LR, train/validate loops
├── main.py                # Training pipeline & 5 result visualizations
├── requirements.txt       # Dependencies
└── LICENSE                # MIT
```

---

## Citation

```bibtex
@article{hou2026convnext,
  title     = {ConvNeXt-Based Multimodal Skin Cancer Classification Model
               with Cross-Modal Attention Pooling},
  author    = {Yuxin Hou and Junjie Han and Xinyue Zha and Ruijie Xu
               and Kuizhang Zhao and Guanfeng Wu},
  year      = {2026},
  school    = {Southwest Jiaotong University},
  note      = {Supported by Fundamental Research Funds for the Central
               Universities (202510613070, 2682025ZTPY057) and
               Sichuan Medical Innovation Association (YCH-KY-YCZD2024-302)}
}
```

## License

MIT — see [LICENSE](LICENSE).

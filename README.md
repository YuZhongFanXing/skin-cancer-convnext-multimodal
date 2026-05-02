# ConvNeXt-Based Multimodal Skin Cancer Classification with Cross-Modal Attention Pooling

<div align="center">

**Yuxin Hou** · Junjie Han · Xinyue Zha · Ruijie Xu · Kuizhang Zhao · **Guanfeng Wu\***

School of Mathematics, Southwest Jiaotong University, Chengdu, China

![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch)
![Dataset](https://img.shields.io/badge/Dataset-ISIC%202019-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Accuracy](https://img.shields.io/badge/Accuracy-89.1%25-brightgreen)

</div>

---

## Abstract

This study proposes a **multimodal skin cancer classification model** based on the **ConvNeXt** architecture and **cross-modal attention pooling (CMAP)** . To address the characteristics of image data and clinical metadata (age, gender, anatomical site) in the ISIC 2019 dataset, we designed a novel multimodal fusion strategy. The core innovations include:

1. Adopting a pre-trained **ConvNeXt-Tiny** model as the image feature extractor, leveraging its powerful visual representation capabilities — it retains convolution's advantage in capturing local visual details while enhancing global feature association through modern design elements inspired by Transformers.
2. Introducing a **cross-modal attention pooling** mechanism that utilizes metadata information to guide the weight allocation of image regions, thereby achieving deep fusion of image features and metadata features.
3. Transforming image features into regional vector features through **spatial pyramid projection** to provide fine-grained input for the attention mechanism.

Experimental results on the ISIC 2019 dataset achieve **89.1% accuracy** and **0.862 macro-F1**, significantly outperforming simple feature concatenation baselines. Notably, for rare classes such as Actinic Keratosis (AK) and Vascular lesions (VASC), recall rates improved by **6–9%** .

---

## Motivation

Skin cancers represent the most prevalent categories of cancers diagnosed globally. The two most common types are basal cell carcinoma and squamous cell carcinoma, with melanoma being the third most prevalent. Melanoma is highly treatable when detected early; however, advanced melanoma can metastasize to internal organs, potentially resulting in mortality. An estimated **212,200 new melanoma cases** are expected in the U.S. in 2025 alone.

Barriers to healthcare access contribute to delayed detection. Patients experience substantial wait times — ranging from **33.9 to 73.4 days** — when consulting dermatologists. Unimodal frameworks are inherently constrained by morphological similarities between malignant melanomas and benign nevi. Clinical metadata provides complementary contextual information unobtainable through visual patterns alone, making **multimodal integration necessary** to address diagnostic bottlenecks.

---

## Related Work

### CNN-Based Approaches
- **VGG16**: 89.7% accuracy on HAM10000 dataset; excels in spatial feature learning [1].
- **EfficientNet-B0**: 93.2% accuracy on ISIC 2020; reduces inference latency by 40% vs. larger variants [2].
- **ResNet50**: 87.4% AUC on ISIC 2019; faces challenges with fine-grained lesion subtypes [3].

### Multimodal Methods
Recent work combines image data with patient metadata (gender, age, lesion location) to improve classification. However, existing multimodal systems have significant shortcomings:
- Ignoring correlations between clinical and dermatoscopic images [4]
- Limited generalization ability across diagnostic tasks [5]
- Complexity in implementation and interpretation, limiting clinical application [6]

### Data Augmentation & Imbalance
SMOTE-based augmentation and preprocessing techniques (hair removal, lesion segmentation) have been proposed to address severe category imbalance in skin cancer datasets [2].

---

## Method

### Architecture Overview

The proposed multimodal framework based on **ConvNeXt-Tiny** comprises three key components:

```
  ┌──────────────────────┐       ┌──────────────────────┐
  │  Dermoscopic Image   │       │   Clinical Metadata    │
  │     (224 × 224)      │       │ (age, sex, site, ...)  │
  └──────────┬───────────┘       └───────────┬───────────┘
             │                               │
     ┌───────▼────────┐              ┌───────▼────────┐
     │  ConvNeXt-Tiny  │              │  MLP Encoder    │
     │  (pretrained)   │              │  Linear+GeLU+Dp │
     └───────┬────────┘              └───────┬────────┘
             │                               │
     ┌───────▼────────┐                      │
     │ Spatial Pyramid│  16 regions          │  128-dim
     │   Projection   │  × 256-dim           │
     │  (4 × 4 grid)  │                      │
     └───────┬────────┘                      │
             └──────────────┬───────────────┘
                            │
                   ┌────────▼────────┐
                   │  Cross-Modal     │
                   │  Attention       │
                   │  Pooling (CMAP)  │
                   └────────┬────────┘
                            │  384-dim fused
                   ┌────────▼────────┐
                   │    Classifier     │
                   │   MLP → 8-class  │
                   └──────────────────┘
```

Although Transformer architectures have demonstrated success in NLP tasks, we argue that their direct application to medical image analysis faces three fundamental limitations: (a) global attention mechanisms struggle to capture localized pathological features in dermoscopic images, (b) quadratic computational complexity with increasing resolution, and (c) overfitting tendencies in small-sample scenarios. **ConvNeXt-Tiny was chosen as a modern convolutional network** that not only retains convolution's advantage in capturing local visual details but also enhances global feature association and long-range information modeling capabilities.

### Cross-Modal Attention Pooling (CMAP)

The CMAP module is the core innovation of this work. Given image region features **R ∈ ℝ^(B×N×DR)** and metadata features **M ∈ ℝ^(B×DM)** , the mechanism proceeds as follows:

**Step 1 — Metadata Expansion**: The metadata feature M is expanded to match the number of image regions N:

> M_exp ∈ ℝ^(B×N×DM)

**Step 2 — Feature Concatenation**: For each region *i*, the image region feature **rᵢ** is concatenated with the expanded metadata feature **m_exp,ᵢ**:

> **Cᵢ** = [**rᵢ** ; **m_exp,ᵢ**] ∈ ℝ^(DR+DM) &nbsp;&nbsp;&nbsp; *(Equation 4)*

**Step 3 — Attention Score Calculation**: The combined features are fed into a two-layer MLP with GELU activation:

> **sᵢ** = AttentionNet(**Cᵢ**) &nbsp;&nbsp;&nbsp; *(Equation 5)*
>
> where AttentionNet = Linear(DR+DM, H_dim) → GELU → Linear(H_dim, 1)

**Step 4 — Softmax Normalization**: Raw scores are normalized across all regions:

> **wᵢ** = exp(sᵢ) / Σⱼ exp(sⱼ) &nbsp;&nbsp;&nbsp; *(Equation 6)*

**Step 5 — Weighted Feature Aggregation**: A single fused feature vector **F** is produced by weighted summation:

> **F** = Σᵢ **wᵢ** · **Cᵢ** &nbsp;&nbsp;&nbsp; *(Equation 7)*

**Step 6 — Post-processing**: The fused feature undergoes **FC → BatchNorm → GELU → Dropout(0.3)** to yield the final representation for downstream classification.

This attention mechanism ensures the model can **selectively focus on the most relevant visual cues conditioned on patient metadata**, leading to more robust and interpretable decision-making. For example, the model learns to focus on lesion core areas and edge transition zones for young patients (where benign nevi typically present with color uniformity and regular edges), while attending to scattered texture abnormalities across aging skin for elderly patients.

### Loss Function

To address the class imbalance in skin lesion classification (e.g., NV: 12,875 vs. DF: 239), we employ a **weighted cross-entropy loss**:

> L_CE = − Σ_c **w_c** · ŷ_c · log(p_c) &nbsp;&nbsp;&nbsp; *(Equation 1)*

where **w_c** = total_samples / class_samples (inverse frequency weighting), amplifying the contribution of minority classes during training.

Additionally, **label smoothing regularization** is applied to prevent overconfidence:

> ŷ_c^smooth = (1 − ε) · ŷ_c + ε / C &nbsp;&nbsp;&nbsp; *(Equation 2)*

where ε = 0.1 is the smoothing factor, and C = 8 is the number of classes.

---

## Dataset

The **ISIC 2019** dataset contains **25,331 dermoscopic images** across **8 diagnostic categories** with severe class imbalance:

| Abbr. | Class                    | Samples | Frequency |
|-------|--------------------------|---------|-----------|
| MEL   | Melanoma                 | 4,522   | 17.85%    |
| NV    | Melanocytic Nevus        | 12,875  | 50.83%    |
| BCC   | Basal Cell Carcinoma     | 3,323   | 13.12%    |
| AK    | Actinic Keratosis        | 867     | 3.42%     |
| BKL   | Benign Keratosis         | 2,624   | 10.36%    |
| DF    | Dermatofibroma           | 239     | 0.94%     |
| VASC  | Vascular Lesion          | 253     | 1.00%     |
| SCC   | Squamous Cell Carcinoma  | 628     | 2.48%     |

**Setup**: Download from the [ISIC Archive](https://challenge.isic-archive.com/data/) and organize as:

```
data/
├── ISIC_2019_Training_GroundTruth.csv
├── ISIC_2019_Training_Metadata.csv
└── ISIC_2019_Training_Input/
    ├── ISIC_0000000.jpg
    ├── ISIC_0000001.jpg
    └── ...
```

### Data Preprocessing

- **Metadata**: Missing categorical values (gender, anatomic site) filled as "Unknown"; missing ages imputed with median; age Z-score normalized; categorical variables one-hot encoded (sex: 3D, anatomic site: 9D)
- **Training augmentation**: Random horizontal/vertical flips, ±20° rotation, random resized cropping (80%-100% scale), color jittering (±20% brightness, contrast, saturation)
- **Metadata noise**: Gaussian noise (σ = 0.1) injected into metadata during training for robustness
- **Train/Val split**: 80/20 stratified by class labels (random seed 42)

---

## Project Structure

```
skin-cancer-convnext-multimodal/
├── config.py                # Hyperparameters and path configuration
├── data_preprocessing.py    # ISIC dataset loading, metadata encoding, augmentation
├── model.py                 # ConvNeXt backbone, Spatial Pyramid, CMAP, full model
├── train_utils.py           # Training/validation loops, AdamW optimizer, LR scheduler
├── main.py                  # Training pipeline and 5-type result visualization
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
├── data/                    # ⬅ Place ISIC 2019 dataset here
│   └── .gitkeep
└── results/                 # Training outputs
    └── models/              # Saved model checkpoints (*.pth)
```

---

## Usage

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YuZhongFanXing/skin-cancer-convnext-multimodal.git
cd skin-cancer-convnext-multimodal

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download ISIC 2019 dataset and place under data/

# 4. Run training
python main.py
```

### Training Configuration

| Parameter           | Value                    |
|---------------------|--------------------------|
| Image Size          | 224 × 224                |
| Batch Size          | 64                       |
| Epochs              | 30                       |
| Optimizer           | AdamW (weight decay=3e-4)|
| Learning Rate       | 1e-4                     |
| LR Schedule         | Cosine annealing + 3-epoch linear warmup |
| Precision           | Mixed FP16 (if CUDA)     |
| Early Stopping      | Patience = 8             |
| Label Smoothing     | ε = 0.1                  |
| GPU                 | NVIDIA V100              |

### Results Output

After training, `results/` contains:

| File | Description |
|------|-------------|
| `models/best_model.pth` | Best checkpoint (model + optimizer + label encoder) |
| `training_history.png` | 4-panel: loss, accuracy, train attn heatmap, val attn heatmap |
| `confusion_matrix.png` | 8-class confusion matrix |
| `classification_report.csv` | Per-class precision, recall, F1, support |
| `attention_distribution.png` | Final attention weight bar charts |
| `validation_auc.png` | AUC curve over training epochs |

---

## Experimental Results

### Backbone Comparison (Image-only)

| Backbone         | Accuracy | AUC    | Precision | Recall | F1     |
|-----------------|----------|--------|-----------|--------|--------|
| ResNet50        | 0.7389   | 0.9215 | 0.5725    | 0.7450 | 0.6295 |
| VGG16           | 0.8244   | 0.9331 | 0.7624    | 0.7832 | 0.7722 |
| EfficientNet-B0 | 0.8488   | 0.9213 | 0.7303    | 0.8146 | 0.7617 |
| **ConvNeXt-Tiny** | **0.8769** | **0.9321** | **0.8612** | **0.8169** | **0.8372** |

ConvNeXt-Tiny leads comprehensively. Its advantage stems from: (1) modern convolution design capturing fine-grained local details (irregular edges, heterogeneous colors), and (2) enhanced global feature association for modeling overall lesion morphology.

### Fusion Strategy Ablation

| Fusion Strategy         | Accuracy | AUC    | Precision | Recall | F1     |
|------------------------|----------|--------|-----------|--------|--------|
| Simple Concatenation   | 0.8810   | 0.9516 | 0.8506    | 0.8278 | 0.8383 |
| **CMAP (This work)**  | **0.8909** | **0.9563** | **0.8732** | **0.8517** | **0.8619** |

### Analysis of CMAP Effectiveness

The ablation study demonstrates three key advantages of cross-modal attention over simple concatenation:

1. **Dynamic weight assignment**: Based on clinical metadata semantics (patient age, anatomical site), dynamic attention weights are assigned to different image regions, enabling the model to prioritize discriminative regions strongly correlated with metadata.
2. **Enhanced modal interaction**: Information interaction channels are constructed in the cross-modal "vision–metadata" space, allowing clinical prior knowledge to directly guide the screening and integration of visual features.
3. **Robustness improvement**: Through complementarity and verification of multi-modal information, errors caused by a single modality (e.g., imaging noise, metadata recording deviations) are reduced.

---

## Dependencies

| Package       | Version |
|---------------|---------|
| PyTorch       | ≥ 1.12  |
| torchvision   | ≥ 0.13  |
| timm          | ≥ 0.6   |
| numpy         | ≥ 1.21  |
| pandas        | ≥ 1.3   |
| scikit-learn  | ≥ 1.0   |
| opencv-python | ≥ 4.5   |
| matplotlib    | ≥ 3.5   |
| seaborn       | ≥ 0.11  |
| tqdm          | ≥ 4.62  |

```bash
pip install -r requirements.txt
```

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
               Universities (Grant No: 202510613070, 2682025ZTPY057) and
               Sichuan Medical Innovation Association (Grant No: YCH-KY-YCZD2024-302)}
}
```

---

## Acknowledgments

This work is supported by:
- Fundamental Research Funds for the Central Universities (Grant No: 202510613070, 2682025ZTPY057)
- Sichuan Medical Innovation Association (Grant No: YCH-KY-YCZD2024-302)

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

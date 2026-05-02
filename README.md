# ConvNeXt-Based Multimodal Skin Cancer Classification with Cross-Modal Attention Pooling

**Yuxin Hou** · Junjie Han · Xinyue Zha · Ruijie Xu · Kuizhang Zhao · **Guanfeng Wu\***

*School of Mathematics, Southwest Jiaotong University, Chengdu, China*

*PyTorch 2.0 · ISIC 2019 Dataset · Accuracy 89.1% · MIT License*

---

## Abstract

This study proposes a **multimodal skin cancer classification model** based on the **ConvNeXt** architecture and **cross-modal attention pooling (CMAP)** . To address the characteristics of image data and clinical metadata (age, gender, anatomical site) in the ISIC 2019 dataset, we designed a novel multimodal fusion strategy. The core innovations include:

1. Adopting a pre-trained **ConvNeXt-Tiny** model as the image feature extractor, leveraging its powerful visual representation capabilities -- it retains convolution's advantage in capturing local visual details while enhancing global feature association through modern design elements inspired by Transformers.
2. Introducing a **cross-modal attention pooling** mechanism that utilizes metadata information to guide the weight allocation of image regions, thereby achieving deep fusion of image features and metadata features.
3. Transforming image features into regional vector features through **spatial pyramid projection** to provide fine-grained input for the attention mechanism.

Experimental results on the ISIC 2019 dataset achieve **89.1% accuracy** and **0.862 macro-F1**, significantly outperforming simple feature concatenation baselines. Notably, for rare classes such as Actinic Keratosis (AK) and Vascular lesions (VASC), recall rates improved by **6-9%** .

---

## Motivation

Skin cancers represent the most prevalent categories of cancers diagnosed globally. The two most common types are basal cell carcinoma and squamous cell carcinoma, with melanoma being the third most prevalent. Melanoma is highly treatable when detected early; however, advanced melanoma can metastasize to internal organs, potentially resulting in mortality. An estimated **212,200 new melanoma cases** are expected in the U.S. in 2025 alone.

Barriers to healthcare access contribute to delayed detection. Patients experience substantial wait times -- ranging from **33.9 to 73.4 days** -- when consulting dermatologists. Unimodal frameworks are inherently constrained by morphological similarities between malignant melanomas and benign nevi. Clinical metadata provides complementary contextual information unobtainable through visual patterns alone, making **multimodal integration necessary** to address diagnostic bottlenecks.

---

## Related Work

### CNN-Based Approaches
- **VGG16**: 89.7% accuracy on HAM10000 dataset; excels in spatial feature learning. [1]
- **EfficientNet-B0**: 93.2% accuracy on ISIC 2020; reduces inference latency by 40% vs. larger variants. [2]
- **ResNet50**: 87.4% AUC on ISIC 2019; faces challenges with fine-grained lesion subtypes. [3]

### Multimodal Methods
Recent work combines image data with patient metadata (gender, age, lesion location) to improve classification. However, existing multimodal systems have significant shortcomings:
- Ignoring correlations between clinical and dermatoscopic images [4]
- Limited generalization ability across diagnostic tasks [5]
- Complexity in implementation and interpretation, limiting clinical application [6]

### Data Augmentation & Imbalance
SMOTE-based augmentation and preprocessing techniques (hair removal, lesion segmentation) have been proposed to address severe category imbalance in skin cancer datasets. [2]

---

## Method

### Architecture Overview

The proposed multimodal framework based on **ConvNeXt-Tiny** comprises three key components: (1) an image feature extraction branch, (2) a metadata processing branch, and (3) a cross-modal attention fusion module.

```
  Dermoscopic Image (224x224)        Clinical Metadata (age, sex, site...)
         |                                      |
  ConvNeXt-Tiny (pretrained)            MLP Encoder (Linear+GELU+Dropout)
         |                                      |
  Spatial Pyramid Projection                   |
   4x4 grid -> 16 regions x 256-dim            |  128-dim
         |                                      |
         +------------ concat -----------------+
                        |
              Cross-Modal Attention Pooling (CMAP)
              AttentionNet: Linear -> GELU -> Linear -> Softmax
              Weighted Sum -> FC -> BN -> GELU -> Dropout
                        |  384-dim fused
                   MLP Classifier
                        |  8-class output
```

Although Transformer architectures have demonstrated success in NLP tasks, their direct application to medical image analysis faces three fundamental limitations: (a) global attention mechanisms struggle to capture localized pathological features in dermoscopic images, (b) quadratic computational complexity with increasing resolution, and (c) overfitting tendencies in small-sample scenarios. **ConvNeXt-Tiny** was chosen as a modern convolutional network that retains convolution's advantage in capturing local visual details while enhancing global feature association and long-range information modeling capabilities.

### Cross-Modal Attention Pooling (CMAP) -- Core Innovation

The CMAP module is the key technical contribution. Given image region features **R** with shape (B, N, DR) and metadata features **M** with shape (B, DM):

**Step 1 -- Metadata Expansion**: M is expanded to match the number of image regions N, producing M_exp with shape (B, N, DM).

**Step 2 -- Feature Concatenation**: For each region i, the image region feature r_i is concatenated with the expanded metadata feature:

> C_i = [r_i ; m_exp,i]    (dimension: DR + DM)

**Step 3 -- Attention Score Calculation**: Combined features pass through a two-layer MLP with GELU activation:

> s_i = AttentionNet(C_i)
>
> where AttentionNet = Linear(DR+DM, hidden) -> GELU -> Linear(hidden, 1)

**Step 4 -- Softmax Normalization**: Raw scores normalized across regions:

> w_i = exp(s_i) / sum_j exp(s_j)

**Step 5 -- Weighted Aggregation**: A single fused feature vector F is produced:

> F = sum_i w_i * C_i

**Step 6 -- Post-processing**: Fused features undergo FC -> BatchNorm -> GELU -> Dropout(0.3) to yield the final representation for classification.

This attention mechanism ensures the model can **selectively focus on the most relevant visual cues conditioned on patient metadata**. For example, the model learns to focus on lesion core areas and edge transition zones for young patients (where benign nevi typically present with color uniformity and regular edges), while attending to scattered texture abnormalities across aging skin for elderly patients.

### Loss Function

To address the class imbalance (e.g., NV: 12,875 vs. DF: 239), we employ a **weighted cross-entropy loss**:

> L_CE = - sum_c w_c * y_c * log(p_c)

where w_c = total_samples / class_samples (inverse frequency weighting), amplifying minority class contributions during training.

Additionally, **label smoothing regularization** prevents overconfidence:

> y_c_smooth = (1 - epsilon) * y_c + epsilon / C

where epsilon = 0.1 and C = 8 (number of classes).

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

### Data Preparation

Download from the [ISIC Archive](https://challenge.isic-archive.com/data/) and organize as:

```
data/
├── ISIC_2019_Training_GroundTruth.csv
├── ISIC_2019_Training_Metadata.csv
└── ISIC_2019_Training_Input/
    ├── ISIC_0000000.jpg
    ├── ISIC_0000001.jpg
    └── ...
```

**Preprocessing details**:
- Metadata: missing categorical values (gender, anatomic site) filled as "Unknown"; missing ages imputed with median; age Z-score normalized; categorical variables one-hot encoded (sex: 3D, anatomic site: 9D).
- Training augmentation: random horizontal/vertical flips, +/-20 degree rotation, random resized cropping (80%-100% scale), color jittering (+/-20% brightness, contrast, saturation).
- Gaussian noise (std=0.1) injected into metadata during training for robustness.
- Train/val split: 80/20 stratified by class labels (random seed 42).

---

## Project Structure

```
skin-cancer-convnext-multimodal/
├── config.py                # Hyperparameters and path configuration
├── data_preprocessing.py    # ISIC dataset loading, metadata encoding, augmentation
├── model.py                 # ConvNeXt backbone, Spatial Pyramid, CMAP, full model
├── train_utils.py           # Training/validation loops, AdamW optimizer, LR scheduler
├── main.py                  # Training pipeline and result visualization (5 types)
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
├── data/                    # <-- Place ISIC 2019 dataset here
│   └── .gitkeep
└── results/                 # Training outputs (created at runtime)
    └── models/              # Saved model checkpoints (*.pth)
```

---

## Usage

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

| Parameter           | Value                     |
|---------------------|---------------------------|
| Image Size          | 224 x 224                 |
| Batch Size          | 64                        |
| Epochs              | 30                        |
| Optimizer           | AdamW (weight decay=3e-4) |
| Learning Rate       | 1e-4                      |
| LR Schedule         | Cosine annealing + 3-epoch linear warmup |
| Mixed Precision     | FP16 (if CUDA available)  |
| Early Stopping      | Patience = 8              |
| Label Smoothing     | epsilon = 0.1             |
| GPU                 | NVIDIA V100               |

### Results Output

After training, the `results/` directory will contain:

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

### Backbone Comparison (Image-only Baselines)

| Backbone          | Accuracy | AUC    | Precision | Recall | F1     |
|-------------------|----------|--------|-----------|--------|--------|
| ResNet50          | 0.7389   | 0.9215 | 0.5725    | 0.7450 | 0.6295 |
| VGG16             | 0.8244   | 0.9331 | 0.7624    | 0.7832 | 0.7722 |
| EfficientNet-B0   | 0.8488   | 0.9213 | 0.7303    | 0.8146 | 0.7617 |
| **ConvNeXt-Tiny** | **0.8769** | **0.9321** | **0.8612** | **0.8169** | **0.8372** |

ConvNeXt-Tiny leads comprehensively across accuracy, precision, recall, and F1. Its advantage stems from: (1) modern convolution design capturing fine-grained local details (irregular edges, heterogeneous colors), and (2) enhanced global feature association for modeling overall lesion morphology. As noted in the paper, VGG16 excels at shallow texture features but lacks high-level semantic encoding; EfficientNet-B0 balances efficiency well but cannot match ConvNeXt-Tiny's ability to model global morphological associations.

### Fusion Strategy Ablation (Multimodal)

| Fusion Strategy        | Accuracy | AUC    | Precision | Recall | F1     |
|------------------------|----------|--------|-----------|--------|--------|
| Simple Concatenation   | 0.8810   | 0.9516 | 0.8506    | 0.8278 | 0.8383 |
| **CMAP (This work)**   | **0.8909** | **0.9563** | **0.8732** | **0.8517** | **0.8619** |

### Why CMAP Outperforms Simple Concatenation

1. **Dynamic weight assignment**: Clinical metadata semantics (patient age, anatomical site) dynamically assign attention weights to different image regions, enabling the model to prioritize discriminative regions strongly correlated with metadata -- unlike simple concatenation where "vision" and "metadata" features coexist loosely without inter-modal interaction.

2. **Enhanced modal interaction**: Information interaction channels are explicitly constructed in the cross-modal space, allowing clinical prior knowledge to directly guide the screening and integration of visual features.

3. **Robustness improvement**: Through complementarity and verification of multi-modal information, errors caused by a single modality (e.g., imaging noise, metadata recording deviations) are reduced -- the attention mechanism provides a natural "verification" pathway between modalities.

---

## Dependencies

| Package       | Version |
|---------------|---------|
| PyTorch       | >= 1.12 |
| torchvision   | >= 0.13 |
| timm          | >= 0.6  |
| numpy         | >= 1.21 |
| pandas        | >= 1.3  |
| scikit-learn  | >= 1.0  |
| opencv-python | >= 4.5  |
| matplotlib    | >= 3.5  |
| seaborn       | >= 0.11 |
| tqdm          | >= 4.62 |

```bash
pip install -r requirements.txt
```

---

## References

1. A. Adebiyi et al., "Comparison of three deep learning models in accurate classification of 770 dermoscopy skin lesion images," *AMIA Summits on Translational Science Proceedings*, 2024.
2. M. Harahap et al., "Skin cancer classification using efficientnet architecture," *Bulletin of Electrical Engineering and Informatics*, 2024.
3. S. A. Khan et al., "A fine-tuned efficientnet-b1 framework for multiclass skin cancer classification," in *International Conference on Generative Artificial Intelligence*, Springer, 2024.
4. Y. Wei and L. Ji, "Multi-modal bilinear fusion with hybrid attention mechanism for multi-label skin lesion classification," *Multimedia Tools and Applications*, 2024.
5. K. Wang et al., "Skin malignancy classification using patients' skin images and meta-data: Multimodal fusion for improving fairness," in *MIDL*, 2024.
6. J. Dhar et al., "Multimodal fusion learning with dual attention for medical imaging," in *IEEE/CVF WACV*, 2025.

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

This project is licensed under the MIT License -- see [LICENSE](LICENSE) for details.

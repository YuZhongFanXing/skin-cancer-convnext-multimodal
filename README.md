# ConvNeXt-Based Multimodal Skin Cancer Classification with Cross-Modal Attention Pooling

**Yuxin Hou** · Junjie Han · Xinyue Zha · Ruijie Xu · Kuizhang Zhao · **Guanfeng Wu\***

*School of Mathematics, Southwest Jiaotong University, Chengdu, China*

---

## Abstract

This study proposes a **multimodal skin cancer classification model** based on the **ConvNeXt** architecture and **cross-modal attention pooling (CMAP)**, aiming to improve the accuracy and efficiency of skin cancer diagnosis. To address the characteristics of image data and clinical metadata (age, gender, anatomical site) in the ISIC 2019 dataset, we designed a novel multimodal fusion strategy. The core innovations include:

1. Adopting a pre-trained **ConvNeXt-Tiny** model as the image feature extractor, leveraging its powerful visual representation capabilities -- it retains convolution's advantage in capturing local visual details while enhancing global feature association through modern design elements inspired by Transformers.
2. Introducing a **cross-modal attention pooling** mechanism that utilizes metadata information to guide the weight allocation of image regions, thereby achieving deep fusion of image features and metadata features.
3. Transforming image features into regional vector features through **spatial pyramid projection** to provide fine-grained input for the attention mechanism.

Experimental results on the ISIC 2019 dataset achieve **89.1% accuracy** and **0.862 macro-F1**, significantly outperforming traditional concatenation-based multimodal fusion methods. Notably, for rare classes such as Actinic Keratosis (AK) and Vascular lesions (VASC), recall rates improved by **6-9%**.

---

## Motivation

Skin cancers represent the most prevalent categories of cancers diagnosed globally. The two most common types are basal cell carcinoma and squamous cell carcinoma, with melanoma being the third most prevalent. Melanoma is highly treatable when detected early; however, advanced melanoma can metastasize to internal organs, potentially resulting in mortality. An estimated **212,200 new melanoma cases** are expected in the U.S. in 2025 alone. Additionally, projections indicate 8,430 individuals will die from melanoma in the United States in 2025.

Barriers to healthcare access and availability contribute to delays in detection and impede effective treatment. Patients experience substantial wait times -- ranging from **33.9 to 73.4 days** -- when consulting dermatologists regarding evolving nevi. Insufficient numbers and uneven distribution of dermatologists exacerbate late detection. Unimodal frameworks are inherently constrained by morphological similarities, particularly the visual indistinguishability between malignant melanomas and benign nevi. Clinical metadata (age, gender, anatomical site) provides complementary contextual information unobtainable through visual patterns alone, making **multimodal integration necessary** to address diagnostic bottlenecks.

---

## Related Work

### CNN-Based Skin Cancer Classification

Recent advances in convolutional neural networks have significantly enhanced automated skin cancer diagnosis:
- **VGG16**: 89.7% accuracy on HAM10000 dataset; excels in spatial feature learning through deep hierarchical feature extraction [1].
- **EfficientNet-B0**: 93.2% accuracy on ISIC 2020; reduces inference latency by 40% compared to larger variants, suitable for resource-constrained clinical environments [2].
- **ResNet50**: 87.4% AUC on ISIC 2019; faces challenges with fine-grained lesion subtypes despite residual learning advantages [3].

### Multimodal Skin Cancer Classification

Researchers have explored combining image data with patient metadata (gender, age, lesion location) to improve classification accuracy. However, existing multimodal systems have significant shortcomings:
- Ignoring correlations between clinical and dermatoscopic images [4]
- Limited generalization ability across diagnostic tasks [5]
- Complexity in implementation and interpretation, limiting clinical application [6]

### Data Augmentation and Imbalance Treatment

Skin cancer datasets often suffer from severe category imbalance. SMOTE-based data augmentation and preprocessing techniques (hair removal, lesion segmentation, feature extraction) have been proposed to improve model input quality [2][7].

---

## Method

### Architecture Overview

We propose a multimodal framework based on **ConvNeXt-Tiny**, comprising three key components: (1) an image feature extraction branch, (2) a metadata processing branch, and (3) a cross-modal attention fusion module.

```
  Dermoscopic Image (224x224)          Clinical Metadata (age, sex, site...)
         |                                         |
  ConvNeXt-Tiny (pretrained)              MLP Encoder
  features_only, out_indices=[3]          Linear(meta_dim->256)->GELU->Drop(0.3)
  768-dim feature maps                    ->Linear(256->128)->GELU->Drop(0.2)
         |                                         |
  Spatial Pyramid Projection                      |
  AdaptiveAvgPool2d(4x4) -> 1x1 Conv              |  128-dim
  16 regions x 256-dim                             |
         |                                         |
         +--------------- concat ------------------+
                          |
              Cross-Modal Attention Pooling (CMAP)
              Metadata expanded: M -> M_exp (B,N,DM)
              Combined: C_i = [r_i ; m_exp,i]
              Score: s_i = AttentionNet(C_i)  [Linear->GELU->Linear]
              Weight: w_i = softmax(s_i)
              Fused: F = sum(w_i * C_i) -> FC -> BN -> GELU -> Dropout(0.3)
                          |  384-dim fused
                    MLP Classifier
                    Linear(384->256)->BN->GELU->Dropout(0.4)->Linear(256->8)
                          |  8-class output
```

Although Transformer architectures have demonstrated success in NLP tasks, we argue that their direct application to medical image analysis faces three fundamental limitations: (a) global attention mechanisms struggle to capture localized pathological features in dermoscopic images, (b) quadratic computational complexity with increasing resolution, and (c) overfitting tendencies in small-sample scenarios. **ConvNeXt-Tiny** was chosen as a modern convolutional network that not only retains convolution's advantage in capturing local visual details but also enhances global feature association and long-range information modeling capabilities through designs inspired by Transformers.

### Cross-Modal Attention Pooling (CMAP) -- Core Innovation

The CMAP module is the key technical contribution. Given image region features **R** in R^(BxNxDR) (B=batch, N=16 regions, DR=256) and metadata features **M** in R^(BxDM) (DM=128):

**Step 1 -- Metadata Expansion**: M is expanded to match N image regions, producing **M_exp** in R^(BxNxDM).

**Step 2 -- Feature Concatenation** (Equation 4): For each region i, concatenate image region feature r_i with expanded metadata m_exp,i:

> **C_i = [r_i ; m_exp,i]**  in  R^(DR+DM)

**Step 3 -- Attention Score Calculation** (Equation 5): Combined features pass through a two-layer MLP with GELU activation:

> **s_i = AttentionNet(C_i)**
>
> where AttentionNet = Linear(DR+DM, hidden) -> GELU -> Linear(hidden, 1)

**Step 4 -- Softmax Normalization** (Equation 6): Raw scores normalized across all N regions:

> **w_i = exp(s_i) / sum_j exp(s_j)**

**Step 5 -- Weighted Feature Aggregation** (Equation 7): A single fused feature vector F is produced by weighted summation:

> **F = sum_i w_i * C_i**  in  R^(DR+DM)

**Step 6 -- Post-processing**: Fused features undergo FC -> BatchNorm -> GELU -> Dropout(0.3) to yield the final representation for downstream classification.

This attention mechanism ensures the model can **selectively focus on the most relevant visual cues conditioned on patient metadata**, leading to more robust and interpretable decision-making. For example, for a 30-year-old female patient with an upper extremity lesion, the model assigns high attention weights (0.144-0.166) to the lesion core and edge transition zones while ignoring irrelevant skin areas, leveraging the benign prior of "young female + upper extremity" to classify correctly as Nevus (NV). For an 80-year-old male with a head/neck lesion, attention scatters across multiple texture-abnormal regions, combining the prior of "elderly male + head/neck" to identify seborrheic keratosis (BKL) based on aging skin patterns.

### Loss Function

To address the severe class imbalance in skin lesion classification (e.g., NV: 12,875 vs. DF: 239), we employ a **weighted cross-entropy loss** (Equation 1):

> L_CE = - sum_c **w_c** * y_c * log(p_c)

where **w_c** = total_samples / class_samples (inverse frequency weighting), amplifying the contribution of minority classes during training.

Additionally, **label smoothing regularization** is applied to prevent overconfidence (Equation 2):

> y_c^smooth = (1 - epsilon) * y_c + epsilon / C

where epsilon = 0.1 is the smoothing factor and C = 8 classes.

---

## Dataset

The **ISIC 2019** dataset is a large-scale dermoscopic image collection curated for skin lesion analysis, containing **25,331 images** across **8 diagnostic categories** with severe class imbalance:

| Class | Name                      | Samples | Frequency |
|-------|---------------------------|---------|-----------|
| MEL   | Melanoma                  | 4,522   | 17.85%    |
| NV    | Melanocytic Nevus         | 12,875  | 50.83%    |
| BCC   | Basal Cell Carcinoma      | 3,323   | 13.12%    |
| AK    | Actinic Keratosis         | 867     | 3.42%     |
| BKL   | Benign Keratosis          | 2,624   | 10.36%    |
| DF    | Dermatofibroma            | 239     | 0.94%     |
| VASC  | Vascular Lesion           | 253     | 1.00%     |
| SCC   | Squamous Cell Carcinoma   | 628     | 2.48%     |

### Setup

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

### Data Preprocessing

- **Metadata**: Missing categorical values (gender, anatomic site) filled as "Unknown"; missing ages imputed with median; age Z-score normalized; categorical variables one-hot encoded (sex: 3D, anatomic site: 9D)
- **Training augmentation**: Random horizontal/vertical flips, +/-20 degree rotation, random resized cropping (80%-100% scale), color jittering (+/-20% brightness, contrast, saturation) to simulate clinical imaging variations
- **Metadata noise**: Gaussian noise (sigma=0.1) injected during training for robustness
- **Train/validation split**: 80/20 stratified by class labels (random seed 42)

---

## Project Structure

```
skin-cancer-convnext-multimodal/
├── config.py                # Hyperparameters, paths, device selection
├── data_preprocessing.py    # ISIC dataset loading, metadata encoding, augmentation
├── model.py                 # ConvNeXt backbone, SpatialPyramid, CMAP, full model
├── train_utils.py           # Training/validation loops, AdamW, cosine LR scheduler
├── main.py                  # Training pipeline + 5-type result visualization
├── requirements.txt         # Python dependencies
├── .gitignore
├── .gitattributes
├── LICENSE                  # MIT License
├── data/                    # <-- Place ISIC 2019 dataset here
│   └── .gitkeep
└── results/                 # Training outputs (created at runtime)
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

After training, `results/` will contain:

| File | Description |
|------|-------------|
| `models/best_model.pth` | Best checkpoint (model + optimizer + label encoder + class weights) |
| `training_history.png` | 4-panel: loss, accuracy, train attention heatmap, val attention heatmap |
| `confusion_matrix.png` | 8-class confusion matrix |
| `classification_report.csv` | Per-class precision, recall, F1-score, support |
| `attention_distribution.png` | Final attention weight distribution bar charts |
| `validation_auc.png` | Validation AUC curve over training epochs |

---

## Experimental Results

### Backbone Comparison (Image-only Baselines)

| Backbone          | Accuracy | AUC    | Precision | Recall | F1     |
|-------------------|----------|--------|-----------|--------|--------|
| ResNet50          | 0.7389   | 0.9215 | 0.5725    | 0.7450 | 0.6295 |
| VGG16             | 0.8244   | 0.9331 | 0.7624    | 0.7832 | 0.7722 |
| EfficientNet-B0   | 0.8488   | 0.9213 | 0.7303    | 0.8146 | 0.7617 |
| **ConvNeXt-Tiny** | **0.8769** | **0.9321** | **0.8612** | **0.8169** | **0.8372** |

ConvNeXt-Tiny leads comprehensively in accuracy, precision, recall, and F1-score. Its advantage stems from: (1) modern convolution design capturing fine-grained local details (irregular edges, heterogeneous colors), and (2) enhanced global feature association for modeling overall lesion morphology and spatial relationships with surrounding tissues. VGG16 has a slight edge in AUC (0.9331) by capturing shallow texture features, but performs mediocrely in other metrics due to lack of efficient high-level semantic encoding. EfficientNet-B0 balances parameter count and computational efficiency well but cannot match ConvNeXt-Tiny's ability to model global morphological associations.

### Fusion Strategy Ablation (Multimodal)

| Fusion Strategy        | Accuracy | AUC    | Precision | Recall | F1     |
|------------------------|----------|--------|-----------|--------|--------|
| Simple Concatenation   | 0.8810   | 0.9516 | 0.8506    | 0.8278 | 0.8383 |
| **CMAP (This work)**   | **0.8909** | **0.9563** | **0.8732** | **0.8517** | **0.8619** |

### Why CMAP Outperforms Simple Concatenation

Simple feature concatenation only performs dimension concatenation of visual features and clinical metadata features, lacking active modeling of inter-modal correlations -- the two modalities exist in a state of loose coexistence and cannot fully exert the role of metadata guiding visual feature focus. In contrast, CMAP achieves performance breakthroughs through:

1. **Dynamic weight assignment**: Based on clinical metadata semantics (patient age, anatomical site), dynamic attention weights are assigned to different image regions, enabling the model to prioritize discriminative regions strongly correlated with metadata.
2. **Enhanced modal interaction**: Information interaction channels are constructed in the cross-modal vision-metadata space, allowing clinical prior knowledge to directly guide the screening and integration of visual features.
3. **Robustness improvement**: Through complementarity and verification of multi-modal information, errors caused by a single modality (e.g., imaging noise, metadata recording deviations) are reduced, enhancing adaptability to complex cases.

Rare class (AK, VASC) recall rates improved by **6-9%**, demonstrating the model's robustness in handling class imbalance.

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

[1] S. B. Ummapure et al., "Skin cancer classification using VGG-16 and GoogLeNet CNN models," *International Journal of Computer Applications*, 2023.

[2] M. Harahap et al., "Skin cancer classification using EfficientNet architecture," *Bulletin of Electrical Engineering and Informatics*, 2024.

[3] S. A. Khan et al., "A fine-tuned EfficientNet-B1 framework for multiclass skin cancer classification," in *International Conference on Generative Artificial Intelligence*, Springer, 2024.

[4] Y. Wei and L. Ji, "Multi-modal bilinear fusion with hybrid attention mechanism for multi-label skin lesion classification," *Multimedia Tools and Applications*, 2024.

[5] K. Wang et al., "Skin malignancy classification using patients' skin images and meta-data: Multimodal fusion for improving fairness," in *MIDL*, 2024.

[6] J. Dhar et al., "Multimodal fusion learning with dual attention for medical imaging," in *IEEE/CVF WACV*, 2025.

[7] G. H. Dagnaw et al., "Skin cancer classification using vision transformers and explainable artificial intelligence," *Journal of Medical Artificial Intelligence*, 2024.

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

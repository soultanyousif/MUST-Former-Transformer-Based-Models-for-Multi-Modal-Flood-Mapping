# MUST-Former: Transformer-Based Models for Multi-Modal Flood Mapping

A family of transformer-based architectures for robust flood detection from Sentinel-1 (SAR) and Sentinel-2 (optical) satellite imagery, built on the SegFormer backbone. This repository contains the official implementation and training code for the MUST-Former models evaluated on the Sen1Floods11 dataset.

## Overview

MUST-Former provides five model variants covering three input configurations:

| Model | Input | Fusion Strategy |
|-------|-------|-----------------|
| MUST-Former-SAR | Sentinel-1 (VV, VH) | -- |
| MUST-Former-Optical | Sentinel-2 (13 bands + NDVI/NDWI) | -- |
| MUST-Former-Projector | SAR + Optical | Additive 1x1 conv projection |
| MUST-Former-CrossAttn | SAR + Optical | Bi-directional cross-attention |
| MUST-Former-PCA | SAR + Optical | Channel standardization + projection |

> **Note:** MUST stands for **Misr University for Science and Technology**. Former refers to the transformer architecture.

## Key Results

### Sen1Floods11 Test Split

| Model | Water IoU | Mean IoU |
|-------|-----------|----------|
| MUST-Former-SAR | 0.6782 | 0.8125 |
| MUST-Former-Optical | 0.8097 | 0.8901 |
| MUST-Former-Projector | 0.8112 | 0.8912 |
| MUST-Former-CrossAttn | 0.7624 | 0.8628 |
| **MUST-Former-PCA** | **0.8326** | **0.9035** |

### Unseen Bolivia Split (Geographic Generalization)

| Model | Water IoU | Mean IoU |
|-------|-----------|----------|
| MUST-Former-SAR | 0.6339 | 0.7751 |
| MUST-Former-Optical | 0.8143 | 0.8896 |
| MUST-Former-Projector | 0.7881 | 0.8749 |
| MUST-Former-CrossAttn | 0.7719 | 0.8647 |
| **MUST-Former-PCA** | **0.8205** | **0.8932** |



## Visualizations

### Qualitative Results

For qualitative segmentation results on the Sen1Floods11 Test Split, see:

```
Qualitative results/Qualitative results.png
```

### Confusion Matrices

For per-model confusion matrices, see:

```
confusion matrix/confusion_matrices.png
```

### Regional Performance Heatmaps

For performance of the 5 models across all geographic regions, see:

```
Regional Heatmap/regional_heatmaps.png
```

### Benchmark Comparisons

**SAR vs. Baseline**

| Model | Split | Mean IoU |
|-------|-------|----------|
| FCNN-ResNet50 (Baseline) | Test | 0.3125 |
| **MUST-Former-SAR** | **Test** | **0.8125** |
| FCNN-ResNet50 (Baseline) | Bolivia | 0.3524 |
| **MUST-Former-SAR** | **Bolivia** | **0.7751** |

**Optical vs. Foundation Models (Test Split)**

| Metric | MUST-Former-Optical (~27.35M) | Prithvi-100M (100M) | Prithvi-CAFE (45.5M) |
|--------|-------------------------------|---------------------|----------------------|
| Water IoU | 80.97% | 81.98% | 83.41% |
| Mean IoU | 89.01% | 89.59% | 90.50% |

**Optical vs. Foundation Models (Bolivia Split -- Transferability)**

| Metric | MUST-Former-Optical (~27.35M) | Prithvi-100M (100M) | Prithvi-CAFE (45.5M) |
|--------|-------------------------------|---------------------|----------------------|
| Water IoU | **81.43%** | 76.62% | 81.37% |
| Mean IoU | **88.96%** | 86.02% | 88.87% |

**Fusion vs. U-Net Baseline**

| Method | Test Water IoU | Test Mean IoU | Bolivia Water IoU | Bolivia Mean IoU |
|--------|----------------|---------------|-------------------|------------------|
| Early Fusion U-Net | 0.7871 | 0.8810 | 0.8123 | 0.8889 |
| MUST-Former-Projector | 0.8112 | 0.8958 | 0.7881 | 0.8810 |
| MUST-Former-CrossAttn | 0.7624 | 0.8652 | 0.7719 | 0.8713 |
| **MUST-Former-PCA** | **0.8326** | **0.9035** | **0.8205** | **0.8932** |

### Reproducibility (3 Seeds: 0, 42, 123)

| Model | Water IoU (Mean +/- Std) | mIoU (Mean +/- Std) |
|-------|--------------------------|---------------------|
| MUST-Former-SAR | 0.6674 +/- 0.0081 | 0.8059 +/- 0.0052 |
| MUST-Former-Optical | 0.7910 +/- 0.0185 | 0.8794 +/- 0.0106 |
| MUST-Former-Projector | 0.8090 +/- 0.0025 | 0.8900 +/- 0.0014 |
| **MUST-Former-PCA** | **0.8287 +/- 0.0045** | **0.9013 +/- 0.0025** |
| MUST-Former-CrossAttn | 0.7657 +/- 0.0028 | 0.8645 +/- 0.0014 |

## Architecture

### Fusion Strategies

**Projector Fusion:** Simple 1x1 convolution projects SAR and optical features independently to 3 channels each, fused via element-wise addition.

**Cross-Attention Fusion:** Bi-directional multi-head cross-attention between SAR and optical feature tokens. Uses spatial downsampling (16x) to manage memory, with FFN and layer normalization.

**PCA Fusion:** Channel-wise standardization (zero-mean, unit-variance per spatial location) followed by a 1x1 convolution to 3 channels. Prevents either modality from dominating due to arbitrary scale differences.

### Backbone

All variants use the **SegFormer MiT-B2** encoder (NVIDIA) with an MLP decoder head. Pre-trained ImageNet weights are used for the encoder.

- Docker will be added soon 

## Installation

```bash
pip install torch torchvision torchaudio
pip install transformers rasterio albumentations
pip install pandas matplotlib scikit-learn tqdm
```

## Dataset Setup

Download the [Sen1Floods11 dataset](https://github.com/cloudtostreet/Sen1Floods11) and organize as follows:

```
Sen1Floods11/
  data/
    flood_events/HandLabeled/
      S1Hand/          # Sentinel-1 VV, VH tiles
      S2Hand/          # Sentinel-2 13-band tiles
      LabelHand/       # Binary flood labels
      JRCWaterHand/    # JRC permanent water masks
  splits/flood_handlabeled/
    flood_train_data.csv
    flood_valid_data.csv
    flood_test_data.csv
    flood_bolivia_data.csv
  data/Sen1Floods11_Metadata.geojson
```

Update `PathConfig` in the code to point to your dataset location.

## Training

Each model is trained independently. The training pipeline includes:

- **Loss:** Hybrid Focal-Dice loss (`dice_weight=0.6`, `focal_alpha=0.75`, `focal_gamma=2.0`)
- **Optimizer:** AdamW (`lr=5e-5`, `weight_decay=1e-4`)
- **Scheduler:** Linear warmup (10 epochs) + Cosine annealing
- **Augmentation:** Horizontal/vertical flip, rotation, affine, brightness/contrast, Gaussian blur, elastic transform
- **Regional weighting:** Per-sample weights based on regional water ratio to handle class imbalance
- **AMP:** Automatic Mixed Precision training on NVIDIA GPUs

### Train a Single Model

```python
# Example: Train MUST-Former-PCA
model_name = 'fusion_pca'
config = TRAIN_CONFIG.copy()
config['model_name'] = model_name

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    train_df, val_df, test_df,
    batch_size=8, include_s1=True, include_s2=True
)

# Build model
model = get_model_by_name('fusion_pca').to(config['device'])

# Loss, optimizer, scheduler
loss_fn = HybridLoss(...).to(config['device'])
optimizer, scheduler = create_optimizer_scheduler(model, config)

# Training loop with early stopping
for epoch in range(1, config['epochs'] + 1):
    train_log, _ = train_epoch(...)
    if epoch % config['val_freq'] == 0:
        val_log, val_iou, per_region_iou = validate_epoch(...)
        # Save best model based on Water IoU
```

### Available Model Names

```python
MODEL_CONFIGS = {
    's1_only':        {'model_type': 's1_only', 'fusion_type': 'projector'},
    's2_only':        {'model_type': 's2_only', 'fusion_type': 'projector'},
    'fusion_projector': {'model_type': 'fusion', 'fusion_type': 'projector'},
    'fusion_attention': {'model_type': 'fusion', 'fusion_type': 'attention'},
    'fusion_pca':     {'model_type': 'fusion', 'fusion_type': 'pca'},
}
```

## Citation

If you use this code or the MUST-Former models in your research, please cite:

```bibtex
Will be added soon, 2 papers are on their way!
1- IEEE https://ufe.edu.eg/3scea2026/: It's accepted, and we are only waiting for the proceedings
title: Efficient Transformer Architecture Outperforms Foundation Models for Satellite-Based Flood Detection
2- THE Q2 Spanish journal Inteligencia Artificial https://journal.iberamia.org/index.php/intartif/index : our paper is accepted we minor revision
Title:MUST-Former: A Family of Transformer-Based Models for Robust Multi-Modal Flood Mapping from Satellite Data.
```
<img width="590" height="763" alt="image" src="https://github.com/user-attachments/assets/ad83b667-6f6f-497a-80c2-36a78df0dae5" />

<img width="1061" height="345" alt="image" src="https://github.com/user-attachments/assets/5c6a7ca8-dba1-47d8-b07a-901952411178" />


Additionally, cite the underlying datasets and backbone:

```bibtex
@INPROCEEDINGS{9150760,
  author={Bonafilia, Derrick and Tellman, Beth and Anderson, Tyler and Issenberg, Erica},
  booktitle={2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  title={Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1},
  year={2020},
  pages={835-845},
  doi={10.1109/CVPRW50498.2020.00113}
}

@inproceedings{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  booktitle={NeurIPS},
  year={2021}
}
```

## License

This project is released for academic and research use.

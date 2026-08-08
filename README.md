# 👁️ Glaucoma Segmentation & Progression Scoring using MobileNetV2 U-Net
### *Lightweight Deep Learning for Optic Disc & Cup Segmentation and Clinical CDR Estimation*

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![TensorFlow](https://img.shields.io/badge/Keras-TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Project Overview
Glaucoma is a leading cause of irreversible blindness worldwide. Early clinical diagnosis relies on precise segmentation of the **Optic Disc (OD)** and **Optic Cup (OC)** from retinal fundus photographs to compute the **Cup-to-Disc Ratio (CDR)** and evaluate neuroretinal rim loss.

This repository presents a high-accuracy, lightweight deep learning segmentation system built on an enhanced **MobileNetV2 U-Net** backbone with custom attention modules, uncertainty quantification, and clinical post-processing.

---

## 🔬 Key Architectural Innovations

```mermaid
flowchart LR
    Input[Retinal Fundus Image] --> ROI[High-Res Disc ROI Crop 1.2x]
    ROI --> CLAHE[CLAHE + Noise Augmentation]
    CLAHE --> Enc[MobileNetV2 Inverted Residual Encoder]
    Enc --> Attn[Attention Blocks: CBAM / MSCA / ECA]
    Attn --> Dec[U-Net / FPN Decoder + PPM]
    Dec --> MC[Monte Carlo Dropout Uncertainty Head]
    MC --> TTA[4-Pass Flip Test-Time Augmentation]
    TTA --> PP[Morphological Closing + Topological Disc/Cup Fill]
    PP --> Out[OD & OC Masks + CDR Score]
```

### 1. **Lightweight & Fast MobileNetV2 Backbone**
- Parameter-efficient inverted residual bottleneck layers with depthwise separable convolutions, enabling edge-device and real-time clinical deployment.

### 2. **Multi-Scale Attention Mechanisms**
- **CBAM (Convolutional Block Attention Module)**: Channel & spatial attention refinement on skip connections.
- **MSCA (Multi-Scale Convolutional Attention)** & **LBFR (Local-Boundary Feature Refinement)**: Sharp delineation of blurry optic cup borders.
- **PPM (Pyramid Pooling Module)**: Captures global anatomical context across multiple receptive fields.

### 3. **Advanced Loss Formulation & Uncertainty**
- **Focal EIoU Loss + Boundary Loss**: Solves severe class imbalance between cup, disc, and background pixels.
- **Monte Carlo (MC) Dropout**: Epistemic Bayesian uncertainty estimation providing clinicians with confidence heatmaps.

### 4. **Clinical Multi-Stage Pipeline**
- **High-Res ROI Extraction**: Dynamic 1.2x–1.5x optic disc bounding crop.
- **Post-Processing (PP)**: Largest connected component extraction, morphological closing, hole filling, and topological constraint enforcement ($\text{Cup} \subseteq \text{Disc}$).
- **Cosine Annealing ($T_0=15, T_{mult}=2$) & Label Smoothing (0.05)**.

---

## 📊 Benchmark Datasets & Quantitative Results

The models were evaluated across three major clinical fundus datasets:
- **REFUGE (Retinal Fundus Glaucoma Challenge)** — 1,200 clinical images.
- **Drishti-GS** — Gold-standard clinical ground truth.
- **RIM-ONE (v3)** — High-resolution optic nerve head imaging.

| Dataset / Variant | Optic Disc Dice ($\text{OD}$) | Optic Cup Dice ($\text{OC}$) | Mean IoU | Validation Loss |
| :--- | :--- | :--- | :--- | :--- |
| **REFUGE (Best Variant + TTA + PP)** | **0.9612** | **0.8521** | **0.7839** | **0.3421** |
| **Drishti-GS Benchmark** | **0.9740** | **0.8910** | **0.8120** | **0.3110** |
| **RIM-ONE Benchmark** | **0.9580** | **0.8460** | **0.7690** | **0.3580** |

---

## 📂 Repository Structure

```
├── Final models/                          # Benchmark evaluation notebooks & results
│   ├── Refuge - best.ipynb                # High-accuracy REFUGE model notebook
│   ├── Refuge - baseline.ipynb            # Baseline model comparison
│   ├── Drishti - both models.ipynb        # Drishti-GS evaluation
│   ├── Rim one - both models.ipynb        # RIM-ONE evaluation
│   ├── confusion_matrix.png               # Diagnostic classification matrix
│   ├── loss_accuracy.png                  # Training & validation convergence curves
│   └── progression.png                    # Clinical progression scoring curve
│
├── ROI_V8_boosted_phase4.py              # Full training pipeline (CLAHE + TTA + PP)
├── MC_dropout.py                          # Bayesian Monte Carlo uncertainty estimation
├── cbam.py                                # Channel & Spatial Attention module
├── FPN.py                                 # Feature Pyramid Network decoder
├── Focal_EIoU.py                          # Focal Efficient-IoU custom loss
└── deeper+focal+mc+msca+lbfr+ppm.py       # Full composite architecture
```

---

## 🚀 How to Run

### Installation
```bash
git clone https://github.com/RogMaverick18/glaucoma-segmentation-mobilenetv2.git
cd glaucoma-segmentation-mobilenetv2
pip install -r requirements.txt
```

### Model Evaluation / Inference
```bash
python ROI_V8_boosted_phase4.py
```

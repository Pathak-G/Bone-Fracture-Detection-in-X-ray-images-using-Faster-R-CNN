# Bone Fracture Detection in X-ray Images using Faster R-CNN

A deep learning project implementing Faster R-CNN with a ResNet-50 Feature Pyramid Network backbone to automatically detect bone fractures in X-ray images.

Built as part of the CS6482 Deep Reinforcement Learning module (MSc in Artificial Intelligence and Machine Learning, University of Limerick).

---

## Results

| Metric | Score |
|---|---|
| Mean Precision | 0.6104 ± 0.0222 |
| Mean Recall | 0.6601 ± 0.0174 |
| Mean F1 Score | 0.6337 ± 0.0039 |
| Mean mAP@0.5 | 0.6294 ± 0.0037 |

Evaluated using 3-fold cross-validation. Aggregate confusion matrix: TP=1378, FP=884, FN=710.

---

## Dataset

The [Bone Fracture Detection dataset](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project) by pkdarabi (Kaggle, CC BY 4.0) contains X-ray images annotated with YOLO-style polygon coordinates.

| Split | Images |
|---|---|
| Train | 3,631 |
| Validation | 348 |
| Test | 169 |
| **Total** | **4,148** |

After filtering images with empty label files: 1,804 training samples and 173 validation samples were used.

---

## Architecture

**Faster R-CNN** with a **ResNet-50 + FPN** backbone, pre-trained on MS COCO.

The pipeline:
1. ResNet-50 backbone extracts feature maps from the input X-ray image
2. Feature Pyramid Network (FPN) builds a multi-scale feature hierarchy (P2–P5)
3. Region Proposal Network (RPN) generates candidate fracture regions
4. RoI Align crops fixed-size features for each proposal
5. Box head + FastRCNNPredictor classifies regions and regresses bounding boxes

The classification head was replaced with a 2-class predictor (background / fracture).

---

## Preprocessing

- YOLO polygon annotations converted to axis-aligned bounding boxes (min/max x, y of polygon vertices)
- Pixel values normalised from [0, 255] to [0, 1]
- Images with empty label files excluded

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Framework | PyTorch |
| Epochs | 5 |
| Batch size | 2 |
| Optimiser | SGD (momentum=0.9, weight decay=5e-4) |
| Learning rate | 0.005 |
| LR scheduler | StepLR (step=3, gamma=0.1) |
| IoU threshold | 0.5 |
| Score threshold | 0.5 |
| Cross-validation | 3-fold |
| Backbone | ResNet-50 FPN (COCO pre-trained) |

---

## Experiments

### Backbone Comparison (ResNet-50 FPN vs MobileNetV3 FPN)

| Backbone | Final Loss |
|---|---|
| ResNet-50 FPN | **0.1065** |
| MobileNetV3 FPN | 0.3310 |

ResNet-50 FPN converged consistently across all 5 epochs. MobileNetV3 showed increasing loss from epoch 3 onwards, indicating underfitting — its lighter architecture lacks the representational capacity needed for subtle fracture features.

### Learning Rate Comparison (lr=0.005 vs lr=0.01)

| Learning Rate | Final Loss |
|---|---|
| 0.005 | **0.1008** |
| 0.01 | 0.1161 |

lr=0.005 achieves 14.7% lower final loss and more stable convergence, avoiding gradient overshooting.

---

## Setup

```bash
git clone https://github.com/Pathak-G/Bone-Fracture-Detection-in-X-ray-images-using-Faster-R-CNN>.git
cd Bone-Fracture-Detection-in-X-ray-images-using-Faster-R-CNN
pip install torch torchvision
pip install kaggle matplotlib numpy
```

To download the dataset, place your `kaggle.json` API key in `~/.kaggle/` then run the notebook — the dataset is downloaded programmatically.

---

## Usage

Open and run `code.ipynb` in Jupyter or Google Colab. The notebook covers:

- Dataset download and extraction
- Data visualisation and preprocessing
- Model training with 3-fold cross-validation
- Evaluation (precision, recall, F1, mAP@0.5, confusion matrix, PR curves)
- Backbone and learning rate experiments
- Inference visualisation

---

## References

- Ren et al. (2015) — Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
- He et al. (2016) — Deep Residual Learning for Image Recognition
- Lin et al. (2017) — Feature Pyramid Networks for Object Detection
- pkdarabi — Bone Fracture Detection Dataset (Kaggle, CC BY 4.0)

---

## Authors

- **Govind Pathak** (25020749) — data pipeline, model training, experiments
- **Ralston Dcruz** (25051032) — dataset analysis, evaluation metrics, architecture comparison

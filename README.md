# Intel Natural Scene Classification with Transfer Learning

An end-to-end deep learning project implementing **Transfer Learning** in PyTorch using a pre-trained **ResNet-18** architecture for natural scene classification on the Intel Image Classification dataset.

---

## 📂 Project Overview
This project explores how pre-trained convolutional neural networks (CNNs) can be adapted to new visual domains with minimal training data and computational overhead. The pipeline covers data preprocessing, custom augmentations, frozen feature extraction, gradual fine-tuning, and detailed error analysis.

---

## 📊 Dataset
* **Source:** Intel Image Classification Dataset (Kaggle)
* **Classes (6):** `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`
* **Data Splits:** 
  * Training Set: ~11,929 images (with robust augmentations)
  * Validation Set: ~2,105 images
  * Test Set: 3,000 images (held out for final evaluation)

---

## ⚙️ Methodology & Architecture

1. **Preprocessing & Augmentation:**
   * Standardized using ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
   * Applied `RandomResizedCrop`, `RandomHorizontalFlip`, and `ColorJitter` to prevent overfitting during training.

2. **Phase 1: Feature Extraction**
   * Loaded a pre-trained **ResNet-18** model and froze all feature extraction layers (`requires_grad = False`).
   * Replaced the final fully connected classification head with a custom linear layer mapping to the 6 target classes.
   * **Trainable Parameters:** 3,078 out of ~11.1M.
   * **Best Validation Accuracy:** **90.7%**.

3. **Phase 2: Fine-Tuning**
   * Unfroze the final residual block (`layer4`) to allow high-level feature adaptation.
   * Employed **differential learning rates** ($1\times 10^{-4}$ for the pretrained layer and $1\times 10^{-3}$ for the new classifier head).
   * **Trainable Parameters:** ~8.4M.
   * **Best Validation Accuracy:** **93.1%**.

---

## 📈 Results & Performance

* **Test Accuracy (Evaluated Once):** **92.5%**
* **Classification Metrics:**
  * High precision and recall across classes like `forest` and `sea` ($>0.95$ F1-score).
  * Identified weakest class: **Glacier** (Recall: $0.799$), frequently confused with **Mountain** due to overlapping high-altitude icy and rocky visual patterns.

---

## 🔍 Error Analysis & Insights
* **Most Instructive Error:** An image with the true label `buildings` was misclassified as `street` with high confidence. Upon inspection, the image captured a sidewalk storefront with a strong street-level perspective, proving that some dataset samples are **genuinely ambiguous** even to human observers.
* **Confusion Matrix Insights:** The highest inter-class confusion occurred between glaciers and mountains, reflecting shared structural textures in natural landscape photography.

---

## 🚀 Next Steps (Week 6)
Transitioning from standard benchmark datasets to domain-specific applications: developing an **eye-blink gesture classification** model for assistive robotics interfaces.

---

## 🛠️ Tech Stack
* **Language:** Python 3.13
* **Framework:** PyTorch & Torchvision
* **Environment:** Kaggle Notebooks (GPU T4 Accelerated)
* **Libraries:** NumPy, Matplotlib, Scikit-Learn, PIL

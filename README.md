# Food-101 Classification with EfficientNetB0

**Module 7 Final Project: Advanced Machine Learning & AI**

This project implements a complete deep learning pipeline for fine-grained food classification using transfer learning with EfficientNetB0 on the Food-101 dataset.

## 🎯 Project Overview

- **Dataset**: Food-101 (101,000 images, 101 food categories)
- **Model**: EfficientNetB0 with transfer learning
- **Training Strategy**: Two-phase approach (frozen base → selective fine-tuning)
- **Performance**: ~68-72% accuracy on 101-class classification
- **Authors**: Joshua Laubach, Krystonia Katoa, Felix Elias

## 📊 Key Results

| Model | Test Accuracy | Test F1 | Training Time | Parameters |
|-------|---------------|---------|---------------|------------|
| Baseline CNN | 0.99% | 0.0002 | - | ~2-5M |
| Custom CNN | 27.5% | 0.27 | - | ~2-4M |
| **EfficientNetB0** | **66-72%** | **0.66-0.72** | **~2-3 hours** | **4.18M (150K trainable)** |

## 🚀 Quick Start

### Prerequisites
Install the project dependencies with pip so the notebooks run with the expected libraries and versions:

```bash
pip install -r requirements.txt
```

### Running the Notebook
1. Clone this repository
2. Open `FinalProject.ipynb` in Jupyter Lab/Notebook
3. Run cells sequentially (dataset will auto-download via Hugging Face)
4. Training takes 2-3 hours on GPU, longer on CPU

## 📁 Project Structure

```
├── FinalProject.ipynb          # Main project notebook
├── Milestone_01.ipynb          # EDA and preprocessing experiments
├── Milestone_02.ipynb          # Model comparison experiments
├── README.md                   # This file
├── requirements.txt            # Python dependencies for notebooks
├── .gitignore                  # Git ignore rules
└── models/                     # Saved model checkpoints (excluded from git)
```

## 🧠 Model Architecture

**EfficientNetB0 Transfer Learning Pipeline:**
- **Base**: Frozen EfficientNetB0 (ImageNet pretrained)
- **Head**: GlobalAveragePooling2D → BatchNormalization → Dropout(0.4) → Dense(101, softmax)
- **Training**: Phase 1 (frozen base) → Phase 2 (fine-tune top 30 layers)

## 🔬 Technical Highlights

### Data Processing
- Stratified 80/10/10 train/val/test split
- Real-time augmentation: horizontal flips, rotations, zoom, contrast
- EfficientNet-specific preprocessing pipeline
- Class-balanced training (1,000 images per class)

### Training Strategy
1. **Phase 1**: Train classification head only (15 epochs, lr=1e-3)
2. **Phase 2**: Unfreeze top layers, fine-tune (10 epochs, lr=1e-4)
3. **Regularization**: Dropout, label smoothing, early stopping
4. **Optimization**: Adam optimizer with LR reduction on plateau

### Evaluation Metrics
- Top-1 and Top-5 accuracy
- Macro F1-score (handles class balance)
- Confusion matrix analysis
- Per-class performance breakdown

## 📈 Results Analysis

### Performance Evolution
- **Milestone 1**: Data exploration → identified 101-class balance, image quality issues
- **Milestone 2**: Model comparison → EfficientNetB0 achieved 66% (vs 27% custom CNN, 1% baseline)
- **Final Project**: Two-phase fine-tuning → improved to 68-72% accuracy

### Key Findings
- Transfer learning provides 66x improvement over baseline CNN
- Fine-tuning top layers adds 2-6 percentage points
- Most confusion occurs between visually similar foods (pasta types, similar desserts)
- Model achieves practical utility for real-world food recognition

## 🛠 Technical Requirements

- **Python**: 3.8+
- **TensorFlow**: 2.8+
- **RAM**: 8GB+ recommended
- **GPU**: Optional but recommended (CUDA compatible)
- **Storage**: ~50GB for full dataset (auto-downloaded)

## 📚 Academic Context

This project demonstrates:
- Transfer learning best practices for computer vision
- Systematic model comparison and evaluation
- Production-ready deep learning pipeline implementation
- Fine-grained image classification techniques

## 🤝 Contributing

This is an academic project for OMDS Module 7. The implementation follows best practices for:
- Reproducible research (fixed seeds, documented hyperparameters)
- Proper train/validation/test methodology
- Comprehensive evaluation and error analysis

## 📄 License

Academic use only. Dataset credit: Food-101 by Bossard et al.

## 🔗 References

- Food-101 Dataset: [https://www.vision.ee.ethz.ch/datasets_extra/food-101/](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)
- EfficientNet: Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
- Transfer Learning: Pan & Yang, "A Survey on Transfer Learning"

---

**Course**: OMDS Module 7 - Advanced Machine Learning & AI  
**Date**: December 2025  
**Institution**: Boston University

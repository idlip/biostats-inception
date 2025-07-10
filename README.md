# 🩺 Pneumonia Detection with InceptionV3

A deep learning model that detects pneumonia from chest X-rays with accuracy and sensitivity, providing reliable diagnostic support for healthcare professionals.

---

## 🌟 Key Features

- High Performance: 91% sensitivity ensures minimal missed pneumonia cases
- Fast Inference: Process X-rays in under 2 seconds
- Production Ready: Pre-trained weights included
- Class Imbalance Handling: Weighted training for balanced performance
- Interpretable Results: ROC curves and confusion matrices included

## 🚀 Quick Start
- Suggested to use uv python package manager
- Dataset source: https://www.kaggle.com/datasets/rijulshr/pneumoniamnist/
- Training code: ./train.py
    - Can be used in google colab for notebook purpose as well.

1. get UV package/project manager
```bash
# manually
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
pip install uv
# or via distro package manager
sudo apt install uv
```

2. Clone the repo
```bash
git clone --depth=1 https://github.com/idlip/biostats-inception biostats-inception
cd biostats-inception
```

3. Run with UV
```bash
uv add tensorflow-gpu  numpy matplotlib seaborn keras kagglehub scikit-learn matplotlib
uv run python3 train.py
```

## 📁 Data Format
```text
chest_xray_data.npz
├── train_images.npy  # Shape: (n_samples, height, width)
├── train_labels.npy  # Shape: (n_samples,)
├── val_images.npy
├── val_labels.npy
├── test_images.npy
└── test_labels.npy
```

## 🏗️ Model Architecture

- Base Model: InceptionV3 (pre-trained on ImageNet)
- Custom Layers:
- GlobalAveragePooling2D
- Dense(512) + Dropout(0.5)
- Dense(256) + Dropout(0.3)
- Dense(1, sigmoid)

## 🔮 Future Improvements

- Grad-CAM visualization for interpretability
- Multi-class classification (bacterial vs viral pneumonia)
- Ensemble with other architectures
- Mobile-optimized version

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- InceptionV3 architecture by Google
- Chest X-ray dataset providers
- TensorFlow and Keras teams


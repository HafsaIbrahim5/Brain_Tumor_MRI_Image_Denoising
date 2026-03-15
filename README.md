# 🧠 NeuroClean — Brain MRI Image Denoiser

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-00e5ff?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-7b2fff?style=for-the-badge)

**A deep learning–powered Brain Tumor MRI image denoising system using a Convolutional Autoencoder.**

[🔬 Live Demo](https://35bk9xhgboog7xkq8cmtcj.streamlit.app/) · [📓 Notebook](image-denoising.ipynb) · [📊 Results](#results)

</div>

---

## 📌 Project Overview

NeuroClean is a medical image restoration pipeline that removes noise from Brain Tumor MRI scans using a **Convolutional Autoencoder** trained on 100 grayscale MRI images.

MRI scanners introduce noise from thermal fluctuations, electromagnetic interference, and patient motion. This noise can obscure tumor boundaries and degrade diagnostic quality. NeuroClean learns to map noisy inputs back to clean reconstructions.

---

## 🧬 Model Architecture

```
INPUT  (128×128×1)
  ↓
Conv2D (32, 3×3, ReLU)  →  (128×128×32)
  ↓
MaxPooling2D (2×2)       →  (64×64×32)
  ↓
Conv2D (32, 3×3, ReLU)  →  (64×64×32)
  ↓
MaxPooling2D [BOTTLENECK] → (32×32×32)
  ↓
Conv2D (32, 3×3, ReLU)  →  (32×32×32)
  ↓
UpSampling2D (2×2)       →  (64×64×32)
  ↓
Conv2D (32, 3×3, ReLU)  →  (64×64×32)
  ↓
UpSampling2D (2×2)       →  (128×128×32)
  ↓
Conv2D (1, 3×3, Sigmoid) →  (128×128×1)
OUTPUT (128×128×1)
```

- **Trainable Parameters:** ~28,353
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Input Normalization:** [0, 1]

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Average PSNR | **20.71 dB** |
| Input Size | 128 × 128 px |
| Training Images | 100 Brain Tumor MRI |
| Noise Type | Gaussian (σ = 0.1) |

---

## 🚀 Streamlit App Features

| Feature | Description |
|---------|-------------|
| **Single Image Denoiser** | Upload, corrupt, denoise with live metrics |
| **Batch Processing** | Process multiple MRIs, export as ZIP |
| **5 Noise Types** | Gaussian, Salt & Pepper, Speckle, Poisson, Periodic |
| **5 Denoising Methods** | Autoencoder, Gaussian, Bilateral, Median, NLM, FFT |
| **Quality Metrics** | PSNR, SSIM, MSE, MAE with before/after comparison |
| **Histograms** | Pixel intensity distribution visualization |
| **Image Statistics** | Min, Max, Mean, Std, Median per image |
| **Export** | Download original, noisy, and denoised images |
| **Model Loading** | Load your own `.h5` autoencoder model |
| **Architecture View** | Interactive layer-by-layer model diagram |

---

## ⚙️ Installation & Running

```bash
# 1. Clone the repository
git clone https://github.com/HafsaIbrahim5/neuroclean-mri-denoiser.git
cd neuroclean-mri-denoiser

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## 🗂️ Repository Structure

```
neuroclean-mri-denoiser/
├── app.py                        # Streamlit application
├── image-denoising.ipynb         # Training notebook
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── dataset/                      # Brain tumor MRI images
│   ├── img_001.jpg
│   └── ...
└── neuroclean_autoencoder.h5     # Saved model (after training)
```

---

## 🧪 Training Your Own Model

```python
# In image-denoising.ipynb or train.py
autoencoder.fit(
    noisy_images,   # Corrupted inputs
    clean_images,   # Clean targets
    epochs=50,
    batch_size=16,
    validation_split=0.2
)
autoencoder.save('neuroclean_autoencoder.h5')
```

Then load the `.h5` file in the Streamlit app sidebar.

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **TensorFlow 2.x / Keras** — Model training & inference
- **Streamlit** — Web application
- **OpenCV** — Classical denoising filters
- **scikit-image** — PSNR & SSIM metrics
- **NumPy / Pillow / Matplotlib** — Image processing & visualization

---

## 👩‍💻 Author

**Hafsa Ibrahim** — AI / Machine Learning Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/HafsaIbrahim5)

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

> ⚠️ **Disclaimer:** This project is for educational and research purposes only. Not intended for clinical diagnosis without proper validation and regulatory approval.

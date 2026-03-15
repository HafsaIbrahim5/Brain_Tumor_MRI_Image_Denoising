import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import io
import base64
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import zipfile
import os
import time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroClean · Brain MRI Denoiser",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

:root {
  --bg-deep:    #020818;
  --bg-card:    #040f2a;
  --bg-hover:   #071530;
  --cyan:       #00e5ff;
  --cyan-dim:   #00b8cc;
  --teal:       #00ffd0;
  --violet:     #7b2fff;
  --violet-dim: #5a1fd0;
  --red-warn:   #ff4757;
  --text-bright:#e8f4fd;
  --text-mid:   #a8c8e8;
  --text-dim:   #5a7a9a;
  --border:     rgba(0,229,255,0.15);
  --glow:       0 0 20px rgba(0,229,255,0.25);
  --glow-strong:0 0 40px rgba(0,229,255,0.4);
}

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: var(--bg-deep);
    color: var(--text-bright);
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--cyan-dim); border-radius: 4px; }

/* Main container */
.main .block-container {
    padding: 1.5rem 2rem;
    max-width: 1400px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020c1e 0%, #030e22 100%);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* HERO BANNER */
.hero-banner {
    background: linear-gradient(135deg, #020c1e 0%, #030e22 40%, #050a18 100%);
    border: 1px solid rgba(0,229,255,0.2);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0,229,255,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: 20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(123,47,255,0.05) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.6rem;
    font-weight: 900;
    color: var(--cyan);
    text-shadow: 0 0 30px rgba(0,229,255,0.5);
    letter-spacing: 3px;
    margin: 0 0 0.3rem 0;
    line-height: 1.1;
}
.hero-subtitle {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.2rem;
    color: var(--text-mid);
    letter-spacing: 2px;
    font-weight: 500;
    margin-bottom: 1rem;
}
.hero-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.8rem;
}
.hero-tag {
    background: rgba(0,229,255,0.08);
    border: 1px solid rgba(0,229,255,0.25);
    color: var(--cyan);
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    letter-spacing: 1px;
}
.hero-tag.violet {
    background: rgba(123,47,255,0.08);
    border-color: rgba(123,47,255,0.3);
    color: #b07fff;
}
.hero-tag.teal {
    background: rgba(0,255,208,0.08);
    border-color: rgba(0,255,208,0.25);
    color: var(--teal);
}

/* METRIC CARDS */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}
.metric-card {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: rgba(0,229,255,0.35);
    box-shadow: var(--glow);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
}
.metric-card.violet::before {
    background: linear-gradient(90deg, transparent, var(--violet), transparent);
}
.metric-card.teal::before {
    background: linear-gradient(90deg, transparent, var(--teal), transparent);
}
.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-dim);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--cyan);
    line-height: 1;
}
.metric-card.violet .metric-value { color: #b07fff; }
.metric-card.teal .metric-value { color: var(--teal); }
.metric-unit {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.85rem;
    color: var(--text-dim);
    margin-left: 0.3rem;
}
.metric-desc {
    font-size: 0.8rem;
    color: var(--text-dim);
    margin-top: 0.4rem;
}

/* SECTION HEADERS */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 1.5rem 0 1rem 0;
}
.section-title {
    font-family: 'Orbitron', monospace;
    font-size: 1rem;
    font-weight: 700;
    color: var(--cyan);
    letter-spacing: 2px;
    text-transform: uppercase;
}
.section-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

/* CARDS */
.info-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.info-card h3 {
    font-family: 'Orbitron', monospace;
    font-size: 0.9rem;
    color: var(--cyan);
    margin: 0 0 0.6rem 0;
    letter-spacing: 1px;
}
.info-card p {
    font-size: 0.95rem;
    color: var(--text-mid);
    line-height: 1.6;
    margin: 0;
}

/* COMPARISON IMAGES */
.img-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.img-panel-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: var(--cyan);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

/* SIDEBAR LOGO */
.sidebar-logo {
    font-family: 'Orbitron', monospace;
    font-size: 1.3rem;
    font-weight: 900;
    color: var(--cyan);
    text-shadow: 0 0 15px rgba(0,229,255,0.4);
    text-align: center;
    padding: 0.5rem;
    margin-bottom: 0.2rem;
}
.sidebar-tagline {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.8rem;
    color: var(--text-dim);
    text-align: center;
    letter-spacing: 2px;
    margin-bottom: 1.5rem;
}

/* AUTHOR CARD */
.author-card {
    background: linear-gradient(135deg, rgba(0,229,255,0.05), rgba(123,47,255,0.05));
    border: 1px solid rgba(0,229,255,0.2);
    border-radius: 12px;
    padding: 1.2rem;
    margin-top: 1rem;
}
.author-name {
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem;
    color: var(--cyan);
    margin-bottom: 0.3rem;
    font-weight: 700;
}
.author-title {
    font-size: 0.8rem;
    color: var(--text-dim);
    margin-bottom: 0.8rem;
}
.social-link {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(0,229,255,0.08);
    border: 1px solid rgba(0,229,255,0.2);
    color: var(--cyan) !important;
    font-size: 0.78rem;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    text-decoration: none !important;
    margin: 0.2rem;
    font-family: 'Share Tech Mono', monospace;
    transition: all 0.2s;
}
.social-link:hover {
    background: rgba(0,229,255,0.15);
    box-shadow: 0 0 10px rgba(0,229,255,0.2);
}

/* PIPELINE STEPS */
.pipeline-step {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 0.8rem 0;
    border-bottom: 1px solid rgba(0,229,255,0.07);
}
.step-num {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem;
    font-weight: 900;
    color: rgba(0,229,255,0.2);
    min-width: 40px;
    line-height: 1;
}
.step-content h4 {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    color: var(--text-bright);
    margin: 0 0 0.2rem 0;
}
.step-content p {
    font-size: 0.85rem;
    color: var(--text-dim);
    margin: 0;
}

/* ARCHITECTURE DIAGRAM */
.arch-layer {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin: 0.3rem 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.85rem;
}
.arch-layer.encoder { border-left: 3px solid var(--cyan); }
.arch-layer.bottleneck { border-left: 3px solid var(--violet); }
.arch-layer.decoder { border-left: 3px solid var(--teal); }
.arch-name { color: var(--text-bright); font-weight: 600; }
.arch-shape { font-family: 'Share Tech Mono', monospace; color: var(--text-dim); font-size: 0.8rem; }
.arch-params { color: var(--cyan); font-family: 'Share Tech Mono', monospace; font-size: 0.75rem; }

/* BADGES */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1px;
}
.badge-cyan { background: rgba(0,229,255,0.12); color: var(--cyan); border: 1px solid rgba(0,229,255,0.3); }
.badge-violet { background: rgba(123,47,255,0.12); color: #b07fff; border: 1px solid rgba(123,47,255,0.3); }
.badge-teal { background: rgba(0,255,208,0.1); color: var(--teal); border: 1px solid rgba(0,255,208,0.25); }
.badge-red { background: rgba(255,71,87,0.12); color: #ff6b7a; border: 1px solid rgba(255,71,87,0.3); }

/* STATUS BAR */
.status-bar {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1rem;
    border-radius: 8px;
    font-size: 0.85rem;
    margin: 0.5rem 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
}
.status-bar.info { background: rgba(0,229,255,0.07); border: 1px solid rgba(0,229,255,0.2); color: var(--cyan); }
.status-bar.success { background: rgba(0,255,208,0.07); border: 1px solid rgba(0,255,208,0.2); color: var(--teal); }
.status-bar.warning { background: rgba(255,71,87,0.07); border: 1px solid rgba(255,71,87,0.2); color: #ff6b7a; }

/* Streamlit widget overrides */
.stSlider [data-testid="stSliderThumb"] { background: var(--cyan) !important; }
.stSelectbox select, .stMultiSelect { border-color: var(--border) !important; }
div[data-testid="stTabs"] button { font-family: 'Rajdhani', sans-serif; font-weight: 600; letter-spacing: 1px; }
.stButton > button {
    background: linear-gradient(135deg, rgba(0,229,255,0.1), rgba(0,229,255,0.05));
    border: 1px solid rgba(0,229,255,0.4);
    color: var(--cyan);
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    letter-spacing: 2px;
    font-weight: 700;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    transition: all 0.3s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,229,255,0.2), rgba(0,229,255,0.1));
    box-shadow: 0 0 20px rgba(0,229,255,0.3);
}
.stDownloadButton > button {
    background: linear-gradient(135deg, rgba(0,255,208,0.12), rgba(0,255,208,0.06));
    border: 1px solid rgba(0,255,208,0.4);
    color: var(--teal);
    font-family: 'Orbitron', monospace;
    font-size: 0.72rem;
    letter-spacing: 2px;
    border-radius: 8px;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--bg-card);
    border: 1px dashed rgba(0,229,255,0.3);
    border-radius: 12px;
}

/* Expander */
.streamlit-expanderHeader {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    color: var(--text-mid);
}

/* Divider */
hr { border-color: var(--border); }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────


def add_gaussian_noise(image: np.ndarray, std: float = 0.1) -> np.ndarray:
    """Add Gaussian noise to image."""
    noise = np.random.normal(0, std, image.shape)
    noisy = np.clip(image.astype(np.float32) / 255.0 + noise, 0, 1)
    return (noisy * 255).astype(np.uint8)


def add_salt_pepper_noise(image: np.ndarray, amount: float = 0.05) -> np.ndarray:
    """Add salt & pepper noise."""
    noisy = image.copy()
    total = image.size
    n_salt = int(total * amount * 0.5)
    n_pepper = int(total * amount * 0.5)
    # Salt
    coords = [np.random.randint(0, i - 1, n_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255
    # Pepper
    coords = [np.random.randint(0, i - 1, n_pepper) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0
    return noisy


def add_speckle_noise(image: np.ndarray, std: float = 0.1) -> np.ndarray:
    """Add speckle (multiplicative) noise."""
    img_f = image.astype(np.float32) / 255.0
    noise = np.random.normal(0, std, image.shape)
    noisy = np.clip(img_f + img_f * noise, 0, 1)
    return (noisy * 255).astype(np.uint8)


def add_poisson_noise(image: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Add Poisson noise."""
    img_f = image.astype(np.float32) / 255.0
    vals = len(np.unique(img_f))
    vals = 2 ** np.ceil(np.log2(vals)) * scale
    noisy = np.random.poisson(img_f * vals) / float(vals)
    return (np.clip(noisy, 0, 1) * 255).astype(np.uint8)


def add_periodic_noise(
    image: np.ndarray, frequency: float = 30.0, amplitude: float = 0.15
) -> np.ndarray:
    """Add periodic sinusoidal noise (simulates scanner artifacts)."""
    img_f = image.astype(np.float32) / 255.0
    h, w = image.shape[:2]
    x = np.arange(w)
    noise_1d = amplitude * np.sin(2 * np.pi * frequency * x / w)
    noise_2d = np.tile(noise_1d, (h, 1))
    if len(image.shape) == 3:
        noise_2d = noise_2d[:, :, np.newaxis]
    noisy = np.clip(img_f + noise_2d, 0, 1)
    return (noisy * 255).astype(np.uint8)


def denoise_gaussian_filter(image: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Classical Gaussian blur denoising."""
    if len(image.shape) == 3 and image.shape[2] == 1:
        img = image[:, :, 0]
        result = cv2.GaussianBlur(img, (0, 0), sigma)
        return result[:, :, np.newaxis]
    return cv2.GaussianBlur(image, (0, 0), sigma)


def denoise_bilateral(
    image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75
) -> np.ndarray:
    """Bilateral filter – edge-preserving denoising."""
    if len(image.shape) == 3 and image.shape[2] == 1:
        img = image[:, :, 0]
        result = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
        return result[:, :, np.newaxis]
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def denoise_median(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Median filter – excellent for salt & pepper."""
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    if len(image.shape) == 3 and image.shape[2] == 1:
        img = image[:, :, 0]
        result = cv2.medianBlur(img, ksize)
        return result[:, :, np.newaxis]
    return cv2.medianBlur(image, ksize)


def denoise_nlm(image: np.ndarray, h: float = 10) -> np.ndarray:
    """Non-local Means denoising."""
    if len(image.shape) == 3 and image.shape[2] == 1:
        img = image[:, :, 0]
        result = cv2.fastNlMeansDenoising(img, None, h, 7, 21)
        return result[:, :, np.newaxis]
    return cv2.fastNlMeansDenoising(image, None, h, 7, 21)


def denoise_wavelet_like(image: np.ndarray, threshold: float = 20) -> np.ndarray:
    """Simple frequency-domain thresholding (simulates wavelet denoising)."""
    if len(image.shape) == 3 and image.shape[2] == 1:
        img = image[:, :, 0].astype(np.float32)
    else:
        img = image.astype(np.float32)
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    magnitude = np.abs(dft_shift)
    dft_shift[magnitude < threshold] = 0
    img_back = np.fft.ifft2(np.fft.ifftshift(dft_shift)).real
    result = np.clip(img_back, 0, 255).astype(np.uint8)
    if len(image.shape) == 3 and image.shape[2] == 1:
        return result[:, :, np.newaxis]
    return result


def compute_metrics(original: np.ndarray, processed: np.ndarray):
    """Compute PSNR, SSIM, MSE, MAE between two images."""
    orig_f = original.astype(np.float64) / 255.0
    proc_f = processed.astype(np.float64) / 255.0
    if orig_f.shape != proc_f.shape:
        proc_f = cv2.resize(proc_f.squeeze(), (orig_f.shape[1], orig_f.shape[0]))
        if len(orig_f.shape) == 3:
            proc_f = proc_f[:, :, np.newaxis]

    squeeze_orig = orig_f.squeeze()
    squeeze_proc = proc_f.squeeze()

    mse = np.mean((squeeze_orig - squeeze_proc) ** 2)
    mae = np.mean(np.abs(squeeze_orig - squeeze_proc))
    psnr = (
        peak_signal_noise_ratio(squeeze_orig, squeeze_proc, data_range=1.0)
        if mse > 0
        else float("inf")
    )
    ssim = structural_similarity(squeeze_orig, squeeze_proc, data_range=1.0)
    return {"PSNR": psnr, "SSIM": ssim, "MSE": mse, "MAE": mae}


def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img)


def np_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return Image.fromarray(arr)


def img_to_bytes(img_arr: np.ndarray, fmt: str = "PNG") -> bytes:
    pil = np_to_pil(img_arr)
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return buf.getvalue()


def get_image_stats(img: np.ndarray) -> dict:
    flat = img.flatten().astype(np.float32)
    return {
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "median": float(np.median(flat)),
    }


def build_autoencoder():
    """Build the Convolutional Autoencoder architecture from the notebook."""
    inp = Input(shape=(128, 128, 1), name="input")
    # Encoder
    x = Conv2D(32, (3, 3), activation="relu", padding="same", name="enc_conv1")(inp)
    x = MaxPooling2D((2, 2), padding="same", name="enc_pool1")(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same", name="enc_conv2")(x)
    encoded = MaxPooling2D((2, 2), padding="same", name="bottleneck")(x)
    # Decoder
    x = Conv2D(32, (3, 3), activation="relu", padding="same", name="dec_conv1")(encoded)
    x = UpSampling2D((2, 2), name="dec_up1")(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same", name="dec_conv2")(x)
    x = UpSampling2D((2, 2), name="dec_up2")(x)
    decoded = Conv2D(1, (3, 3), activation="sigmoid", padding="same", name="output")(x)
    model = tf.keras.Model(inp, decoded, name="ConvAutoencoder")
    return model


def autoencoder_denoise(image: np.ndarray, model) -> np.ndarray:
    """Denoise using the loaded autoencoder model."""
    img_f = image.astype(np.float32) / 255.0
    if img_f.ndim == 2:
        img_f = img_f[:, :, np.newaxis]
    resized = cv2.resize(img_f, (128, 128))
    resized = resized[:, :, np.newaxis] if resized.ndim == 2 else resized
    inp = np.expand_dims(resized, axis=0)
    pred = model.predict(inp, verbose=0)[0]
    # Resize back if needed
    h, w = image.shape[:2]
    out = cv2.resize(pred.squeeze(), (w, h))
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def make_histogram_fig(images: dict) -> plt.Figure:
    """Plot pixel intensity histograms for multiple images."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#040f2a")
    ax.set_facecolor("#020818")
    colors = {"Original": "#00e5ff", "Noisy": "#ff4757", "Denoised": "#00ffd0"}
    for label, img in images.items():
        flat = img.flatten().astype(np.float32)
        ax.hist(
            flat,
            bins=64,
            alpha=0.6,
            color=colors.get(label, "#aaa"),
            label=label,
            density=True,
            histtype="stepfilled",
        )
    ax.set_xlabel("Pixel Intensity", color="#a8c8e8", fontsize=9)
    ax.set_ylabel("Density", color="#a8c8e8", fontsize=9)
    ax.tick_params(colors="#5a7a9a", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#0d2040")
    ax.legend(
        facecolor="#040f2a", edgecolor="#1a3a5a", labelcolor="#a8c8e8", fontsize=8
    )
    ax.grid(True, alpha=0.1, color="#1a3a5a")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "model" not in st.session_state:
    st.session_state.model = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🧠 NeuroClean</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-tagline">BRAIN MRI DENOISER</div>', unsafe_allow_html=True
    )
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠  Home",
            "🔬  Single Image",
            "📦  Batch Processing",
            "🏗️  Architecture",
            "📊  About",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "**⚙️ MODEL**", help="Load a pre-trained .h5 model or use classical methods"
    )
    model_file = st.file_uploader(
        "Load Autoencoder (.h5)", type=["h5", "keras"], key="model_upload"
    )
    if model_file:
        try:
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
                tmp.write(model_file.read())
                tmp_path = tmp.name
            import tensorflow as tf

            st.session_state.model = tf.keras.models.load_model(tmp_path)
            st.session_state.model_loaded = True
            st.markdown(
                '<div class="status-bar success">✓ MODEL LOADED</div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.markdown(
                f'<div class="status-bar warning">✗ LOAD FAILED</div>',
                unsafe_allow_html=True,
            )
    else:
        if st.session_state.model_loaded:
            st.markdown(
                '<div class="status-bar success">✓ MODEL ACTIVE</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-bar info">ℹ CLASSICAL MODE</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(
        """
    <div class="author-card">
        <div class="author-name">Hafsa Ibrahim</div>
        <div class="author-title">AI / Machine Learning Engineer</div>
        <a class="social-link" href="https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/" target="_blank">💼 LinkedIn</a>
        <a class="social-link" href="https://github.com/HafsaIbrahim5" target="_blank">🐙 GitHub</a>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────
if page == "🏠  Home":
    st.markdown(
        """
    <div class="hero-banner">
        <div class="hero-title">NEUROCLEAN</div>
        <div class="hero-subtitle">BRAIN TUMOR MRI · CONVOLUTIONAL AUTOENCODER · IMAGE DENOISING</div>
        <div class="hero-tags">
            <span class="hero-tag">TensorFlow 2.x</span>
            <span class="hero-tag">Keras</span>
            <span class="hero-tag teal">Convolutional Autoencoder</span>
            <span class="hero-tag violet">Brain Tumor MRI</span>
            <span class="hero-tag">PSNR · SSIM</span>
            <span class="hero-tag teal">Image Restoration</span>
            <span class="hero-tag">Deep Learning</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(
            """
        <div class="section-header">
            <span class="section-title">PROJECT OVERVIEW</span>
            <div class="section-line"></div>
        </div>
        <div class="info-card">
            <h3>🎯 WHAT IS THIS?</h3>
            <p>NeuroClean is a deep learning–powered image restoration system designed specifically for <strong>Brain Tumor MRI scans</strong>.
            It uses a <strong>Convolutional Autoencoder</strong> trained to map noisy grayscale MRI images back to their clean originals,
            enabling radiologists and researchers to work with higher-quality diagnostic images.</p>
        </div>
        <div class="info-card">
            <h3>🧬 THE PROBLEM</h3>
            <p>MRI scanners inevitably introduce noise during acquisition — from thermal fluctuations, electromagnetic interference, and patient motion.
            This noise degrades image quality, potentially obscuring critical anatomical features and tumor boundaries.
            Reliable denoising is therefore a clinical necessity, not just an aesthetic improvement.</p>
        </div>
        <div class="info-card">
            <h3>⚡ THE SOLUTION</h3>
            <p>A <strong>Convolutional Autoencoder</strong> learns a compressed, noise-invariant representation of MRI structure in its bottleneck layer.
            The decoder then reconstructs the clean image from this representation, effectively separating signal from noise.
            Trained on 100 Brain Tumor MRI images with synthetic Gaussian noise corruption.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="section-header">
            <span class="section-title">PIPELINE</span>
            <div class="section-line"></div>
        </div>
        <div class="info-card">
        <div class="pipeline-step">
            <div class="step-num">01</div>
            <div class="step-content">
                <h4>Load MRI Images</h4>
                <p>100 grayscale brain tumor scans, resized to 128×128 px, normalized to [0,1]</p>
            </div>
        </div>
        <div class="pipeline-step">
            <div class="step-num">02</div>
            <div class="step-content">
                <h4>Inject Noise</h4>
                <p>Gaussian noise (σ=0.1) added to clean images to create noisy training pairs</p>
            </div>
        </div>
        <div class="pipeline-step">
            <div class="step-num">03</div>
            <div class="step-content">
                <h4>Train Autoencoder</h4>
                <p>Conv2D encoder compresses + MaxPool, then Conv2DTranspose decoder reconstructs</p>
            </div>
        </div>
        <div class="pipeline-step">
            <div class="step-num">04</div>
            <div class="step-content">
                <h4>Evaluate Quality</h4>
                <p>PSNR ≈ 20.71 dB • SSIM measures structural fidelity restoration</p>
            </div>
        </div>
        <div class="pipeline-step">
            <div class="step-num">05</div>
            <div class="step-content">
                <h4>Deploy & Compare</h4>
                <p>Interactive comparison of CNN autoencoder vs classical methods</p>
            </div>
        </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="section-header">
            <span class="section-title">MODEL RESULTS</span>
            <div class="section-line"></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">Average PSNR</div>
                <div class="metric-value">20.71<span class="metric-unit">dB</span></div>
                <div class="metric-desc">Peak Signal-to-Noise Ratio</div>
            </div>
            <div class="metric-card teal">
                <div class="metric-label">Input Size</div>
                <div class="metric-value" style="font-size:1.3rem">128<span class="metric-unit">×128</span></div>
                <div class="metric-desc">Grayscale MRI resolution</div>
            </div>
        </div>
        <div class="metric-row">
            <div class="metric-card violet">
                <div class="metric-label">Training Set</div>
                <div class="metric-value">100<span class="metric-unit">imgs</span></div>
                <div class="metric-desc">Brain tumor MRI scans</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Noise Type</div>
                <div class="metric-value" style="font-size:1rem; margin-top:0.2rem">GAUSS</div>
                <div class="metric-desc">σ = 0.1 standard deviation</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# PAGE: SINGLE IMAGE
# ─────────────────────────────────────────────
elif page == "🔬  Single Image":
    st.markdown(
        """
    <div class="hero-banner" style="padding:1.2rem 2rem">
        <div class="hero-title" style="font-size:1.6rem">SINGLE IMAGE DENOISER</div>
        <div class="hero-subtitle" style="font-size:0.95rem">UPLOAD · CORRUPT · RESTORE · COMPARE</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Controls ─────────────────────────────
    col_ctrl, col_main = st.columns([1, 3])

    with col_ctrl:
        st.markdown(
            '<div class="info-card"><h3>🎛️ CONTROLS</h3>', unsafe_allow_html=True
        )

        uploaded = st.file_uploader(
            "Upload MRI Image",
            type=["png", "jpg", "jpeg", "bmp", "tif"],
            key="single_upload",
        )
        st.markdown("**Noise Type**")
        noise_type = st.selectbox(
            "",
            [
                "Gaussian",
                "Salt & Pepper",
                "Speckle",
                "Poisson",
                "Periodic (Scanner Artifact)",
            ],
            label_visibility="collapsed",
        )
        noise_level = st.slider("Noise Level", 0.01, 0.5, 0.15, 0.01)

        st.markdown("**Denoising Method**")
        methods_available = [
            "Gaussian Filter",
            "Bilateral Filter (Edge-Preserving)",
            "Median Filter",
            "Non-Local Means (NLM)",
            "Frequency Domain (FFT)",
        ]
        if st.session_state.model_loaded:
            methods_available.insert(0, "✨ Autoencoder (Deep Learning)")

        method = st.selectbox("", methods_available, label_visibility="collapsed")

        if method == "Gaussian Filter":
            m_sigma = st.slider("Sigma", 0.5, 5.0, 1.5, 0.1)
        elif method == "Bilateral Filter (Edge-Preserving)":
            b_sigma_c = st.slider("Sigma Color", 10, 150, 75, 5)
            b_sigma_s = st.slider("Sigma Space", 10, 150, 75, 5)
        elif method == "Median Filter":
            med_k = st.slider("Kernel Size", 3, 11, 5, 2)
        elif method == "Non-Local Means (NLM)":
            nlm_h = st.slider("h (Filter Strength)", 3, 30, 10, 1)
        elif method == "Frequency Domain (FFT)":
            fft_thr = st.slider("Threshold", 1, 100, 20, 1)

        process_btn = st.button("⚡ PROCESS IMAGE")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_main:
        if uploaded is None:
            st.markdown(
                """
            <div class="info-card" style="text-align:center; padding:3rem 2rem; border-style:dashed;">
                <div style="font-size:3rem; margin-bottom:1rem">🧠</div>
                <div style="font-family:'Orbitron',monospace; color:var(--cyan); font-size:0.9rem; letter-spacing:2px;">
                    UPLOAD AN MRI IMAGE TO BEGIN
                </div>
                <div style="color:var(--text-dim); font-size:0.85rem; margin-top:0.5rem;">
                    Supports PNG, JPG, BMP, TIFF · Grayscale or RGB
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            pil_img = Image.open(uploaded).convert("L")
            orig_np = pil_to_np(pil_img)

            # Apply noise
            if noise_type == "Gaussian":
                noisy_np = add_gaussian_noise(orig_np, std=noise_level)
            elif noise_type == "Salt & Pepper":
                noisy_np = add_salt_pepper_noise(orig_np, amount=noise_level)
            elif noise_type == "Speckle":
                noisy_np = add_speckle_noise(orig_np, std=noise_level)
            elif noise_type == "Poisson":
                noisy_np = add_poisson_noise(orig_np, scale=max(0.1, 1 - noise_level))
            else:
                noisy_np = add_periodic_noise(
                    orig_np, frequency=30 * noise_level * 10, amplitude=noise_level
                )

            # Denoise
            denoised_np = None
            if process_btn:
                with st.spinner("Processing..."):
                    if method == "✨ Autoencoder (Deep Learning)":
                        denoised_np = autoencoder_denoise(
                            noisy_np, st.session_state.model
                        )
                    elif method == "Gaussian Filter":
                        denoised_np = denoise_gaussian_filter(noisy_np, sigma=m_sigma)
                    elif method == "Bilateral Filter (Edge-Preserving)":
                        denoised_np = denoise_bilateral(
                            noisy_np, sigma_color=b_sigma_c, sigma_space=b_sigma_s
                        )
                    elif method == "Median Filter":
                        denoised_np = denoise_median(noisy_np, ksize=med_k)
                    elif method == "Non-Local Means (NLM)":
                        denoised_np = denoise_nlm(noisy_np, h=nlm_h)
                    elif method == "Frequency Domain (FFT)":
                        denoised_np = denoise_wavelet_like(noisy_np, threshold=fft_thr)
                st.session_state["denoised_single"] = denoised_np
            elif "denoised_single" in st.session_state:
                denoised_np = st.session_state["denoised_single"]

            # Display images
            if denoised_np is not None:
                ic1, ic2, ic3 = st.columns(3)
                imgs_to_show = [
                    ("ORIGINAL", orig_np, ""),
                    ("NOISY", noisy_np, noise_type.upper()),
                    (
                        "DENOISED",
                        denoised_np.squeeze() if denoised_np.ndim == 3 else denoised_np,
                        method.replace("✨ ", "").upper(),
                    ),
                ]
                for col_, (label, arr, badge) in zip([ic1, ic2, ic3], imgs_to_show):
                    with col_:
                        disp = arr.squeeze() if arr.ndim == 3 else arr
                        st.markdown(
                            f'<div class="img-panel-label">{label}</div>',
                            unsafe_allow_html=True,
                        )
                        st.image(disp, use_container_width=True, clamp=True)
                        if badge:
                            st.markdown(
                                f'<div style="text-align:center;"><span class="badge badge-cyan">{badge}</span></div>',
                                unsafe_allow_html=True,
                            )

                # Metrics
                den_sq = denoised_np.squeeze() if denoised_np.ndim == 3 else denoised_np
                m_noisy = compute_metrics(orig_np, noisy_np)
                m_denoised = compute_metrics(orig_np, den_sq)

                st.markdown(
                    """
                <div class="section-header">
                    <span class="section-title">QUALITY METRICS</span>
                    <div class="section-line"></div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                mc1, mc2, mc3, mc4 = st.columns(4)
                metrics_display = [
                    ("PSNR", m_denoised["PSNR"], m_noisy["PSNR"], "dB", "cyan"),
                    ("SSIM", m_denoised["SSIM"], m_noisy["SSIM"], "", "violet"),
                    ("MSE", m_denoised["MSE"], m_noisy["MSE"], "", "teal"),
                    ("MAE", m_denoised["MAE"], m_noisy["MAE"], "", "cyan"),
                ]
                for col_, (name, val_d, val_n, unit, style) in zip(
                    [mc1, mc2, mc3, mc4], metrics_display
                ):
                    with col_:
                        delta = (
                            val_d - val_n if name in ["PSNR", "SSIM"] else val_n - val_d
                        )
                        arrow = "▲" if delta > 0 else "▼"
                        delta_color = "var(--teal)" if delta > 0 else "var(--red-warn)"
                        st.markdown(
                            f"""
                        <div class="metric-card {style}">
                            <div class="metric-label">{name}</div>
                            <div class="metric-value" style="font-size:1.4rem">{val_d:.4f}<span class="metric-unit">{unit}</span></div>
                            <div style="font-size:0.75rem; color:{delta_color}; font-family:'Share Tech Mono',monospace; margin-top:0.3rem">
                                {arrow} {abs(delta):.4f} vs noisy
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                # Histogram
                with st.expander("📊 Pixel Intensity Histograms"):
                    hist_imgs = {
                        "Original": orig_np,
                        "Noisy": noisy_np,
                        "Denoised": den_sq,
                    }
                    fig = make_histogram_fig(hist_imgs)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                # Image statistics
                with st.expander("📐 Image Statistics"):
                    sc1, sc2, sc3 = st.columns(3)
                    for col_, (lbl, arr) in zip(
                        [sc1, sc2, sc3],
                        [
                            ("Original", orig_np),
                            ("Noisy", noisy_np),
                            ("Denoised", den_sq),
                        ],
                    ):
                        stats = get_image_stats(arr)
                        with col_:
                            st.markdown(f"**{lbl}**")
                            for k, v in stats.items():
                                st.markdown(f"`{k.upper()}:` **{v:.2f}**")

                # Download
                st.markdown(
                    """
                <div class="section-header">
                    <span class="section-title">EXPORT</span>
                    <div class="section-line"></div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
                dl1, dl2, dl3 = st.columns(3)
                with dl1:
                    st.download_button(
                        "⬇ DOWNLOAD ORIGINAL",
                        img_to_bytes(orig_np),
                        "original.png",
                        "image/png",
                    )
                with dl2:
                    st.download_button(
                        "⬇ DOWNLOAD NOISY",
                        img_to_bytes(noisy_np),
                        "noisy.png",
                        "image/png",
                    )
                with dl3:
                    st.download_button(
                        "⬇ DOWNLOAD DENOISED",
                        img_to_bytes(den_sq),
                        "denoised.png",
                        "image/png",
                    )

            else:
                # Show original + noisy before processing
                ic1, ic2 = st.columns(2)
                with ic1:
                    st.markdown(
                        '<div class="img-panel-label">ORIGINAL</div>',
                        unsafe_allow_html=True,
                    )
                    st.image(orig_np, use_container_width=True, clamp=True)
                with ic2:
                    st.markdown(
                        '<div class="img-panel-label">NOISY (PREVIEW)</div>',
                        unsafe_allow_html=True,
                    )
                    st.image(noisy_np, use_container_width=True, clamp=True)
                st.markdown(
                    '<div class="status-bar info">👆 Click PROCESS IMAGE to denoise</div>',
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────
# PAGE: BATCH PROCESSING
# ─────────────────────────────────────────────
elif page == "📦  Batch Processing":
    st.markdown(
        """
    <div class="hero-banner" style="padding:1.2rem 2rem">
        <div class="hero-title" style="font-size:1.6rem">BATCH PROCESSOR</div>
        <div class="hero-subtitle" style="font-size:0.95rem">PROCESS MULTIPLE MRI IMAGES AT ONCE</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    bc1, bc2 = st.columns([1, 3])

    with bc1:
        st.markdown(
            '<div class="info-card"><h3>⚙️ BATCH SETTINGS</h3>', unsafe_allow_html=True
        )
        batch_noise = st.selectbox(
            "Noise Type", ["Gaussian", "Salt & Pepper", "Speckle"]
        )
        batch_noise_lvl = st.slider("Noise Level", 0.01, 0.4, 0.15, 0.01, key="b_noise")
        batch_method = st.selectbox(
            "Denoising Method",
            [
                "Bilateral Filter (Edge-Preserving)",
                "Gaussian Filter",
                "Median Filter",
                "Non-Local Means (NLM)",
            ],
        )
        show_results = st.checkbox("Show Results Grid", True)
        st.markdown("</div>", unsafe_allow_html=True)

    with bc2:
        batch_files = st.file_uploader(
            "Upload Multiple MRI Images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

        if batch_files:
            st.markdown(
                f'<div class="status-bar info">📁 {len(batch_files)} IMAGE(S) LOADED</div>',
                unsafe_allow_html=True,
            )

            if st.button("⚡ PROCESS ALL"):
                results = []
                psnr_list, ssim_list = [], []

                prog = st.progress(0)
                status = st.empty()

                for i, f in enumerate(batch_files):
                    status.markdown(
                        f'<div class="status-bar info">⟳ Processing {f.name}...</div>',
                        unsafe_allow_html=True,
                    )
                    pil = Image.open(f).convert("L")
                    orig = pil_to_np(pil)

                    if batch_noise == "Gaussian":
                        noisy = add_gaussian_noise(orig, std=batch_noise_lvl)
                    elif batch_noise == "Salt & Pepper":
                        noisy = add_salt_pepper_noise(orig, amount=batch_noise_lvl)
                    else:
                        noisy = add_speckle_noise(orig, std=batch_noise_lvl)

                    if batch_method == "Gaussian Filter":
                        denoised = denoise_gaussian_filter(noisy)
                    elif batch_method == "Bilateral Filter (Edge-Preserving)":
                        denoised = denoise_bilateral(noisy)
                    elif batch_method == "Median Filter":
                        denoised = denoise_median(noisy)
                    else:
                        denoised = denoise_nlm(noisy)

                    denoised_sq = denoised.squeeze() if denoised.ndim == 3 else denoised
                    m = compute_metrics(orig, denoised_sq)
                    psnr_list.append(m["PSNR"])
                    ssim_list.append(m["SSIM"])
                    results.append((f.name, orig, noisy, denoised_sq, m))
                    prog.progress((i + 1) / len(batch_files))

                status.markdown(
                    '<div class="status-bar success">✓ ALL IMAGES PROCESSED</div>',
                    unsafe_allow_html=True,
                )

                # Summary metrics
                st.markdown(
                    """
                <div class="section-header">
                    <span class="section-title">BATCH SUMMARY</span>
                    <div class="section-line"></div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                sm1, sm2, sm3, sm4 = st.columns(4)
                with sm1:
                    st.markdown(
                        f"""<div class="metric-card"><div class="metric-label">IMAGES</div>
                    <div class="metric-value">{len(results)}</div></div>""",
                        unsafe_allow_html=True,
                    )
                with sm2:
                    st.markdown(
                        f"""<div class="metric-card"><div class="metric-label">AVG PSNR</div>
                    <div class="metric-value">{np.mean(psnr_list):.2f}<span class="metric-unit">dB</span></div></div>""",
                        unsafe_allow_html=True,
                    )
                with sm3:
                    st.markdown(
                        f"""<div class="metric-card violet"><div class="metric-label">AVG SSIM</div>
                    <div class="metric-value">{np.mean(ssim_list):.4f}</div></div>""",
                        unsafe_allow_html=True,
                    )
                with sm4:
                    st.markdown(
                        f"""<div class="metric-card teal"><div class="metric-label">MIN PSNR</div>
                    <div class="metric-value">{np.min(psnr_list):.2f}<span class="metric-unit">dB</span></div></div>""",
                        unsafe_allow_html=True,
                    )

                # PSNR chart
                with st.expander("📊 PSNR per Image"):
                    fig2, ax2 = plt.subplots(figsize=(max(6, len(results) * 0.6), 3.5))
                    fig2.patch.set_facecolor("#040f2a")
                    ax2.set_facecolor("#020818")
                    names = [r[0][:15] for r in results]
                    bars = ax2.bar(
                        names, psnr_list, color="#00e5ff", alpha=0.7, width=0.6
                    )
                    ax2.axhline(
                        np.mean(psnr_list),
                        color="#00ffd0",
                        linestyle="--",
                        linewidth=1.5,
                        label=f"Mean: {np.mean(psnr_list):.2f}dB",
                    )
                    ax2.set_ylabel("PSNR (dB)", color="#a8c8e8", fontsize=9)
                    ax2.tick_params(colors="#5a7a9a", labelsize=7, rotation=30)
                    for spine in ax2.spines.values():
                        spine.set_color("#0d2040")
                    ax2.legend(
                        facecolor="#040f2a",
                        edgecolor="#1a3a5a",
                        labelcolor="#a8c8e8",
                        fontsize=8,
                    )
                    ax2.grid(True, alpha=0.1, color="#1a3a5a", axis="y")
                    plt.tight_layout()
                    st.pyplot(fig2, use_container_width=True)
                    plt.close(fig2)

                # Results grid
                if show_results:
                    st.markdown(
                        """
                    <div class="section-header">
                        <span class="section-title">RESULTS GRID</span>
                        <div class="section-line"></div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                    for name, orig, noisy, denoised, metrics in results:
                        with st.expander(
                            f"🖼️ {name}  |  PSNR: {metrics['PSNR']:.2f} dB  |  SSIM: {metrics['SSIM']:.4f}"
                        ):
                            gc1, gc2, gc3 = st.columns(3)
                            with gc1:
                                st.markdown(
                                    '<div class="img-panel-label">ORIGINAL</div>',
                                    unsafe_allow_html=True,
                                )
                                st.image(orig, use_container_width=True, clamp=True)
                            with gc2:
                                st.markdown(
                                    '<div class="img-panel-label">NOISY</div>',
                                    unsafe_allow_html=True,
                                )
                                st.image(noisy, use_container_width=True, clamp=True)
                            with gc3:
                                st.markdown(
                                    '<div class="img-panel-label">DENOISED</div>',
                                    unsafe_allow_html=True,
                                )
                                st.image(denoised, use_container_width=True, clamp=True)

                # ZIP download
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for name, orig, noisy, denoised, _ in results:
                        stem = os.path.splitext(name)[0]
                        zf.writestr(f"original/{stem}_original.png", img_to_bytes(orig))
                        zf.writestr(f"noisy/{stem}_noisy.png", img_to_bytes(noisy))
                        zf.writestr(
                            f"denoised/{stem}_denoised.png", img_to_bytes(denoised)
                        )
                zip_buf.seek(0)
                st.download_button(
                    "⬇ DOWNLOAD ALL RESULTS (ZIP)",
                    zip_buf.getvalue(),
                    "neuroclean_batch_results.zip",
                    "application/zip",
                )


# ─────────────────────────────────────────────
# PAGE: ARCHITECTURE
# ─────────────────────────────────────────────
elif page == "🏗️  Architecture":
    st.markdown(
        """
    <div class="hero-banner" style="padding:1.2rem 2rem">
        <div class="hero-title" style="font-size:1.6rem">MODEL ARCHITECTURE</div>
        <div class="hero-subtitle" style="font-size:0.95rem">CONVOLUTIONAL AUTOENCODER · ENCODER–BOTTLENECK–DECODER</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    ac1, ac2 = st.columns([1, 1])

    with ac1:
        st.markdown(
            """
        <div class="section-header">
            <span class="section-title">LAYER BREAKDOWN</span>
            <div class="section-line"></div>
        </div>
        <div class="info-card">

        <div style="margin-bottom:0.5rem">
            <span class="badge badge-cyan">ENCODER</span>
        </div>
        <div class="arch-layer encoder">
            <span class="arch-name">Input</span>
            <span class="arch-shape">(128, 128, 1)</span>
            <span class="arch-params">—</span>
        </div>
        <div class="arch-layer encoder">
            <span class="arch-name">Conv2D (32, 3×3, ReLU)</span>
            <span class="arch-shape">(128, 128, 32)</span>
            <span class="arch-params">320 params</span>
        </div>
        <div class="arch-layer encoder">
            <span class="arch-name">MaxPooling2D (2×2)</span>
            <span class="arch-shape">(64, 64, 32)</span>
            <span class="arch-params">0 params</span>
        </div>
        <div class="arch-layer encoder">
            <span class="arch-name">Conv2D (32, 3×3, ReLU)</span>
            <span class="arch-shape">(64, 64, 32)</span>
            <span class="arch-params">9,248 params</span>
        </div>

        <div style="margin:0.8rem 0 0.5rem 0">
            <span class="badge badge-violet">BOTTLENECK</span>
        </div>
        <div class="arch-layer bottleneck">
            <span class="arch-name">MaxPooling2D (2×2)</span>
            <span class="arch-shape">(32, 32, 32)</span>
            <span class="arch-params">0 params</span>
        </div>

        <div style="margin:0.8rem 0 0.5rem 0">
            <span class="badge badge-teal">DECODER</span>
        </div>
        <div class="arch-layer decoder">
            <span class="arch-name">Conv2D (32, 3×3, ReLU)</span>
            <span class="arch-shape">(32, 32, 32)</span>
            <span class="arch-params">9,248 params</span>
        </div>
        <div class="arch-layer decoder">
            <span class="arch-name">UpSampling2D (2×2)</span>
            <span class="arch-shape">(64, 64, 32)</span>
            <span class="arch-params">0 params</span>
        </div>
        <div class="arch-layer decoder">
            <span class="arch-name">Conv2D (32, 3×3, ReLU)</span>
            <span class="arch-shape">(64, 64, 32)</span>
            <span class="arch-params">9,248 params</span>
        </div>
        <div class="arch-layer decoder">
            <span class="arch-name">UpSampling2D (2×2)</span>
            <span class="arch-shape">(128, 128, 32)</span>
            <span class="arch-params">0 params</span>
        </div>
        <div class="arch-layer decoder">
            <span class="arch-name">Conv2D (1, 3×3, Sigmoid)</span>
            <span class="arch-shape">(128, 128, 1)</span>
            <span class="arch-params">289 params</span>
        </div>

        <div style="margin-top:1rem; padding-top:0.8rem; border-top:1px solid var(--border)">
            <span class="arch-name" style="color:var(--cyan); font-family:'Orbitron',monospace; font-size:0.85rem;">
                Total Trainable Parameters: ~28,353
            </span>
        </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with ac2:
        st.markdown(
            """
        <div class="section-header">
            <span class="section-title">ARCHITECTURE DIAGRAM</span>
            <div class="section-line"></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Build architecture visualization with matplotlib
        fig3, ax3 = plt.subplots(figsize=(5, 9))
        fig3.patch.set_facecolor("#040f2a")
        ax3.set_facecolor("#040f2a")
        ax3.set_xlim(0, 10)
        ax3.set_ylim(0, 20)
        ax3.axis("off")

        layers = [
            ("INPUT", "(128×128×1)", "#870058", 18.5),
            ("Conv2D 32", "(128×128×32)", "#870058", 16.5),
            ("MaxPool", "(64×64×32)", "#870058", 14.5),
            ("Conv2D 32", "(64×64×32)", "#870058", 12.5),
            ("MaxPool\n[BOTTLENECK]", "(32×32×32)", "#4B1745", 10.5),
            ("Conv2D 32", "(32×32×32)", "#870058", 8.5),
            ("UpSample", "(64×64×32)", "#870058", 6.5),
            ("Conv2D 32", "(64×64×32)", "#870058", 4.5),
            ("UpSample", "(128×128×32)", "#870058", 2.5),
            ("Conv2D 1\n[OUTPUT]", "(128×128×1)", "#4B1745", 0.5),
        ]

        for i, (name, shape, color, y) in enumerate(layers):
            # Box
            rect = plt.Rectangle(
                (1.5, y - 0.7),
                7,
                1.4,
                facecolor=color + "18",
                edgecolor=color,
                linewidth=1.5,
                alpha=0.9,
                zorder=2,
            )
            ax3.add_patch(rect)
            ax3.text(
                5,
                y + 0.05,
                name,
                color=color,
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                fontfamily="monospace",
                zorder=3,
            )
            ax3.text(
                5,
                y - 0.35,
                shape,
                color="#5a7a9a",
                fontsize=6.5,
                ha="center",
                va="center",
                fontfamily="monospace",
                zorder=3,
            )
            # Arrow
            if i < len(layers) - 1:
                ax3.annotate(
                    "",
                    xy=(5, layers[i + 1][3] + 0.7),
                    xytext=(5, y - 0.7),
                    arrowprops=dict(
                        arrowstyle="->", color="#1a3a5a", lw=1.5, mutation_scale=12
                    ),
                    zorder=1,
                )

        plt.tight_layout(pad=0.5)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

        st.markdown(
            """
        <div class="info-card" style="margin-top:1rem">
            <h3>🔑 KEY DESIGN CHOICES</h3>
            <p>
            <span class="badge badge-cyan">Padding='same'</span> preserves spatial resolution at each conv layer.<br><br>
            <span class="badge badge-violet">Bottleneck (32×32×32)</span> — 4× spatial compression forces the network to learn noise-invariant features.<br><br>
            <span class="badge badge-teal">Sigmoid output</span> maps reconstructed values to [0, 1], matching the normalized input.<br><br>
            <span class="badge badge-cyan">Adam optimizer</span> + MSE loss penalizes pixel-wise reconstruction error.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Training code
    with st.expander("📋 View Training Code"):
        st.code(
            """
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.optimizers import Adam
import numpy as np

# ─── BUILD MODEL ─────────────────────────────
inp = Input(shape=(128, 128, 1))
# Encoder
x = Conv2D(32, (3,3), activation='relu', padding='same')(inp)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same')(x)
# Decoder
x = Conv2D(32, (3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

autoencoder = Model(inp, decoded)
autoencoder.compile(optimizer=Adam(), loss='mse')

# ─── PREPARE DATA ────────────────────────────
# images: shape (N, 128, 128, 1), normalized [0,1]
noise_factor = 0.1
noisy = images + noise_factor * np.random.normal(size=images.shape)
noisy = np.clip(noisy, 0., 1.)

# ─── TRAIN ───────────────────────────────────
autoencoder.fit(noisy, images,
                epochs=50,
                batch_size=16,
                shuffle=True,
                validation_split=0.2)

# ─── SAVE ────────────────────────────────────
autoencoder.save('neuroclean_autoencoder.h5')
        """,
            language="python",
        )


# ─────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────
elif page == "📊  About":
    st.markdown(
        """
    <div class="hero-banner" style="padding:1.2rem 2rem">
        <div class="hero-title" style="font-size:1.6rem">ABOUT THIS PROJECT</div>
        <div class="hero-subtitle" style="font-size:0.95rem">TECHNICAL DETAILS · AUTHOR · REFERENCES</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    t1, t2 = st.tabs(["📖 Project Details", "👩‍💻 Author & Links"])

    with t1:
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown(
                """
            <div class="info-card">
                <h3>📌 PROJECT SUMMARY</h3>
                <p>This project implements a <strong>Denoising Convolutional Autoencoder</strong> for brain tumor MRI images.
                The model learns to reconstruct clean MRI scans from artificially corrupted inputs, simulating real-world scanner noise.
                It was trained on 100 grayscale brain tumor MRI images (128×128 px) achieving an average PSNR of 20.71 dB.</p>
            </div>
            <div class="info-card">
                <h3>🗂️ DATASET</h3>
                <p>100 Brain Tumor MRI grayscale images loaded from a local dataset directory.
                Each image is resized to 128×128 pixels and normalized to the [0, 1] range.
                Noisy versions are created by adding Gaussian noise (σ=0.1) for supervised training.
                The model trains with (noisy → clean) paired examples.</p>
            </div>
            <div class="info-card">
                <h3>📐 EVALUATION METRICS</h3>
                <p>
                <strong>PSNR</strong> (Peak Signal-to-Noise Ratio): measures logarithmic ratio of max signal to noise power. Higher = better. Achieved: <strong>~20.71 dB</strong>.<br><br>
                <strong>SSIM</strong> (Structural Similarity Index): perceptual metric assessing luminance, contrast, and structure. Range [0,1]. Higher = better.<br><br>
                <strong>MSE</strong>: Mean Squared Error — the direct training objective.<br><br>
                <strong>MAE</strong>: Mean Absolute Error — more robust to outliers than MSE.
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with cc2:
            st.markdown(
                """
            <div class="info-card">
                <h3>🛠️ TECH STACK</h3>
                <p>
                <span class="badge badge-cyan">Python 3.8+</span>
                <span class="badge badge-cyan">TensorFlow 2.x</span>
                <span class="badge badge-violet">Keras</span>
                <span class="badge badge-teal">NumPy</span>
                <span class="badge badge-teal">OpenCV</span>
                <span class="badge badge-cyan">scikit-image</span>
                <span class="badge badge-violet">Matplotlib</span>
                <span class="badge badge-cyan">Streamlit</span>
                <span class="badge badge-teal">PIL/Pillow</span>
                </p>
            </div>
            <div class="info-card">
                <h3>🔬 CLASSICAL METHODS INCLUDED</h3>
                <p>
                <strong>Gaussian Filter</strong>: Smooth by convolving with Gaussian kernel. Fast but blurs edges.<br><br>
                <strong>Bilateral Filter</strong>: Edge-preserving smoothing, considers both spatial distance and intensity difference.<br><br>
                <strong>Median Filter</strong>: Replace each pixel with neighborhood median. Excellent for salt & pepper noise.<br><br>
                <strong>Non-Local Means (NLM)</strong>: Averages similar patches across the whole image. Preserves fine texture.<br><br>
                <strong>Frequency Domain (FFT)</strong>: Suppress low-energy frequencies (noise) in Fourier space.
                </p>
            </div>
            <div class="info-card">
                <h3>📁 REPOSITORY STRUCTURE</h3>
                <p>
                <code>app.py</code> — Main Streamlit application<br>
                <code>requirements.txt</code> — Python dependencies<br>
                <code>image-denoising.ipynb</code> — Training notebook<br>
                <code>dataset/</code> — Brain tumor MRI images<br>
                <code>neuroclean_autoencoder.h5</code> — Saved model weights
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with t2:
        ac1, ac2 = st.columns([1, 1])
        with ac1:
            st.markdown(
                """
            <div class="info-card" style="text-align:center; padding:2.5rem">
                <div style="font-size:4rem; margin-bottom:1rem">👩‍🔬</div>
                <div style="font-family:'Orbitron',monospace; font-size:1.2rem; color:var(--cyan); margin-bottom:0.3rem; font-weight:900;">
                    Hafsa Ibrahim
                </div>
                <div style="color:var(--text-mid); font-size:0.95rem; margin-bottom:1.5rem;">
                    AI / Machine Learning Engineer<br>
                    <span style="color:var(--text-dim); font-size:0.85rem;">Computer Vision · Deep Learning · Medical Imaging</span>
                </div>
                <div style="margin-top:1rem;">
                    <a class="social-link" href="https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/" target="_blank" style="font-size:0.9rem; padding:0.5rem 1.2rem; margin:0.3rem;">
                        💼 LinkedIn Profile
                    </a><br><br>
                    <a class="social-link" href="https://github.com/HafsaIbrahim5" target="_blank" style="font-size:0.9rem; padding:0.5rem 1.2rem; background:rgba(123,47,255,0.1); border-color:rgba(123,47,255,0.3); color:#b07fff;">
                        🐙 GitHub Profile
                    </a>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with ac2:
            st.markdown(
                """
            <div class="info-card">
                <h3>📚 REFERENCES</h3>
                <p>
                <strong>1.</strong> Vincent, P. et al. (2010). Stacked Denoising Autoencoders. <em>JMLR</em>.<br><br>
                <strong>2.</strong> Gondara, L. (2016). Medical Image Denoising Using Convolutional Denoising Autoencoders. <em>IEEE ICDMW</em>.<br><br>
                <strong>3.</strong> Wang, Z. et al. (2004). Image Quality Assessment: From Error Visibility to Structural Similarity. <em>IEEE TIP</em>.<br><br>
                <strong>4.</strong> Buades, A. et al. (2005). A Non-Local Algorithm for Image Denoising. <em>CVPR</em>.<br><br>
                <strong>5.</strong> Tomasi, C. & Manduchi, R. (1998). Bilateral Filtering for Gray and Color Images. <em>ICCV</em>.
                </p>
            </div>
            <div class="info-card">
                <h3>⚖️ LICENSE & USAGE</h3>
                <p>This project is open-source and intended for educational and research purposes.
                The model was trained on publicly available brain tumor MRI data.
                Not intended for clinical diagnosis without proper validation.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
    <div style="text-align:center; padding:2rem 0 1rem 0;">
        <div style="font-family:'Orbitron',monospace; font-size:0.75rem; color:var(--text-dim); letter-spacing:3px;">
            NEUROCLEAN · BRAIN MRI DENOISER · BUILT WITH STREAMLIT
        </div>
        <div style="font-size:0.8rem; color:var(--text-dim); margin-top:0.3rem;">
            <a href="https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/" target="_blank" style="color:var(--cyan); text-decoration:none;">LinkedIn</a>
            &nbsp;·&nbsp;
            <a href="https://github.com/HafsaIbrahim5" target="_blank" style="color:#b07fff; text-decoration:none;">GitHub</a>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

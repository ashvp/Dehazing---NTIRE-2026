# 🌙 Nighttime Image Dehazing — NTIRE 2026

A deep-learning based approach for **nighttime image dehazing**, developed for the **NTIRE 2026 Night Time Image Dehazing Challenge**.

The project focuses on restoring clear nighttime images affected by haze, glow, non-uniform illumination, color distortion, and sensor noise. The primary restoration model implemented in this repository is **FFA-Net (Feature Fusion Attention Network)**, fine-tuned from pretrained indoor dehazing weights.

---

## 📌 Problem

Nighttime image dehazing is considerably more challenging than conventional daytime dehazing because nighttime scenes contain:

* 🌫️ Non-uniform haze and atmospheric scattering
* 💡 Strong glow and halo around light sources
* 🌑 Extremely dark regions and low signal-to-noise ratios
* 🎨 Color casts caused by low-light imaging
* 🔆 Coexisting overexposed and underexposed regions
* 📷 Significant sensor noise

These challenges are important for applications such as autonomous driving, surveillance, intelligent transportation systems, and downstream computer-vision tasks.

---

## 🏆 NTIRE 2026 Challenge

The challenge uses a small real-world paired nighttime dataset designed to test generalization.

According to the project presentation:

| Split      | Description                                |
| ---------- | ------------------------------------------ |
| Training   | 25 paired high-resolution nighttime images |
| Validation | 5 paired images                            |
| Test       | ~50+ hidden images                         |
| Evaluation | PSNR, SSIM and LPIPS                       |

The dataset contains real nighttime haze, glow, noise, color casts, and pixel-level clear ground truth.

Because the dataset is small, the project uses patch-based training and augmentation to increase the effective amount of training data.

---

# 🧠 Method

## FFA-Net

The main restoration network is **FFA-Net — Feature Fusion Attention Network**.

The implementation uses:

* **3 Groups**
* **19 Blocks per Group**
* **64 feature channels**
* **Channel Attention (CA)**
* **Pixel Attention (PA)**
* Local residual connections
* Group-level residual connections
* Global residual learning
* Attention-based feature fusion

The architecture follows the FFA-Net design implemented in `train.py`.

### Architecture

```text
                     Hazy Image
                          │
                          ▼
                  ┌──────────────┐
                  │ Pre-Process  │
                  │    3 → 64    │
                  └──────┬───────┘
                         │
              ┌──────────▼──────────┐
              │      Group 1        │
              │   19 FFA Blocks     │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │      Group 2        │
              │   19 FFA Blocks     │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │      Group 3        │
              │   19 FFA Blocks     │
              └──────────┬──────────┘
                         │
                  ┌──────▼──────┐
                  │  Feature    │
                  │   Fusion    │
                  │     CA      │
                  └──────┬──────┘
                         │
                  Pixel Attention
                         │
                         ▼
                  ┌──────────────┐
                  │ Post-Process │
                  │   64 → 3     │
                  └──────┬───────┘
                         │
                         ▼
                   Clear Image
```

Each FFA block combines convolutional feature extraction with **channel attention and pixel attention**, allowing the network to focus on important features in challenging nighttime scenes.

---

# 🔄 Training Pipeline

```text
Hazy / Ground Truth Image Pairs
              │
              ▼
        Load into RAM
              │
              ▼
        Random 256×256 Crop
              │
              ▼
      Data Augmentation
       ├── Horizontal Flip
       ├── Vertical Flip
       └── Random Rotation
              │
              ▼
        FFA-Net
              │
              ▼
       Predicted Image
              │
              ▼
     Charbonnier Loss
              │
              ▼
        Backpropagation
              │
              ▼
        Adam Optimizer
              │
              ▼
      Cosine LR Schedule
              │
              ▼
     Full Image Evaluation
```

The current training configuration uses:

| Parameter             |        Value |
| --------------------- | -----------: |
| Patch size            |    256 × 256 |
| Repeats               |           50 |
| Batch size            |            4 |
| Epochs                |           60 |
| Initial learning rate |     5 × 10⁻⁵ |
| Warm-up               |     3 epochs |
| Optimizer             |         Adam |
| Loss                  |  Charbonnier |
| Gradient clipping     |          1.0 |
| Mixed precision       |     CUDA AMP |
| Multi-GPU             | DataParallel |

These settings are defined directly in `train.py`.

---

# 🎯 Loss Function

The current implementation uses **Charbonnier Loss**:

```text
L(x, y) = mean( √((x - y)² + ε) )
```

with:

```text
ε = 1e-6
```

Charbonnier loss is used as a robust pixel-level reconstruction objective.

The challenge presentation also recommends strong pixel losses such as L1, together with structural, perceptual, color, and nighttime-specific losses for balancing PSNR, SSIM, and LPIPS.

---

# 📊 Evaluation Metrics

The challenge evaluates restoration quality using three complementary metrics.

### PSNR ↑

Measures pixel-level fidelity between the restored image and ground truth.

**Higher is better.**

Typical interpretation from the project specification:

```text
< 15 dB       Very Poor
15–20 dB      Poor
20–25 dB      Acceptable
25–30 dB      Good
> 30 dB       Excellent
```

### SSIM ↑

Measures structural similarity based on:

* Luminance
* Contrast
* Local structure

**Higher is better.**

The project specification considers values above 0.85 excellent.

### LPIPS ↓

Measures perceptual distance using learned deep features.

**Lower is better.**

LPIPS is particularly useful for detecting perceptual artifacts, unnatural textures, color distortions, and overly aggressive dehazing that pixel metrics may not capture.

---

# 🖼️ Qualitative Results

Example comparison on:

**`10_NTHazy.png`**

| Method                 |         PSNR |       SSIM |
| ---------------------- | -----------: | ---------: |
| FFA-Net                | **23.54 dB** | **0.8287** |
| Restormer              |     21.50 dB |     0.7247 |
| Ensemble (0.65 / 0.35) | **23.70 dB** |     0.8190 |
| Ground Truth           |            — |          — |

From the shown example, FFA-Net produces a substantially clearer image than the hazy input while retaining nighttime illumination and scene structure.

The ensemble configuration achieves the highest PSNR in this displayed comparison, while FFA-Net achieves the highest SSIM among the listed learned methods.

> **Note:** These values represent the displayed example result and should not be interpreted as the overall challenge/test-set score.

---

# ⚖️ Ensemble

A simple ensemble is also evaluated by combining the outputs of FFA-Net and another restoration model:

```text
Ensemble = 0.65 × FFA-Net + 0.35 × Restormer
```

For the displayed example:

```text
PSNR : 23.70 dB
SSIM : 0.8190
```

The ensemble provides another way to balance the complementary restoration behavior of different architectures.

---

# 🚀 Pretrained Weights

The FFA-Net implementation supports initialization from pretrained **ITS indoor dehazing weights**:

```text
its_train_ffa_3_19.pk
```

The training script expects:

```python
PRETRAINED = "/kaggle/input/ffa-pretrained/its_train_ffa_3_19.pk"
```

If the weights are unavailable, the script falls back to training from scratch.

---

# 💻 Installation

```bash
git clone <YOUR_REPOSITORY_URL>
cd nighttime-image-dehazing
```

Install the required dependencies:

```bash
pip install torch torchvision
pip install numpy pillow matplotlib tqdm scikit-image
```

For Kaggle, the pretrained checkpoint can be downloaded before running training.

---

# 📁 Dataset Structure

The training script expects paired hazy and ground-truth images:

```text
dataset/
├── train/
│   ├── 01_NTHazy.png
│   ├── 02_NTHazy.png
│   └── ...
│
└── gt/
    ├── 01_GT.png
    ├── 02_GT.png
    └── ...
```

The pairing convention used by the implementation is:

```text
_NTHazy → _GT
```

For example:

```text
10_NTHazy.png
        ↓
10_GT.png
```

---

# 🏋️ Training

Update the dataset paths in `train.py`:

```python
TRAIN_DIR = "/path/to/train"
GT_DIR    = "/path/to/gt"
SAVE_DIR  = "/path/to/checkpoints"
```

Then run:

```bash
python train.py
```

The training script:

1. Loads paired images.
2. Creates random 256×256 patches.
3. Applies random flips and rotations.
4. Fine-tunes FFA-Net.
5. Uses mixed-precision CUDA training.
6. Calculates training PSNR.
7. Performs full-image validation every 5 epochs.
8. Saves the best-performing model.
9. Saves epoch checkpoints as `.pth` and `.zip`.
10. Generates qualitative visualizations every 5 epochs.

---

# 💾 Checkpoints

Checkpoints are stored in:

```text
checkpoints_ffa/
```

The best model is:

```text
ffa_best.pth
```

Epoch checkpoints follow:

```text
ffa_epoch_1.pth
ffa_epoch_1.zip

ffa_epoch_2.pth
ffa_epoch_2.zip

...
```

The best model is selected using validation PSNR.

---

# 🔬 Current Limitations

The current implementation is a strong FFA-Net baseline, but several improvements are possible.

### 1. Small dataset

The challenge dataset contains very few real paired images, making overfitting a major concern.

### 2. Current loss is primarily pixel-based

The current code uses Charbonnier loss. It does not currently include explicit:

* SSIM loss
* LPIPS/perceptual loss
* Color consistency loss
* Brightness realism loss
* Frequency/wavelet loss

These are potential directions suggested by the project design.

### 3. Validation implementation

The current `train.py` constructs `val_files` from `TRAIN_DIR`:

```python
val_files = sorted(os.listdir(TRAIN_DIR))
```

Therefore, unless the dataset directory itself has been arranged differently, the current script evaluates the full images from the training directory rather than a separate held-out validation directory.

For a reliable experiment, a separate validation directory should be used.

---

# 🔮 Future Improvements

Potential extensions include:

* [ ] Add SSIM loss
* [ ] Add perceptual/VGG loss
* [ ] Add LPIPS-based evaluation
* [ ] Add brightness realism loss
* [ ] Add color consistency loss
* [ ] Add frequency-domain/wavelet loss
* [ ] Add nighttime-specific glow suppression
* [ ] Improve noise removal
* [ ] Use a dedicated validation split
* [ ] Experiment with Restormer
* [ ] Improve FFA-Net + Restormer ensemble
* [ ] Test external/synthetic pretraining
* [ ] Perform test-time augmentation
* [ ] Add EMA model weights
* [ ] Compare multiple loss combinations

The challenge presentation specifically highlights color-aware losses, denoising, frequency-domain approaches, brightness realism, and perceptual losses as promising directions.

---

# 📚 Project Structure

```text
nighttime-image-dehazing/
│
├── train.py
├── README.md
│
├── dataset/
│   ├── train/
│   └── gt/
│
├── checkpoints/
│   ├── ffa_best.pth
│   └── ...
│
└── results/
    ├── qualitative/
    └── metrics/
```

---

# 📈 Experimental Summary

### FFA-Net

```text
Architecture:
    3 Groups × 19 Blocks
    64 feature channels
    Channel Attention
    Pixel Attention

Training:
    Patch Size      : 256 × 256
    Batch Size      : 4
    Epochs          : 60
    Learning Rate   : 5e-5
    Optimizer       : Adam
    Loss            : Charbonnier
    Scheduler       : Warmup + Cosine Decay
```

### Example Result

```text
FFA-Net
PSNR : 23.54 dB
SSIM : 0.8287

Restormer
PSNR : 21.50 dB
SSIM : 0.7247

Ensemble (0.65 FFA + 0.35 Restormer)
PSNR : 23.70 dB
SSIM : 0.8190
```

---

# 👥 Team

**NTIRE 2026 Nighttime Image Dehazing Challenge**

This project is developed as part of an image and video processing research project focused on robust nighttime image restoration.

---

# 📖 References

* NTIRE 2026 Night Time Image Dehazing Challenge — project specification and evaluation framework.
* FFA-Net: Feature Fusion Attention Network for Single Image Dehazing.
* FFA-Net implementation used in this project.

---

## ⭐ Key Takeaway

The project explores **attention-based nighttime image dehazing using FFA-Net**, with patch-based training, augmentation, pretrained initialization, mixed-precision optimization, and PSNR/SSIM-based evaluation.

The initial qualitative experiment demonstrates that FFA-Net can significantly improve visibility in nighttime hazy scenes while preserving scene structure and realistic illumination.

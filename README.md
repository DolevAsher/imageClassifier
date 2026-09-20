# 🖼️ ImageClassifier — IML Hackathon

A 20-class image classifier built around a custom residual CNN, fine-tuned
for robustness against real-world image distortions (flips, rotations,
blur, color jitter, and more). Includes an interactive **Streamlit** app
for exploring the architecture and testing the model live.

---

## 🧠 Architecture

The network is built from **5 sequential stages**, each following the
same pattern:

```
Conv2d → BatchNorm → ReLU → ResBlock → MaxPool
```

A `ResBlock` is a standard two-layer residual block with a skip
connection, which helps gradients flow through the deeper network.

| Stage | Channels | Output size |
|:---:|:---:|:---:|
| 1 | 32  | 112 × 112 |
| 2 | 64  | 56 × 56 |
| 3 | 128 | 28 × 28 |
| 4 | 256 | 14 × 14 |
| 5 | 512 | 7 × 7 |

After the 5 stages, an **adaptive average pool** collapses the 7×7
feature map to 1×1, and a dense classifier head
(`Linear → BatchNorm → ReLU → Dropout → Linear`) maps the resulting
512-dim feature vector to the **20 output classes**.

### The 20 Classes

| | | | |
|---|---|---|---|
| Goldfish | Bald Eagle | Toucan | Jellyfish |
| Tiger | African Elephant | Acoustic Guitar | Airliner |
| Balloon | Lighthouse | Castle | Mobile Phone |
| Container Ship | French Horn | Laptop | Sports Car |
| Mushroom | Lemon | Pizza | Daisy |

---

## 🏋️ Training Strategy

Training happened in two phases:

1. **Base training** — trained from scratch with light augmentations.
2. **Fine-tuning** — continued training with heavier augmentations,
   while validating that the model still recognized unmodified images.

**Augmentations used:** horizontal flip, vertical flip, rotation (up to
15°), affine projections, grayscale, color jitter, color inversion,
Gaussian blur.

### Results

| Stage | Clean test accuracy | Distorted test accuracy |
|---|:---:|:---:|
| Pre fine-tuning  | 88.4% | 63.5% |
| Post fine-tuning | **89.2%** | **81.7%** |

Fine-tuning traded a small amount of clean accuracy for a large gain in
robustness to distortions — an **18-point jump** on modified images for
less than 1 point of clean-accuracy cost.

---

## 📁 Repo Structure

```
imageClassifier/
├── model.py           # ModelArchitecture (the CNN) + ResBlock
├── train.py            # Full training pipeline
├── predict.py          # Grader-facing prediction wrapper
├── weights.joblib      # Trained model weights (best checkpoint by F1)
├── streamlit_app.py    # Interactive demo app
├── run_app.py          # One-click launcher (no CLI needed)
├── requirements.txt
└── README.md
```

---

## 🚀 Try It Yourself

First, install dependencies:

```bash
pip install -r requirements.txt
```

Then launch the app one of two ways:

**Option A — one click, no command line:**
Run `run_app.py` (double-click it, or `python run_app.py`). It starts the
Streamlit server for you and skips the first-run email/telemetry prompt.

**Option B — standard Streamlit CLI:**
```bash
streamlit run streamlit_app.py
```

Either way, your browser opens to an app with two tabs:

- **📐 Architecture** — a visual walkthrough of the network design and
  training results above.
- **🎛️ Try It** — upload an image, tweak flip/rotation/blur/color
  controls in the sidebar, and watch the model's top-5 predictions
  update live.

---

## 🔧 Using the Model Programmatically

```python
import joblib
import torch
from model import ModelArchitecture

model = ModelArchitecture(num_classes=20)
model.load_state_dict(joblib.load("weights.joblib"))
model.eval()

# x: tensor of shape [batch_size, 3, 224, 224], ImageNet-normalized
with torch.no_grad():
    preds = model(x).argmax(dim=1)
```
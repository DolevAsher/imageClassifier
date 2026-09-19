"""
Streamlit playground for the IML Hackathon image classifier.

Two tabs:
  1. Architecture — explains the network design (from README.txt)
  2. Try It       — upload an image, apply the same kinds of distortions
                     used during fine-tuning, and see whether the model
                     still recognizes it.

Run with:
    streamlit run streamlit_app.py

Expects these files in the same folder:
    model.py        (defines ModelArchitecture)
    weights.joblib   (trained state_dict)
"""

import io
import json
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

from model import ModelArchitecture

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
NUM_CLASSES = 20
WEIGHTS_PATH = "weights.joblib"
LABELS_PATH = Path(__file__).resolve().parent / "labels.json"
IMAGE_SIZE = 224
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
EXAMPLE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
SLIDER_COMPONENT_DIR = Path(__file__).resolve().parent / "slider_component"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

st.set_page_config(page_title="IML Hackathon Classifier", layout="wide")

_range_slider = components.declare_component(
    "range_slider", path=str(SLIDER_COMPONENT_DIR)
)


def range_slider(label, min_value, max_value, value=0, step=1, key=None):
    """RTL-locale-safe replacement for st.slider (native <input type=range>)."""
    return _range_slider(
        label=label,
        min_value=min_value,
        max_value=max_value,
        value=value,
        step=step,
        key=key,
        default=value,
    )


def list_example_images():
    """Any image files dropped into the examples/ folder next to this script."""
    if not EXAMPLES_DIR.exists():
        return []
    return sorted(
        p for p in EXAMPLES_DIR.iterdir() if p.suffix.lower() in EXAMPLE_EXTENSIONS
    )


def load_default_class_names():
    """Human-readable class names from labels.json, e.g. {"0": "goldfish", ...}."""
    if not LABELS_PATH.exists():
        return None
    with open(LABELS_PATH, "r") as f:
        raw = json.load(f)
    try:
        ordered = [raw[str(i)] for i in range(NUM_CLASSES)]
    except KeyError:
        return None
    return [name.replace("_", " ").title() for name in ordered]


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = ModelArchitecture(num_classes=NUM_CLASSES)
    state_dict = joblib.load(WEIGHTS_PATH)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess(img: Image.Image) -> torch.Tensor:
    """Same eval-time pipeline used in train.py: resize -> center crop -> normalize."""
    pipeline = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return pipeline(img.convert("RGB")).unsqueeze(0)


@torch.no_grad()
def predict(model, img: Image.Image):
    x = preprocess(img)
    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0)
    top_probs, top_idx = torch.topk(probs, k=min(5, NUM_CLASSES))
    pred_class = int(probs.argmax().item())
    return pred_class, top_idx.tolist(), top_probs.tolist()


# --------------------------------------------------------------------------
# Distortions (mirrors the manipulations listed in README.txt)
# --------------------------------------------------------------------------
def apply_distortions(img: Image.Image, opts: dict) -> Image.Image:
    out = img.convert("RGB")

    if opts["h_flip"]:
        out = ImageOps.mirror(out)
    if opts["v_flip"]:
        out = ImageOps.flip(out)
    if opts["rotation"] != 0:
        # PIL rotates counter-clockwise for positive angles, which feels
        # backwards next to a left-to-right slider — negate so dragging
        # the slider right visually rotates the image clockwise.
        out = out.rotate(-opts["rotation"], expand=True, fillcolor=(127, 127, 127))
    if opts["affine"]:
        w, h = out.size
        shear = opts["shear"] / 100.0
        coeffs = (1, shear, -shear * h / 2, 0, 1, 0)
        out = out.transform(out.size, Image.AFFINE, coeffs, fillcolor=(127, 127, 127))
    if opts["grayscale"]:
        out = ImageOps.grayscale(out).convert("RGB")
    if opts["invert"]:
        out = ImageOps.invert(out)
    if opts["blur"] > 0:
        from PIL import ImageFilter

        out = out.filter(ImageFilter.GaussianBlur(radius=opts["blur"]))
    if opts["brightness"] != 1.0 or opts["contrast"] != 1.0 or opts["saturation"] != 1.0:
        from PIL import ImageEnhance

        out = ImageEnhance.Brightness(out).enhance(opts["brightness"])
        out = ImageEnhance.Contrast(out).enhance(opts["contrast"])
        out = ImageEnhance.Color(out).enhance(opts["saturation"])

    return out


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🖼️ IML Hackathon — Image Classifier")

tab_arch, tab_try = st.tabs(["📐 Architecture", "🎛️ Try It"])

# --- Tab 1: Architecture ---------------------------------------------------
with tab_arch:
    st.header("Network Architecture")
    st.markdown(
        """
The network is built from **5 stages**. Each stage repeats the same 5 steps:

1. Convolution
2. Batch normalization
3. ReLU activation
4. Residual block (`ResBlock`) — two conv/BN layers with a skip connection,
   which helps gradients flow through the deeper network
5. Max pooling (downsamples spatial dimensions by 2×)

After the 5 stages, an **adaptive average pool** collapses the remaining
7×7 feature map to 1×1, and a **dense classifier head**
(`Linear → BatchNorm → ReLU → Dropout → Linear`) maps the resulting
512-dimensional feature vector to the 20 output classes.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Stage output shapes")
        st.table(
            {
                "Stage": [1, 2, 3, 4, 5],
                "Channels": [32, 64, 128, 256, 512],
                "Spatial size": ["112×112", "56×56", "28×28", "14×14", "7×7"],
            }
        )
    with col2:
        st.subheader("Training strategy")
        st.markdown(
            """
- **Phase 1:** trained from scratch with light augmentations
- **Phase 2:** fine-tuned with heavier augmentations, while checking
  the model still recognized unmodified images

**Augmentations used:** horizontal flip, vertical flip, rotation (up to
15°), affine projections, grayscale, color jitter, color inversion,
Gaussian blur.
            """
        )

    st.subheader("Reported results")
    st.table(
        {
            "": ["Pre fine-tuning", "Post fine-tuning"],
            "Unmodified test accuracy": ["88.4%", "89.2%"],
            "Modified/distorted test accuracy": ["63.5%", "81.7%"],
        }
    )
    st.caption(
        "Fine-tuning traded a small amount of clean accuracy for a large "
        "gain in robustness to distortions."
    )

# --- Tab 2: Try It -----------------------------------------------------
with tab_try:
    st.header("Upload an image and test the model")

    examples = list_example_images()
    source = "Upload my own"
    if examples:
        source = st.radio(
            "Image source",
            ["Upload my own", "Use an example"],
            horizontal=True,
        )

    uploaded_file = None
    example_choice = None
    if source == "Use an example":
        example_choice = st.selectbox(
            "Choose an example image", examples, format_func=lambda p: p.name
        )
    else:
        uploaded_file = st.file_uploader(
            "Choose an image", type=["png", "jpg", "jpeg", "bmp", "webp"]
        )

    with st.sidebar:
        st.header("Distortion controls")
        opts = {
            "h_flip": st.checkbox("Horizontal flip"),
            "v_flip": st.checkbox("Vertical flip"),
            "rotation": range_slider(
                "Rotation (degrees)", -180, 180, value=0, step=5, key="rotation"
            ),
            "affine": st.checkbox("Affine shear"),
            "shear": range_slider(
                "Shear amount", -50, 50, value=0, step=5, key="shear"
            ),
            "grayscale": st.checkbox("Grayscale"),
            "invert": st.checkbox("Invert colors"),
            "blur": range_slider(
                "Gaussian blur radius", 0, 10, value=0, step=0.5, key="blur"
            ),
            "brightness": range_slider(
                "Brightness", 0.2, 2.0, value=1.0, step=0.1, key="brightness"
            ),
            "contrast": range_slider(
                "Contrast", 0.2, 2.0, value=1.0, step=0.1, key="contrast"
            ),
            "saturation": range_slider(
                "Saturation", 0.0, 2.0, value=1.0, step=0.1, key="saturation"
            ),
        }

    default_names = load_default_class_names()
    class_names_input = st.sidebar.text_input(
        "Override class names (optional, comma-separated, 20 values)",
        help="Leave blank to use the bundled labels.json names."
        if default_names
        else "No labels.json found — leave blank to show class indices 0–19.",
    )
    class_names = default_names
    if class_names_input.strip():
        names = [n.strip() for n in class_names_input.split(",")]
        if len(names) == NUM_CLASSES:
            class_names = names
        else:
            st.sidebar.warning(f"Expected {NUM_CLASSES} names, got {len(names)}. Ignoring.")

    if uploaded_file is not None or example_choice is not None:
        if uploaded_file is not None:
            original_img = Image.open(io.BytesIO(uploaded_file.read()))
        else:
            original_img = Image.open(example_choice)
        distorted_img = apply_distortions(original_img, opts)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(original_img, use_container_width=True)
        with col2:
            st.subheader("Distorted (fed to model)")
            st.image(distorted_img, use_container_width=True)

        model = load_model()
        pred_class, top_idx, top_probs = predict(model, distorted_img)

        def label(i):
            return class_names[i] if class_names else f"Class {i}"

        st.subheader("Prediction")
        st.metric("Predicted class", label(pred_class))

        st.subheader("Top-5 probabilities")
        st.bar_chart(
            {label(i): p for i, p in zip(top_idx, top_probs)}
        )
    else:
        st.info("Upload an image above (or pick an example) to run it through the model.")
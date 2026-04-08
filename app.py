"""
Handwritten Digit & Operator Classifier
========================================
Upload a handwritten image → classify as digit (0–9) or operator (+, −, ×, ÷, =)

Place these files next to app.py:
    cnn_model.keras
    label_encoder.pkl

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io, os, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import streamlit as st
from PIL import Image

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = "cnn_model.keras"
ENCODER_PATH = "label_encoder.pkl"

# ── Class display config ──────────────────────────────────────────────────────
CLASS_INFO = {
    # Digits
    "0": {"emoji": "0️⃣",  "group": "digit",    "color": "#1565c0"},
    "1": {"emoji": "1️⃣",  "group": "digit",    "color": "#1565c0"},
    "2": {"emoji": "2️⃣",  "group": "digit",    "color": "#1565c0"},
    "3": {"emoji": "3️⃣",  "group": "digit",    "color": "#1565c0"},
    "4": {"emoji": "4️⃣",  "group": "digit",    "color": "#1565c0"},
    "5": {"emoji": "5️⃣",  "group": "digit",    "color": "#1565c0"},
    "6": {"emoji": "6️⃣",  "group": "digit",    "color": "#1565c0"},
    "7": {"emoji": "7️⃣",  "group": "digit",    "color": "#1565c0"},
    "8": {"emoji": "8️⃣",  "group": "digit",    "color": "#1565c0"},
    "9": {"emoji": "9️⃣",  "group": "digit",    "color": "#1565c0"},
    # Operators
    "+": {"emoji": "➕",   "group": "operator", "color": "#2e7d32"},
    "-": {"emoji": "➖",   "group": "operator", "color": "#c62828"},
    "*": {"emoji": "✖️",   "group": "operator", "color": "#6a1b9a"},
    "/": {"emoji": "➗",   "group": "operator", "color": "#e65100"},
    "=": {"emoji": "🟰",   "group": "operator", "color": "#00695c"},
    # common alternate spellings
    "mul": {"emoji": "✖️", "group": "operator", "color": "#6a1b9a"},
    "div": {"emoji": "➗", "group": "operator", "color": "#e65100"},
    "add": {"emoji": "➕", "group": "operator", "color": "#2e7d32"},
    "sub": {"emoji": "➖", "group": "operator", "color": "#c62828"},
    "eq":  {"emoji": "🟰", "group": "operator", "color": "#00695c"},
}
FALLBACK_INFO = {"emoji": "❓", "group": "unknown", "color": "#455a64"}


def get_info(label: str):
    key = label.strip().lower()
    return CLASS_INFO.get(label, CLASS_INFO.get(key, FALLBACK_INFO))


# ── Preprocessing (exact same pipeline as notebook) ──────────────────────────
def preprocess(pil_image: Image.Image, target_h: int, target_w: int) -> np.ndarray:
    img = np.array(pil_image.convert("L"))
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img = clahe.apply(img)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (img.astype(np.float32) / 255.0).reshape(1, target_h, target_w, 1)


# ── Load model (cached) ───────────────────────────────────────────────────────
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not TF_AVAILABLE:
        return None, None, "TensorFlow not installed."
    if not os.path.exists(MODEL_PATH):
        return None, None, f"`{MODEL_PATH}` not found next to app.py."
    if not os.path.exists(ENCODER_PATH):
        return None, None, f"`{ENCODER_PATH}` not found next to app.py."
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        return None, None, f"Model error: {e}"
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return model, le, None


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digit & Operator Classifier",
    page_icon="🔢",
    layout="centered",
)

st.markdown("""
<style>
#MainMenu, footer { visibility: hidden; }

/* Upload zone */
[data-testid="stFileUploader"] {
    border: 2.5px dashed #90caf9;
    border-radius: 16px;
    padding: 10px;
    background: #f0f7ff;
}

/* Result card */
.result-card {
    border-radius: 20px;
    padding: 32px 24px;
    text-align: center;
    margin-bottom: 16px;
    border: 2px solid rgba(0,0,0,0.07);
}
.big-symbol   { font-size: 88px; line-height: 1.1; }
.pred-label   { font-size: 42px; font-weight: 800; margin-top: 6px; letter-spacing: 2px; }
.group-badge  {
    display: inline-block;
    padding: 3px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    margin-top: 8px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.conf-text    { font-size: 18px; margin-top: 10px; font-weight: 600; }

/* Image preview card */
.img-card {
    background: #1a1a2e;
    border-radius: 16px;
    padding: 16px;
    text-align: center;
}
.img-label { color: #aaa; font-size: 12px; margin-top: 8px; }

/* Progress bar labels */
.bar-label { font-size: 22px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("🔢 Digit & Operator Classifier")
st.markdown(
    "Upload a handwritten image and the model will classify it as a "
    "**digit (0–9)** or **operator (+  −  ×  ÷  =)**."
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
model, le, load_err = load_model()

if load_err:
    st.error(f"⚠️ {load_err}")
    st.info(
        "**Place these two files next to `app.py` and restart:**\n\n"
        "- `cnn_model.keras`\n"
        "- `label_encoder.pkl`\n\n"
        "Export from Colab:\n"
        "```python\n"
        "import pickle\n"
        "model.save('cnn_model.keras')\n"
        "with open('label_encoder.pkl','wb') as f: pickle.dump(le, f)\n"
        "```"
    )
    st.stop()

# Auto-detect input shape from model
inp_shape = model.input_shape   # (None, H, W, 1)
TARGET_H, TARGET_W = inp_shape[1], inp_shape[2]

# ─────────────────────────────────────────────────────────────────────────────
#  FILE UPLOADER
# ─────────────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "**Upload a handwritten image**",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    label_visibility="visible",
)

# ── Empty state ───────────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown("<br>", unsafe_allow_html=True)

    # Show sample class grid
    st.markdown("##### What can this model classify?")

    digits    = [str(i) for i in range(10)]
    operators = ["+", "-", "*", "/", "="]

    col_d, col_o = st.columns(2)
    with col_d:
        st.markdown(
            "<div style='background:#e3f2fd;border-radius:14px;padding:16px;text-align:center'>"
            "<div style='font-size:13px;font-weight:700;color:#1565c0;letter-spacing:1px;"
            "margin-bottom:10px'>DIGITS</div>"
            "<div style='font-size:26px;letter-spacing:4px'>"
            + "  ".join(digits) +
            "</div></div>",
            unsafe_allow_html=True,
        )
    with col_o:
        display_ops = {"+":" ➕ ", "-":" ➖ ", "*":" ✖️ ", "/":" ➗ ", "=":" 🟰 "}
        st.markdown(
            "<div style='background:#e8f5e9;border-radius:14px;padding:16px;text-align:center'>"
            "<div style='font-size:13px;font-weight:700;color:#2e7d32;letter-spacing:1px;"
            "margin-bottom:10px'>OPERATORS</div>"
            "<div style='font-size:28px'>"
            + "".join(display_ops.values()) +
            "</div></div>",
            unsafe_allow_html=True,
        )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  PREDICT
# ─────────────────────────────────────────────────────────────────────────────
pil_img = Image.open(uploaded)

with st.spinner("Classifying…"):
    tensor = preprocess(pil_img, TARGET_H, TARGET_W)
    preds  = model.predict(tensor, verbose=0)[0]

top_idx   = int(np.argmax(preds))
top_label = le.classes_[top_idx]
top_conf  = float(preds[top_idx]) * 100
info      = get_info(top_label)

# ─────────────────────────────────────────────────────────────────────────────
#  RESULT LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
st.divider()

left, right = st.columns([1, 1], gap="large")

# ── Left: image previews ──────────────────────────────────────────────────────
with left:
    st.markdown("##### 🖼️ Uploaded Image")
    st.image(pil_img, use_container_width=True)

    # Preprocessed preview
    prev = (tensor.squeeze() * 255).astype(np.uint8)
    st.markdown("##### 🔬 Preprocessed")
    st.image(
        prev,
        use_container_width=True,
        caption=f"Resized to {TARGET_H}×{TARGET_W} · CLAHE + Otsu",
        clamp=True,
    )

# ── Right: result card ────────────────────────────────────────────────────────
with right:
    st.markdown("##### 🎯 Prediction")

    # Background colour based on group
    bg_color = "#e3f2fd" if info["group"] == "digit" else "#e8f5e9" if info["group"] == "operator" else "#f5f5f5"
    badge_bg = "#1565c0" if info["group"] == "digit" else "#2e7d32"

    conf_colour = (
        "#2e7d32" if top_conf >= 70
        else "#e65100" if top_conf >= 40
        else "#c62828"
    )

    st.markdown(
        f"<div class='result-card' style='background:{bg_color}'>"
        f"  <div class='big-symbol'>{info['emoji']}</div>"
        f"  <div class='pred-label' style='color:{info['color']}'>{top_label}</div>"
        f"  <div>"
        f"    <span class='group-badge' style='background:{badge_bg};color:white'>"
        f"      {info['group']}"
        f"    </span>"
        f"  </div>"
        f"  <div class='conf-text' style='color:{conf_colour}'>{top_conf:.1f}% confident</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIDENCE BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("##### 📊 All Class Confidences")

sorted_idx = np.argsort(preds)[::-1]
top_n      = min(10, len(le.classes_))   # show top 10

for rank, i in enumerate(sorted_idx[:top_n]):
    lbl  = le.classes_[i]
    prob = float(preds[i])
    inf  = get_info(lbl)
    is_top = (rank == 0)

    col_emoji, col_lbl, col_bar = st.columns([0.5, 0.8, 5])
    with col_emoji:
        st.markdown(
            f"<div style='font-size:{'28' if is_top else '20'}px;"
            f"text-align:center;padding-top:4px'>{inf['emoji']}</div>",
            unsafe_allow_html=True,
        )
    with col_lbl:
        weight = "800" if is_top else "400"
        st.markdown(
            f"<div style='font-weight:{weight};font-size:15px;"
            f"color:{inf['color']};padding-top:6px'>{lbl}</div>",
            unsafe_allow_html=True,
        )
    with col_bar:
        st.progress(prob, text=f"{prob*100:.1f}%")

if len(le.classes_) > top_n:
    with st.expander(f"Show all {len(le.classes_)} classes"):
        for i in sorted_idx[top_n:]:
            lbl  = le.classes_[i]
            prob = float(preds[i])
            inf  = get_info(lbl)
            c1, c2, c3 = st.columns([0.5, 0.8, 5])
            with c1:
                st.markdown(f"<div style='font-size:18px;text-align:center'>{inf['emoji']}</div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div style='font-size:14px'>{lbl}</div>", unsafe_allow_html=True)
            with c3:
                st.progress(prob, text=f"{prob*100:.1f}%")

st.divider()
st.caption(f"Model input: {TARGET_H}×{TARGET_W}px · Preprocessing: CLAHE + Otsu · Built with Streamlit")

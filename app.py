import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import io

# ─── Streamlit page config (must be first Streamlit call) ────────────────────
st.set_page_config(
    page_title="Parts Classifier",
    page_icon="🔩",
    layout="wide"
)

# ─── Settings ────────────────────────────────────────────────────────────────
MODEL_PATH = "mobilenetv3_best.pth"
IMG_SIZE   = (224, 224)

@st.cache_resource(show_spinner="Loading model… 🧠")
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model = models.mobilenet_v3_large(weights=None)
    in_feats = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_feats, len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model, checkpoint["classes"]

model, CLASS_NAMES = load_model()

# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("🔩🆚🔧 Mechanical Parts Image Classifier")
st.markdown(
    "Upload a pic of a **bolt, nut, washer, or locating pin** and I'll guess what it is."
)

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Your image", use_column_width=True)

    # Transform
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    tensor = transform(img).unsqueeze(0)

    # Predict
    with torch.no_grad():
        logits = model(tensor)
    pred_idx = logits.argmax(1).item()
    prob     = torch.softmax(logits, dim=1)[0, pred_idx].item() * 100

    st.success(
        f"### 🧠 Prediction: **{CLASS_NAMES[pred_idx]}**\n"
        f"Confidence: **{prob:.2f}%**"
    )

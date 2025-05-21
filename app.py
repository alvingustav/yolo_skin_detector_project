import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# -------------------- CONFIG --------------------
st.set_page_config(page_title="YOLOv8 Skin Disease Detection", layout="centered")

# -------------------- CUSTOM CSS --------------------
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
    }
    h1 {
        color: #2E8B57;
        text-align: center;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 18px;
        font-weight: 600;
        color: #2E8B57;
    }
    .stFileUploader label {
        font-weight: 500;
    }
    .stAlert > div {
        background-color: #e6ffed;
        border-left: 6px solid #2E8B57;
    }
    .stImage img {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- CONSTANT --------------------
CLASSES = ['Acne', 'Moles', 'Sun_Sunlight_Damage', 'Infestations_Bites']

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return YOLO("my_model1.pt")

model = load_model()

# -------------------- VIDEO TRANSFORMER --------------------
class YOLOTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)[0]

        for r in results.boxes:
            cls_id = int(r.cls[0])
            label = CLASSES[cls_id] if cls_id < len(CLASSES) else f"class {cls_id}"
            conf = float(r.conf[0])
            xyxy = r.xyxy[0].cpu().numpy().astype(int)

            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(img, f"{label} ({conf:.2f})", (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return img

# -------------------- HEADER --------------------
st.markdown("<h1>🧬 Skin Disease Detector</h1>", unsafe_allow_html=True)
st.write("---")

# -------------------- TABS UI --------------------
tab1, tab2, tab3 = st.tabs(["📷 Kamera", "🖼️ Upload Gambar", "ℹ️ Tentang"])

# ----------- TAB 1: Kamera -----------
with tab1:
    st.subheader("📷 Akses Kamera (WebRTC)")
    st.info("Berikan izin kamera saat diminta. Buka dari browser Chrome / Edge / Firefox.")
    webrtc_streamer(
        key="yolo-camera",
        video_transformer_factory=YOLOTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# ----------- TAB 2: Upload Gambar -----------
with tab2:
    st.subheader("🖼️ Upload Gambar Kulit")
    uploaded_file = st.file_uploader("Unggah gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        st.image(image, caption="Gambar Asli", use_column_width=True)

        results = model(img_np)[0]

        for r in results.boxes:
            cls_id = int(r.cls[0])
            label = CLASSES[cls_id] if cls_id < len(CLASSES) else f"class {cls_id}"
            conf = float(r.conf[0])
            xyxy = r.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(img_np, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
            cv2.putText(img_np, f"{label} ({conf:.2f})", (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        st.image(img_np, caption="Hasil Deteksi", use_column_width=True)

# ----------- TAB 3: Tentang -----------
with tab3:
    st.subheader("ℹ️ Tentang Aplikasi")
    st.markdown("""
Aplikasi ini menggunakan model **YOLOv8** untuk mendeteksi beberapa jenis masalah kulit:

- **Acne**
- **Moles**
- **Sunlight Damage**
- **Infestations or Bites**

💡 Teknologi yang digunakan:
- `Streamlit` sebagai UI
- `ultralytics` untuk YOLOv8
- `streamlit-webrtc` untuk kamera real-time

⚠️ **Disclaimer:**  
Aplikasi ini hanya untuk tujuan edukasi dan bukan alat diagnosis medis. Selalu konsultasikan dengan dokter profesional untuk diagnosis resmi.
""")

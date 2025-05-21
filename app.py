import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# -------------------- CONFIG --------------------
st.set_page_config(page_title="YOLOv8 Skin Disease Detection", layout="centered")
CLASSES = ['Acne', 'Moles', 'Sun_Sunlight_Damage', 'Infestations_Bites']

# -------------------- DARK MODE + STYLE --------------------
st.markdown("""
    <style>
    html, body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f5f7fa;
        color: #000000;
    }
    h1 {
        text-align: center;
        color: #2E8B57;
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

    @media (prefers-color-scheme: dark) {
        html, body {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        h1 {
            color: #90ee90 !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: #90ee90 !important;
        }
        .stAlert > div {
            background-color: #334d33 !important;
            border-left: 6px solid #90ee90 !important;
        }
        .stFileUploader label {
            color: #ffffff !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- LOAD YOLO MODEL --------------------
@st.cache_resource
def load_model():
    return YOLO("my_model1.pt")

model = load_model()

# -------------------- VIDEO PROCESSOR --------------------
class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame):
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

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -------------------- UI --------------------
st.markdown("<h1>🧬 Skin Disease Detector</h1>", unsafe_allow_html=True)
st.write("---")

tab1, tab2, tab3 = st.tabs(["📷 Kamera", "🖼️ Upload Gambar", "ℹ️ Tentang"])

# ----------- TAB 1: Kamera -----------
with tab1:
    st.subheader("📷 Akses Kamera (WebRTC)")
    st.info("Berikan izin kamera saat diminta. Gunakan Chrome / Edge / Firefox.")
    webrtc_streamer(
        key="yolo-camera",
        video_processor_factory=YOLOProcessor,
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
- `Streamlit` sebagai UI interaktif
- `ultralytics` untuk YOLOv8
- `streamlit-webrtc` untuk akses kamera langsung

⚠️ **Disclaimer:**  
Aplikasi ini hanya untuk edukasi. Untuk diagnosis resmi, harap konsultasikan dengan tenaga medis profesional.
""")

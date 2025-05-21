import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
from ultralytics import YOLO
import numpy as np

# =================== CONFIG ===================
st.set_page_config(page_title="Skin Disease Detection", layout="centered")

# =================== HEADER ===================
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🧬 Skin Disease Detector</h1>
    <h4 style='text-align: center;'>Powered by YOLOv8 + Webcam</h4>
""", unsafe_allow_html=True)

st.write("---")

# =================== MODEL ===================
@st.cache_resource
def load_model():
    model = YOLO("my_model1.pt")
    return model

model = load_model()
CLASSES = ['Acne', 'Moles', 'Sun_Sunlight_Damage', 'Infestations_Bites']

# =================== VIDEO TRANSFORMER ===================
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

# =================== MAIN STREAM ===================
st.subheader("📷 Akses Kamera dari Browser")
st.info("Berikan izin kamera saat diminta. Gunakan browser di HP/laptop yang mendukung WebRTC (Chrome, Edge, Firefox).")

webrtc_streamer(
    key="yolo-webrtc",
    video_transformer_factory=YOLOTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
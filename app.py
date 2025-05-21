import streamlit as st
import av
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import time

# Config
st.set_page_config(page_title="YOLOv8 Streamlit Detector", layout="centered")
CLASSES = ['Acne', 'Moles', 'Sun_Sunlight_Damage', 'Infestations_Bites']
model_path = "my_model1.pt"
model = YOLO(model_path)

# Custom style + full screen video
st.markdown("""
    <style>
    html, body {
        background-color: #f5f7fa;
    }
    h1 {
        text-align: center;
        color: #2E8B57;
    }
    video {
        width: 100% !important;
        height: 80vh !important;
        object-fit: cover;
        border-radius: 12px;
        box-shadow: 0 0 10px rgba(0,0,0,0.3);
    }
    @media (prefers-color-scheme: dark) {
        html, body {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        h1 {
            color: #90ee90 !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🧬 YOLOv8 Skin Detector</h1>", unsafe_allow_html=True)
st.write("---")

tab1, tab2 = st.tabs(["📷 Kamera", "🖼️ Upload Gambar"])

# =============== Webcam Processor ===============
class YOLOProcessor(VideoProcessorBase):
    def __init__(self):
        self.fps_window = []
        self.start_time = time.perf_counter()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        t0 = time.perf_counter()

        results = model(img)[0]
        object_count = 0

        for r in results.boxes:
            cls_id = int(r.cls[0])
            label = CLASSES[cls_id] if cls_id < len(CLASSES) else f"class {cls_id}"
            conf = float(r.conf[0])
            xyxy = r.xyxy[0].cpu().numpy().astype(int)
            if conf > 0.5:
                object_count += 1
                cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                cv2.putText(img, f"{label}: {int(conf*100)}%", (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        fps = 1.0 / (time.perf_counter() - t0)
        self.fps_window.append(fps)
        if len(self.fps_window) > 30:
            self.fps_window.pop(0)

        avg_fps = sum(self.fps_window) / len(self.fps_window)
        cv2.putText(img, f'FPS: {avg_fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(img, f'Objects: {object_count}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# =============== TAB 1: Webcam ===============
with tab1:
    st.info("Berikan izin kamera dan gunakan browser modern (Chrome/Edge/Firefox)")
    webrtc_streamer(
        key="webcam-yolo",
        video_processor_factory=YOLOProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# =============== TAB 2: Upload Image ===============
with tab2:
    uploaded = st.file_uploader("Unggah gambar (jpg, png)", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_np = np.array(img)
        results = model(img_np)[0]

        object_count = 0
        for r in results.boxes:
            cls_id = int(r.cls[0])
            label = CLASSES[cls_id] if cls_id < len(CLASSES) else f"class {cls_id}"
            conf = float(r.conf[0])
            xyxy = r.xyxy[0].cpu().numpy().astype(int)
            if conf > 0.5:
                object_count += 1
                cv2.rectangle(img_np, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
                cv2.putText(img_np, f"{label}: {int(conf*100)}%", (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        st.image(img, caption=f"Hasil Deteksi ({object_count} objek)", use_column_width=True)

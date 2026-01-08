import streamlit as st
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import time

st.title("Comparison of Pepper Seed Detection Using Two YOLOv8 Models")

# Path model
model1_path = "/home/comvislab/AI-Jetson/Syauqi/Model/Tanpa CLAHE 5 224x224.pt"
model2_path = "/home/comvislab/AI-Jetson/Syauqi/Model/CLAHE 5 224x224.pt"

# Load kedua model
model1 = YOLO(model1_path)
model2 = YOLO(model2_path)

CLASS_COLORS = {
    0: (0, 170, 0),   # lada normal
    1: (255, 0, 0)    # lada rusak
}

# Upload hingga 10 gambar
uploaded_files = st.file_uploader("Upload images for detection (maximum of 10 images)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files[:10]:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        img_resized = cv2.resize(img_np, (640, 640))

        col1, col2 = st.columns(2)
        with st.spinner(f"Memproses: {uploaded_file.name}..."):

            # Deteksi dengan Model 1
            start_time_1 = time.time()
            results1 = model1(img_resized)
            end_time_1 = time.time()

            # Deteksi dengan Model 2
            start_time_2 = time.time()
            results2 = model2(img_resized)
            end_time_2 = time.time()

            def draw_boxes(result, img, model):
                boxes = result.boxes.xyxy
                confs = result.boxes.conf
                class_ids = result.boxes.cls.int().tolist()
                annotated = img.copy()
                for box, conf, class_id in zip(boxes, confs, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = model.names[class_id]
                    color = CLASS_COLORS.get(class_id, (255, 255, 255))
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                    label = f"{class_name} ({conf:.2f})"
                    cv2.putText(annotated, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                return annotated

            result1 = results1[0]
            result2 = results2[0]
            annotated1 = draw_boxes(result1, img_resized, model1)
            annotated2 = draw_boxes(result2, img_resized, model2)

            # Tampilkan gambar dengan hasil masing-masing model
            with col1:
                st.image(annotated1)
                st.markdown(
                    f"""
                    <p style='text-align: center; font-weight: bold;'>Without CLAHE - {uploaded_file.name}</p>
                    <p style='text-align: center;'>Time: {end_time_1 - start_time_1:.2f} Second</p>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.image(annotated2)
                st.markdown(
                    f"""
                    <p style='text-align: center; font-weight: bold;'>CLAHE - {uploaded_file.name}</p>
                    <p style='text-align: center;'>Time: {end_time_2 - start_time_2:.2f} Second</p>
                    """,
                    unsafe_allow_html=True
                )



st.markdown(
    """
    <style>
    /* Latar belakang keseluruhan dengan gradasi biru muda di kiri dan kanan */
    .stApp {
        background: linear-gradient(to right, #add8e6 0%, white 20%, white 80%, #add8e6 100%);
        color: black;
    }
    /* teks hitam */
    .stApp {
        background-color: white;
        color: black;
    }

    /* teks putih pada kotak drag and drop  */
    .stFileUploader > div > div {
        color: white !important;
    }

    .stFileUploader {
        background-color: #999999 !important;
        border-radius: 10px;
        padding: 10px;
    }
    img {
        max-width: 100% !important;
        height: auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)



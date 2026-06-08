import os
import streamlit as st
import cv2
import easyocr
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from groq import Groq
from PIL import Image
import fitz  # PyMuPDF

# ==========================================
# 1. SETUP MODEL & API
# ==========================================
@st.cache_resource
def load_models():
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    reader = easyocr.Reader(['id', 'en'])
    model_path = "./models/yolo26m_doc_layout.pt"
    model = YOLO(model_path)
    return client, reader, model

@st.cache_data
def run_detection_and_ocr(img_array, conf_threshold):
    # Konversi untuk OpenCV
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Deteksi dengan threshold dinamis
    results = model(img_cv, conf=conf_threshold)
    boxes = results[0].boxes
    
    # Plotting hasil deteksi
    img_with_boxes = results[0].plot()
    annotated_img = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    
    # OCR Extraction
    sorted_boxes = sorted(boxes, key=lambda b: b.xyxy[0][1].item())
    page_text = []
    for box in sorted_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = img_cv[y1:y2, x1:x2]
        text_list = reader.readtext(cropped, detail=0)
        if text_list:
            page_text.append(" ".join(text_list))
            
    return annotated_img, "\n".join(page_text)

client, reader, model = load_models()

# ==========================================
# 2. TAMPILAN ANTARMUKA (UI)
# ==========================================
st.set_page_config(page_title="DocAI Analyzer", page_icon="📄", layout="wide")

st.markdown("""
    <style>
        .main-title { font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem; }
        @media (max-width: 768px) {
            .main-title { font-size: 1.5rem; line-height: 1.2; }
            .stButton > button { width: 100%; }
            .block-container { padding-top: 2rem; }
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Deteksi Struktur Document Dengan YOLOv26 dan Menggunakan LLaMA 3.3 Untuk Ekstraksi dan Rangkuman Isi Dokumen</div>', unsafe_allow_html=True)
st.write("Unggah dokumen Anda, dan AI akan otomatis menganalisis serta merangkum isinya.")

# Sidebar untuk Threshold
st.sidebar.header("Pengaturan Deteksi")
conf_threshold = st.sidebar.slider("Sensitivitas Deteksi (Threshold)", 0.0, 1.0, 0.25, 0.05)

uploaded_file = st.file_uploader("Unggah Dokumen (PDF, JPG, PNG)", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner('Sedang memproses dokumen...'):
        images_to_process = []
        if uploaded_file.name.lower().endswith('.pdf'):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in doc:
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images_to_process.append(img)
        else:
            images_to_process.append(Image.open(uploaded_file))

        all_extracted_text = ""
        annotated_images = []

        for img in images_to_process:
            ann_img, text = run_detection_and_ocr(np.array(img), conf_threshold)
            annotated_images.append(ann_img)
            all_extracted_text += text + "\n"

        # Tampilkan Hasil Visual
        st.subheader("Visualisasi Deteksi Layout")
        num_cols = min(len(annotated_images), 5)
        cols = st.columns(num_cols)
        for i, ann_img in enumerate(annotated_images):
            cols[i % num_cols].image(ann_img, use_container_width=True)
        
        st.divider()

        # Hasil Teks
        if all_extracted_text.strip():
            st.subheader("Hasil Ekstraksi Teks")
            st.code(all_extracted_text, language='text')
            
            st.divider()
            
            # Tombol untuk Rangkuman (Agar tidak boros API)
            if st.button("Buat Rangkuman dengan LLaMA 3.3"):
                with st.spinner('Menghubungi AI...'):
                    prompt = f"Tolong jelaskan secara ringkas isi dari dokumen berikut:\n\n{all_extracted_text}"
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama-3.3-70b-versatile",
                        temperature=0.5,
                    )
                    st.subheader("Penjelasan AI:")
                    st.write(chat_completion.choices[0].message.content)
        else:
            st.error("Gagal mengekstrak teks.")

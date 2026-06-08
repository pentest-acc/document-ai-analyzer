import os
import io
import streamlit as st
import cv2
import easyocr
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from groq import Groq
from PIL import Image
import fitz

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

client, reader, model = load_models()

# ==========================================
# 2. TAMPILAN ANTARMUKA (UI)
# ==========================================
st.set_page_config(page_title="DocAI Analyzer", page_icon="📄", layout="wide")

st.markdown("""
    <style>
        .main-title { font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem; }
        @media (max-width: 768px) {
            .block-container { padding-top: 5rem !important; }
            .main-title { font-size: 1.3rem; line-height: 1.2; margin-top: 10px; }
            .stButton > button { width: 100%; }
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.header("⚙️ Pengaturan Deteksi")
conf_threshold = st.sidebar.slider("Sensitivitas Deteksi (Threshold)", 0.0, 1.0, 0.25, 0.05)

st.markdown('<div class="main-title">Deteksi Struktur Document Dengan YOLOv26 dan LLaMA 3.1 Untuk Ekstraksi dan Rangkuman</div>', unsafe_allow_html=True)

if "analysis_started" not in st.session_state: st.session_state.analysis_started = False
if "current_file" not in st.session_state: st.session_state.current_file = None
if "current_threshold" not in st.session_state: st.session_state.current_threshold = conf_threshold

uploaded_file = st.file_uploader("Unggah Dokumen (PDF, JPG, PNG)", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    if st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.analysis_started = False
        if "analysis_results" in st.session_state: del st.session_state["analysis_results"]

    if st.session_state.current_threshold != conf_threshold:
        st.session_state.current_threshold = conf_threshold
        if "analysis_results" in st.session_state: del st.session_state["analysis_results"]

    if st.button("Mulai Analisis Dokumen", type="primary"):
        st.session_state.analysis_started = True

    if st.session_state.analysis_started:
        if "analysis_results" not in st.session_state:
            with st.spinner('Sedang memproses dokumen...'):
                # (Proses OCR & YOLO tetap sama)
                images_to_process = []
                if uploaded_file.name.lower().endswith('.pdf'):
                    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                    for page in doc:
                        pix = page.get_pixmap(dpi=150)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        images_to_process.append(img)
                else:
                    images_to_process.append(Image.open(uploaded_file))

                all_text = ""
                annotated_images = []
                extracted_pictures = []

                for idx, img in enumerate(images_to_process):
                    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    results = model(img_cv, conf=conf_threshold)
                    boxes = results[0].boxes
                    annotated_images.append(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
                    
                    sorted_boxes = sorted(boxes, key=lambda b: b.xyxy[0][1].item())
                    page_text = []
                    for box in sorted_boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_name = model.names[int(box.cls[0])]
                        if cls_name == "Picture":
                            extracted_pictures.append(Image.fromarray(cv2.cvtColor(img_cv[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)))
                        else:
                            text = reader.readtext(img_cv[y1:y2, x1:x2], detail=0)
                            if text: page_text.append(" ".join(text))
                    all_text += "\n".join(page_text) + "\n"

                # --- PROSES AI DIBUAT SEKALI PANGGIL ---
                with st.spinner("AI sedang bekerja (Koreksi + Rangkuman)..."):
                    prompt = f"""Tugas Anda:
1. Koreksi teks OCR berikut (hapus duplikasi, perbaiki ejaan).
2. Jangan terjemahkan bahasa asing (biarkan aslinya).
3. Buat ringkasan poin utama dari teks tersebut.

Teks Mentah OCR:
{all_text}

Format Output:
### Teks Diperbaiki:
[Hasil Koreksi]

### Ringkasan AI:
[Hasil Ringkasan]
"""
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama-3.1-8b-instant", # Model lebih cepat
                        temperature=0.0
                    )
                    full_response = response.choices[0].message.content
                    
                    # Pemisahan output dari AI
                    parts = full_response.split("### Ringkasan AI:")
                    corrected = parts[0].replace("### Teks Diperbaiki:", "").strip()
                    summary = parts[1].strip() if len(parts) > 1 else "Gagal merangkum."

                    st.session_state.analysis_results = {
                        "annotated_images": annotated_images,
                        "extracted_pictures": extracted_pictures,
                        "all_extracted_text": all_text,
                        "corrected_text": corrected,
                        "summary": summary
                    }

        # Render Hasil
        res = st.session_state.analysis_results
        st.subheader("Visualisasi Deteksi Layout")
        st.image(res["annotated_images"], use_container_width=True)
        
        if res["extracted_pictures"]:
            st.subheader("🖼️ Gambar Ditemukan")
            for i, pic in enumerate(res["extracted_pictures"]):
                st.image(pic)
                st.download_button("Unduh Gambar", data=io.BytesIO(), file_name=f"pic_{i}.png")

        st.subheader("Hasil Ekstraksi")
        tab1, tab2 = st.tabs(["📝 Teks Mentah", "✨ Teks Diperbaiki"])
        with tab1: st.code(res["all_extracted_text"])
        with tab2: st.code(res["corrected_text"])
        st.subheader("Ringkasan AI")
        st.write(res["summary"])

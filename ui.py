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
import fitz  # PyMuPDF

# ==========================================
# 1. SETUP MODEL & API
# ==========================================
@st.cache_resource
def load_models():
    # Menggunakan model yang lebih ringan agar tidak terkena RateLimitError
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

# --- SIDEBAR ---
st.sidebar.header("⚙️ Pengaturan Deteksi")
conf_threshold = st.sidebar.slider("Sensitivitas Deteksi (Threshold)", 0.0, 1.0, 0.25, 0.05)

st.markdown('<div class="main-title">Deteksi Struktur Dokumen & Ekstraksi Menggunakan YOLOv26 & LLaMA 3</div>', unsafe_allow_html=True)
st.write("Unggah dokumen Anda untuk analisis cepat.")

# Inisialisasi Memori (Session State)
if "analysis_started" not in st.session_state: st.session_state.analysis_started = False
if "current_file" not in st.session_state: st.session_state.current_file = None
if "current_threshold" not in st.session_state: st.session_state.current_threshold = conf_threshold

uploaded_file = st.file_uploader("Unggah Dokumen (PDF, JPG, PNG)", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Reset memori jika file baru
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
            with st.spinner('Memproses dokumen...'):
                # 1. OCR dan Deteksi
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
                extracted_pictures = []

                for idx, img in enumerate(images_to_process):
                    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    results = model(img_cv, conf=conf_threshold)
                    annotated_images.append(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
                    
                    sorted_boxes = sorted(results[0].boxes, key=lambda b: b.xyxy[0][1].item())
                    page_text = []
                    for box in sorted_boxes:
                        cls_name = model.names[int(box.cls[0])]
                        cropped = img_cv[int(box.xyxy[0][1]):int(box.xyxy[0][3]), int(box.xyxy[0][0]):int(box.xyxy[0][2])]
                        
                        if cls_name == "Picture":
                            extracted_pictures.append(Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)))
                        else:
                            text = reader.readtext(cropped, detail=0)
                            if text: page_text.append(" ".join(text))
                    
                    all_extracted_text += "\n".join(page_text) + "\n"

                # 2. AI Processing (Koreksi & Rangkuman)
                results_data = {
                    "annotated_images": annotated_images,
                    "extracted_pictures": extracted_pictures,
                    "all_extracted_text": all_extracted_text,
                    "corrected_text": "AI gagal memproses (Limit API).",
                    "summary": "AI gagal memproses (Limit API).",
                    "error": False
                }

                try:
                    # Prompt yang sangat ketat untuk hasil yang diinginkan
                    system_prompt = "Anda adalah pengoreksi teks OCR. 1. JANGAN menambah basa-basi. 2. PERTAHANKAN bahasa asli. 3. HAPUS duplikasi teks ganda. 4. HANYA perbaiki ejaan/tata bahasa."
                    
                    response = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Perbaiki teks ini:\n{all_extracted_text}"}
                        ],
                        model="llama3-8b-8192", # Model cepat
                        temperature=0.0
                    )
                    results_data["corrected_text"] = response.choices[0].message.content

                    summary_response = client.chat.completions.create(
                        messages=[{"role": "user", "content": f"Ringkas teks ini:\n{results_data['corrected_text']}"}],
                        model="llama3-8b-8192",
                        temperature=0.3
                    )
                    results_data["summary"] = summary_response.choices[0].message.content
                except Exception as e:
                    results_data["error"] = True

                st.session_state.analysis_results = results_data

        # 3. Render UI dari Memori
        res = st.session_state.analysis_results
        
        st.subheader("Visualisasi Deteksi Layout")
        st.image(res["annotated_images"], use_container_width=True)
        
        if res["extracted_pictures"]:
            st.subheader("🖼️ Gambar Ditemukan")
            for i, pic in enumerate(res["extracted_pictures"]):
                st.image(pic)
                st.download_button("Unduh Gambar", data=io.BytesIO(), file_name=f"gambar_{i}.png", key=f"btn_{i}")

        if res["error"]:
            st.warning("⚠️ Limit API tercapai. Menampilkan versi mentah.")
        
        st.subheader("Hasil Ekstraksi Teks (Perbandingan)")
        tab1, tab2 = st.tabs(["📝 Teks Mentah (Raw OCR)", "✨ Teks Diperbaiki (AI Corrected)"])
        with tab1: st.code(res["all_extracted_text"])
        with tab2: st.code(res["corrected_text"])
        
        st.subheader("Ringkasan AI")
        st.write(res["summary"])

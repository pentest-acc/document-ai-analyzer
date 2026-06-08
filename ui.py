import streamlit as st
import cv2
import os
import easyocr
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from groq import Groq
from PIL import Image
import fitz  # PyMuPDF

# ==========================================
# 1. SETUP MODEL & API
# ==========================================
@st.cache_resource
def load_models():
    # 1. Mengambil API Key dari pengaturan rahasia Streamlit Cloud
    # Jika Anda masih mencoba secara lokal di laptop, gunakan ini sementara:
    # client = Groq(api_key="gsk_oKeX6Bh1FxAqHCp...") 
    # Tapi jika sudah siap upload ke GitHub, gunakan ini:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    
    # 2. Inisialisasi EasyOCR
    reader = easyocr.Reader(['id', 'en'])
    
    # 3. Mengunduh otomatis model YOLOv26 dari Hugging Face
    DOWNLOAD_PATH = Path("./models")
    DOWNLOAD_PATH.mkdir(exist_ok=True)
    
    # Memilih versi medium (m) sesuai dengan file yang Anda gunakan sebelumnya
    selected_model_file = "yolo26m_doc_layout.pt" 
    
    model_path = hf_hub_download(
        repo_id="Armaggheddon/yolo26-document-layout",
        filename=selected_model_file,
        repo_type="model",
        local_dir=DOWNLOAD_PATH,
    )
    
    # 4. Inisialisasi Model YOLO
    model = YOLO(model_path)
    
    return client, reader, model

client, reader, model = load_models()

# ==========================================
# 2. TAMPILAN ANTARMUKA (UI)
# ==========================================
st.set_page_config(page_title="DocAI Analyzer", page_icon="📄", layout="wide")

# SUNTIKAN CSS UNTUK OPTIMASI TAMPILAN MOBILE
st.markdown("""
    <style>
        /* Pengaturan default untuk Desktop */
        .main-title {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        /* Pengaturan khusus jika dibuka di layar HP (lebar maksimal 768px) */
        @media (max-width: 768px) {
            .main-title {
                font-size: 1.5rem; /* Judul mengecil di HP agar tidak memakan layar */
                line-height: 1.2;
            }
            .stButton > button {
                width: 100%; /* Tombol Analisis menjadi full-width di HP agar mudah dipencet jempol */
            }
            .block-container {
                padding-top: 2rem; /* Mengurangi jarak kosong di bagian atas layar HP */
            }
        }
    </style>
""", unsafe_allow_html=True)

# Memanggil judul menggunakan HTML agar CSS di atas bisa bekerja
st.markdown('<div class="main-title">Deteksi Gambar Presisi dengan YOLOv26 dan LLaMA 3.3 untuk Ekstraksi dan Rangkuman Otomatis Isi Dokumen 📄</div>', unsafe_allow_html=True)
st.write("Unggah dokumen Anda, dan AI akan otomatis menganalisis serta merangkum isinya.")

uploaded_file = st.file_uploader("Unggah Dokumen (PDF, JPG, PNG)", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    if st.button("Mulai Analisis Dokumen", type="primary"):
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

            for idx, img in enumerate(images_to_process):
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                results = model(img_cv)
                boxes = results[0].boxes
                
                # Plotting hasil deteksi
                img_with_boxes = results[0].plot()
                annotated_images.append(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
                
                # OCR Extraction
                sorted_boxes = sorted(boxes, key=lambda b: b.xyxy[0][1].item())
                page_text = []
                for box in sorted_boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cropped = img_cv[y1:y2, x1:x2]
                    text_list = reader.readtext(cropped, detail=0)
                    if text_list:
                        page_text.append(" ".join(text_list))
                
                if len(images_to_process) > 1:
                    all_extracted_text += f"\n--- HALAMAN {idx + 1} ---\n"
                all_extracted_text += "\n".join(page_text) + "\n"

            # ---------------------------------------------------------
            # TAMPILAN GALLERY CARD (Kecil ke Samping)
            # ---------------------------------------------------------
            st.subheader("Visualisasi Deteksi Layout")
            st.caption("Klik gambar untuk memperbesar")
            
            # Tampilkan dalam 5 kolom (Thumbnails)
            num_cols = 5
            cols = st.columns(num_cols)
            for i, ann_img in enumerate(annotated_images):
                cols[i % num_cols].image(ann_img, use_container_width=True)
            
            st.divider()

            # ---------------------------------------------------------
            # HASIL TEKS & PENJELASAN AI
            # ---------------------------------------------------------
            if all_extracted_text.strip():
                # Area Teks yang Rapi untuk Disalin
                st.subheader("Hasil Ekstraksi Teks")
                st.code(all_extracted_text, language='text')
                
                st.divider()
                
                prompt = f"Tolong jelaskan secara ringkas isi dari dokumen berikut. Buat poin-poin utama agar mudah dipahami:\n\n{all_extracted_text}"
                
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.5,
                )
                
                st.subheader("Penjelasan AI berdasarkan Gambar atau Dokumen yang kamu kirim sebagai berikut:")
                st.write(chat_completion.choices[0].message.content)
            else:
                st.error("Gagal mengekstrak teks.")
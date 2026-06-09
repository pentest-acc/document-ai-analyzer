import os
# =========================================================================
# ANTI-CRASH PADDLEOCR WINDOWS & LINUX (HARUS BERADA DI PALING ATAS)
# =========================================================================
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

import io
import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from groq import Groq
from PIL import Image
import fitz  # PyMuPDF
from paddleocr import PaddleOCR

# ==========================================
# 1. SETUP MODEL & API (VERSI PRODUCTION GITHUB)
# ==========================================
@st.cache_resource
def load_models():
    # Mengambil API Key dari Secrets Streamlit Cloud agar aman
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    # show_log=False untuk meredam log berisik di server
    reader = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
    model_path = "./models/yolo26m_doc_layout.pt" 
    model = YOLO(model_path)
    return client, reader, model

client, reader, model = load_models()

# ==========================================
# 2. TAMPILAN ANTARMUKA (UI) PREMIUM GLASSMORPHISM
# ==========================================
st.set_page_config(page_title="DocAI Project", page_icon="⚡", layout="wide")

# SUNTIKAN CSS KHUSUS DENGAN STRUKTUR STRING YANG VALID
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght=300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }

        /* Latar Belakang Utama */
        .stApp {
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
            color: #ffffff;
        }

        /* Sidebar Glassmorphism */
        [data-testid="stSidebar"] {
            background-color: rgba(15, 32, 39, 0.6) !important;
            backdrop-filter: blur(15px);
            border-right: 1px solid rgba(255,255,255,0.1);
        }

        /* Styling Judul Utama */
        .premium-title {
            font-size: 3rem;
            font-weight: 700;
            background: -webkit-linear-gradient(45deg, #00f2fe, #4facfe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px;
            padding-bottom: 0px;
        }
        .subtitle {
            font-size: 1.1rem;
            color: #a8b2d1;
            margin-bottom: 2rem;
            font-weight: 300;
        }

        /* Styling Tombol Utama (Glow Effect) */
        div.stButton > button {
            background: linear-gradient(45deg, #00f2fe, #4facfe);
            color: white;
            border: none;
            border-radius: 50px;
            padding: 0.5rem 2rem;
            font-weight: 600;
            letter-spacing: 1px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
            width: 100%;
        }
        div.stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(79, 172, 254, 0.6);
            color: white;
            border-color: transparent;
        }

        /* Kotak Unggah File (Dropzone) */
        [data-testid="stFileUploadDropzone"] {
            background: rgba(255, 255, 255, 0.03);
            border: 2px dashed rgba(79, 172, 254, 0.5);
            border-radius: 20px;
            padding: 2rem;
            transition: all 0.3s ease;
        }
        [data-testid="stFileUploadDropzone"]:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: #00f2fe;
        }

        h1, h2, h3, h4, h5, h6, p, span, label {
            color: #ffffff;
        }

        /* Styling Grid Card Foto Terisolasi */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 15px !important;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            padding: 10px;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0, 242, 254, 0.15);
            border-color: rgba(0, 242, 254, 0.4) !important;
        }
        
        [data-testid="stVerticalBlockBorderWrapper"] img {
            max-height: 180px !important; 
            object-fit: contain !important; 
            border-radius: 8px;
            margin-bottom: 10px;
        }

        /* Membatasi tinggi visualisasi utama */
        [data-testid="stImage"] img {
            max-height: 400px; 
            object-fit: contain;
            border-radius: 10px;
        }

        @media (max-width: 768px) {
            .block-container { padding-top: 4rem !important; }
            .premium-title { font-size: 2rem; line-height: 1.2; }
            [data-testid="stVerticalBlockBorderWrapper"] img {
                max-height: 150px !important;
            }
        }
    </style>
""", unsafe_allow_html=True)

# INISIALISASI DATABASE CACHE RIWAYAT SESI
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = {}

st.sidebar.header("⚙️ Pengaturan Deteksi")
conf_threshold = st.sidebar.slider("Sensitivitas Deteksi (Threshold)", 0.0, 1.0, 0.25, 0.05)

# PANEL RIWAYAT MEMORI DI SIDEBAR
st.sidebar.markdown("---")
st.sidebar.subheader("📜 Memori Riwayat Sesi")
if st.session_state.analysis_history:
    for cache_key in st.session_state.analysis_history.keys():
        name_part, thresh_part = cache_key.split("||")
        st.sidebar.caption(f"🟢 {name_part} *(Thresh: {thresh_part})*")
else:
    st.sidebar.caption("Belum ada riwayat analisis ter-cache.")

st.markdown('<div class="premium-title">DocAI Project</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sistem ekstraksi dokumen otonom dengan integrasi LLaMA 3.3 (Groq API).</div>', unsafe_allow_html=True)

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

    st.write("")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("MULAI ANALISIS DOKUMEN", type="primary"):
            st.session_state.analysis_started = True

    if st.session_state.analysis_started:
        st.divider()
        
        unique_cache_key = f"{uploaded_file.name}||{conf_threshold}"
        
        if "analysis_results" not in st.session_state:
            
            if unique_cache_key in st.session_state.analysis_history:
                st.toast("⚡ Mengambil hasil dari memori riwayat (Instant Cache)!", icon="ℹ️")
                st.session_state.analysis_results = st.session_state.analysis_history[unique_cache_key]
            
            else:
                with st.spinner('Sedang memproses dokumen dengan YOLO dan PaddleOCR (Mohon tunggu)...'):
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
                        boxes = results[0].boxes
                        
                        img_with_boxes = results[0].plot()
                        annotated_images.append(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
                        
                        sorted_boxes = sorted(boxes, key=lambda b: b.xyxy[0][1].item())
                        page_text = []
                        
                        yolo_found_text = False
                        
                        for box in sorted_boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls_id = int(box.cls[0])
                            cls_name = model.names[cls_id]
                            cropped = img_cv[y1:y2, x1:x2]
                            
                            if cls_name == "Picture":
                                pic_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                                pil_img = Image.fromarray(pic_rgb)
                                extracted_pictures.append(pil_img)
                            else:
                                yolo_found_text = True
                                ocr_result = reader.ocr(cropped)
                                if ocr_result and ocr_result[0]:
                                    for line in ocr_result[0]:
                                        page_text.append(line[1][0])
                        
                        if not yolo_found_text:
                            ocr_result = reader.ocr(img_cv)
                            if ocr_result and ocr_result[0]:
                                for line in ocr_result[0]:
                                    page_text.append(line[1][0])
                        
                        if len(images_to_process) > 1:
                            all_extracted_text += f"\n--- HALAMAN {idx + 1} ---\n"
                        all_extracted_text += "\n".join(page_text) + "\n"

                    st.session_state.analysis_results = {
                        "annotated_images": annotated_images,
                        "extracted_pictures": extracted_pictures,
                        "all_extracted_text": all_extracted_text,
                        "corrected_text": None,
                        "summary": None,
                        "api_error": False
                    }

                    if all_extracted_text.strip():
                        try:
                            with st.spinner("AI LLaMA 3.3 sedang mengoreksi dan merangkum dokumen..."):
                                combined_prompt = f"""Tugas Anda ada dua: 
1. Koreksi teks OCR berikut (hapus duplikasi, perbaiki ejaan, JANGAN terjemahkan bahasa asing, pertahankan format halaman).
2. Buat ringkasan poin utama dari teks tersebut.

Teks Mentah OCR:
{all_extracted_text}

Anda WAJIB memberikan output dengan format persis seperti ini:
### TEKS KOREKSI ###
[Masukkan hasil teks yang sudah dikoreksi di sini]

### RANGKUMAN ###
[Masukkan hasil rangkuman di sini]"""
                                
                                completion = client.chat.completions.create(
                                    messages=[{"role": "user", "content": combined_prompt}],
                                    model="llama-3.3-70b-versatile",
                                    temperature=0.1, 
                                )
                                
                                full_response = completion.choices[0].message.content
                                
                                if "### RANGKUMAN ###" in full_response:
                                    parts = full_response.split("### RANGKUMAN ###")
                                    corrected = parts[0].replace("### TEKS KOREKSI ###", "").strip()
                                    summary = parts[1].strip()
                                else:
                                    corrected = full_response
                                    summary = "Gagal memisahkan rangkuman. Coba lagi."

                                st.session_state.analysis_results["corrected_text"] = corrected
                                st.session_state.analysis_results["summary"] = summary
                                
                        except Exception as e:
                            st.session_state.analysis_results["api_error"] = True
                
                st.session_state.analysis_history[unique_cache_key] = st.session_state.analysis_results

        res = st.session_state.analysis_results

        doc_name = st.session_state.current_file
        conf_level_pct = int(conf_threshold * 100)
        
        st.subheader(f"🔍 Hasil Deteksi Layout Dokumen **{doc_name}** dengan Confidence Level **{conf_level_pct}%** Adalah:")
        ann_images = res["annotated_images"]
        for i in range(0, len(ann_images), 4):
            cols = st.columns(4)
            for j in range(4):
                if i + j < len(ann_images):
                    with cols[j]:
                        st.image(ann_images[i+j], width='stretch')
        st.divider()

        if res["extracted_pictures"]:
            st.subheader(f"🖼️ Gambar yang Ditemukan di Dokumen **{doc_name}**:")
            pics = res["extracted_pictures"]
            
            for i in range(0, len(pics), 4):
                cols = st.columns(4)
                for j in range(4):
                    if i + j < len(pics):
                        pic = pics[i + j]
                        with cols[j]:
                            with st.container(border=True):
                                st.image(pic, width='stretch')
                                
                                buf = io.BytesIO()
                                pic.save(buf, format="PNG")
                                byte_im = buf.getvalue()
                                
                                st.download_button(
                                    label="📥 Unduh Aset", 
                                    data=byte_im, 
                                    file_name=f"ekstraksi_gambar_{i+j+1}.png", 
                                    mime="image/png", 
                                    key=f"dl_btn_{i+j}",
                                    width='stretch'
                                )
            st.divider()

        if res["all_extracted_text"].strip():
            st.subheader(f"Hasil Ekstraksi Teks Dokumen **{doc_name}** (Perbandingan)")
            
            if res.get("api_error"):
                st.warning("⚠️ Proses koreksi AI terhenti (kemungkinan limit API/Jaringan). Menampilkan teks mentah:")
                st.code(res["all_extracted_text"], language='text')
            elif res["corrected_text"]:
                tab1, tab2 = st.tabs(["📝 Teks Mentah (Raw OCR)", "✨ Teks Diperbaiki (AI Corrected)"])
                with tab1:
                    st.caption("Ini adalah hasil pembacaan asli dari PaddleOCR sebelum diproses AI.")
                    st.code(res["all_extracted_text"], language='text')
                with tab2:
                    st.caption("Ini adalah teks yang telah dikoreksi tata bahasa dan spasinya oleh LLaMA 3.3.")
                    st.code(res["corrected_text"], language='text')

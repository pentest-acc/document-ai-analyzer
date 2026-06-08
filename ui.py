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
        .main-title {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        @media (max-width: 768px) {
            .block-container {
                padding-top: 5rem !important; 
            }
            .main-title {
                font-size: 1.3rem; 
                line-height: 1.2;
                margin-top: 10px;
            }
            .stButton > button {
                width: 100%;
            }
        }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.header("⚙️ Pengaturan Deteksi")
conf_threshold = st.sidebar.slider(
    "Sensitivitas Deteksi (Threshold)", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.25, 
    step=0.05
)

st.markdown('<div class="main-title">Deteksi Struktur, Buat Ekstraksi dan Ringkasan Document Berbentuk Gambar Atau PDF Dengan YOLOv26 dan Bantuan LLaMA</div>', unsafe_allow_html=True)
st.write("Unggah dokumen Anda, dan AI akan otomatis menganalisis serta merangkum isinya.")

# Inisialisasi Memori (Session State)
if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "current_threshold" not in st.session_state:
    st.session_state.current_threshold = conf_threshold

uploaded_file = st.file_uploader("Unggah Dokumen (PDF, JPG, PNG)", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    if st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.analysis_started = False
        if "analysis_results" in st.session_state:
            del st.session_state["analysis_results"]

    if st.session_state.current_threshold != conf_threshold:
        st.session_state.current_threshold = conf_threshold
        if "analysis_results" in st.session_state:
            del st.session_state["analysis_results"]

    if st.button("Mulai Analisis Dokumen", type="primary"):
        st.session_state.analysis_started = True

    if st.session_state.analysis_started:
        
        if "analysis_results" not in st.session_state:
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
                extracted_pictures = []

                for idx, img in enumerate(images_to_process):
                    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    
                    results = model(img_cv, conf=conf_threshold)
                    boxes = results[0].boxes
                    
                    img_with_boxes = results[0].plot()
                    annotated_images.append(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
                    
                    sorted_boxes = sorted(boxes, key=lambda b: b.xyxy[0][1].item())
                    page_text = []
                    
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
                            text_list = reader.readtext(cropped, detail=0)
                            if text_list:
                                page_text.append(" ".join(text_list))
                    
                    if len(images_to_process) > 1:
                        all_extracted_text += f"\n--- HALAMAN {idx + 1} ---\n"
                    all_extracted_text += "\n".join(page_text) + "\n"

                st.session_state.analysis_results = {
                    "annotated_images": annotated_images,
                    "extracted_pictures": extracted_pictures,
                    "all_extracted_text": all_extracted_text,
                    "corrected_text": None,
                    "summary": None
                }

                if all_extracted_text.strip():
                    with st.spinner("Memproses teks dengan AI..."):
                        # Prompt diperketat: Hapus teks ganda, tapi JANGAN terjemahkan bahasa asing
                        proofread_prompt = f"""Anda adalah sistem pemroses teks otomatis. Tugas Anda HANYA memperbaiki ejaan, tata bahasa, salah ketik (typo), dan spasi dari teks mentah OCR di bawah ini.

ATURAN MUTLAK:
1. JANGAN tambahkan kalimat basa-basi pengantar atau penutup. Langsung berikan hasil teksnya saja.
2. PERTAHANKAN BAHASA ASLI SECARA KETAT. Jika kalimat dalam Bahasa Asing (seperti Inggris), perbaiki grammar/ejaannya dalam bahasa tersebut. JANGAN diterjemahkan ke Bahasa Indonesia. Jika ada campuran bahasa, perbaiki masing-masing sesuai bahasa aslinya.
3. HAPUS TEKS GANDA. Jika ada kata atau kalimat yang terulang/ganda akibat tumpang tindih pembacaan kotak OCR, hapus duplikasinya agar kalimat menjadi padu, logis, dan tidak berulang.
4. Jangan mengubah format penanda halaman (seperti --- HALAMAN 1 ---).

Teks Mentah OCR:
{all_extracted_text}"""
                        proofread_completion = client.chat.completions.create(
                            messages=[{"role": "user", "content": proofread_prompt}],
                            model="llama-3.3-70b-versatile",
                            temperature=0.1, 
                        )
                        st.session_state.analysis_results["corrected_text"] = proofread_completion.choices[0].message.content

                    with st.spinner("Membuat rangkuman dokumen..."):
                        summary_prompt = f"Tolong jelaskan secara ringkas isi dari dokumen berikut. Buat poin-poin utama agar mudah dipahami:\n\n{st.session_state.analysis_results['corrected_text']}"
                        summary_completion = client.chat.completions.create(
                            messages=[{"role": "user", "content": summary_prompt}],
                            model="llama-3.3-70b-versatile",
                            temperature=0.5,
                        )
                        st.session_state.analysis_results["summary"] = summary_completion.choices[0].message.content

        # =========================================================
        # RENDER TAMPILAN BERDASARKAN MEMORI (SESSION STATE)
        # =========================================================
        res = st.session_state.analysis_results

        st.subheader("Visualisasi Deteksi Layout")
        num_cols = 5
        cols = st.columns(num_cols)
        for i, ann_img in enumerate(res["annotated_images"]):
            cols[i % num_cols].image(ann_img, use_container_width=True)
        st.divider()

        if res["extracted_pictures"]:
            st.subheader("🖼️ Gambar yang Ditemukan di Dokumen")
            pic_cols = st.columns(min(len(res["extracted_pictures"]), 4))
            for i, pic in enumerate(res["extracted_pictures"]):
                with pic_cols[i % 4]:
                    st.image(pic, use_container_width=True)
                    
                    buf = io.BytesIO()
                    pic.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="Unduh Gambar",
                        data=byte_im,
                        file_name=f"ekstraksi_gambar_{i+1}.png",
                        mime="image/png",
                        key=f"download_btn_{i}" 
                    )
            st.divider()

        if res["all_extracted_text"].strip() and res["corrected_text"]:
            st.subheader("Hasil Ekstraksi Teks (Perbandingan)")
            
            # --- TAMPILAN TAB UNTUK VERSI MENTAH VS PERBAIKAN ---
            tab1, tab2 = st.tabs(["📝 Teks Mentah (Raw OCR)", "✨ Teks Diperbaiki (AI Corrected)"])
            
            with tab1:
                st.caption("Ini adalah hasil pembacaan asli dari EasyOCR sebelum diproses AI (Berguna untuk analisis akurasi Model YOLO & OCR).")
                st.code(res["all_extracted_text"], language='text')
                
            with tab2:
                st.caption("Ini adalah teks yang telah dikoreksi tata bahasa dan spasinya, serta dihapus duplikasi teksnya oleh LLaMA 3.3 tanpa mengubah bahasa aslinya.")
                st.code(res["corrected_text"], language='text')
                
            st.divider()
            
            st.subheader("Penjelasan AI berdasarkan Dokumen:")
            st.write(res["summary"])
        else:
            st.error("Gagal mengekstrak teks. Pastikan dokumen Anda memiliki tulisan yang bisa dibaca.")

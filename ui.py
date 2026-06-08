import os
import io # Tambahan untuk fitur unduh gambar
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

st.markdown('<div class="main-title">Deteksi Struktur Document Dengan YOLOv26 dan Menggunakan LLaMA 3.3 Untuk Ekstraksi dan Rangkuman Isi Dokumen</div>', unsafe_allow_html=True)
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
            extracted_pictures = [] # Tempat menyimpan gambar/logo yang dipotong

            for idx, img in enumerate(images_to_process):
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                results = model(img_cv)
                boxes = results[0].boxes
                
                # Plotting hasil deteksi
                img_with_boxes = results[0].plot()
                annotated_images.append(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
                
                # OCR dan Image Extraction
                sorted_boxes = sorted(boxes, key=lambda b: b.xyxy[0][1].item())
                page_text = []
                
                for box in sorted_boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id] # Membaca nama label (misal: "Picture", "Title", "Text")
                    
                    cropped = img_cv[y1:y2, x1:x2]
                    
                    # Jika AI mengenali kotak ini sebagai Gambar/Picture
                    if cls_name == "Picture":
                        pic_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(pic_rgb)
                        extracted_pictures.append(pil_img)
                    
                    # Jika kotak adalah teks
                    else:
                        text_list = reader.readtext(cropped, detail=0)
                        if text_list:
                            page_text.append(" ".join(text_list))
                
                if len(images_to_process) > 1:
                    all_extracted_text += f"\n--- HALAMAN {idx + 1} ---\n"
                all_extracted_text += "\n".join(page_text) + "\n"

            # ---------------------------------------------------------
            # TAMPILAN GALLERY CARD
            # ---------------------------------------------------------
            st.subheader("Visualisasi Deteksi Layout")
            num_cols = 5
            cols = st.columns(num_cols)
            for i, ann_img in enumerate(annotated_images):
                cols[i % num_cols].image(ann_img, use_container_width=True)
            
            st.divider()

            # ---------------------------------------------------------
            # FITUR BARU: TAMPILKAN DAN UNDUH GAMBAR
            # ---------------------------------------------------------
            if extracted_pictures:
                st.subheader("🖼️ Gambar yang Ditemukan di Dokumen")
                pic_cols = st.columns(min(len(extracted_pictures), 4))
                for i, pic in enumerate(extracted_pictures):
                    with pic_cols[i % 4]:
                        st.image(pic, use_container_width=True)
                        
                        # Mengubah gambar ke format bytes untuk tombol unduh
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

            # ---------------------------------------------------------
            # HASIL TEKS & PENJELASAN AI
            # ---------------------------------------------------------
            if all_extracted_text.strip():
                # FITUR BARU: AI POST-PROCESSING (Proofreading)
                st.subheader("Hasil Ekstraksi Teks")
                with st.spinner("Memperbaiki dan merapikan susunan teks dengan AI..."):
                    proofread_prompt = f"Anda adalah asisten editor. Perbaiki ejaan, salah ketik, dan susunan kalimat dari hasil ekstraksi teks OCR berikut agar mudah dibaca, rapi, dan logis. JANGAN mengubah makna, menambah, atau mengurangi informasi aslinya. Jika teksnya sudah bagus, biarkan saja:\n\n{all_extracted_text}"
                    
                    proofread_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": proofread_prompt}],
                        model="llama-3.3-70b-versatile",
                        temperature=0.1, # Temperature rendah agar AI tidak mengarang bebas
                    )
                    corrected_text = proofread_completion.choices[0].message.content

                # Tampilkan teks yang sudah dikoreksi
                st.code(corrected_text, language='text')
                
                st.divider()
                
                # Rangkuman Dokumen
                st.subheader("Penjelasan AI berdasarkan Dokumen:")
                with st.spinner("Membuat rangkuman dokumen..."):
                    summary_prompt = f"Tolong jelaskan secara ringkas isi dari dokumen berikut. Buat poin-poin utama agar mudah dipahami:\n\n{corrected_text}"
                    
                    summary_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": summary_prompt}],
                        model="llama-3.3-70b-versatile",
                        temperature=0.5,
                    )
                    st.write(summary_completion.choices[0].message.content)
            else:
                st.error("Gagal mengekstrak teks. Pastikan dokumen Anda memiliki tulisan yang bisa dibaca.")

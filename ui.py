import os
# =========================================================================
# ANTI-CRASH PADDLEOCR WINDOWS (HARUS BERADA DI PALING ATAS)
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
# 1. SETUP MODEL & API (VERSI GITHUB SECURE)
# ==========================================
@st.cache_resource
def load_models():
    # Mengambil API Key dari brankas rahasia Streamlit, BUKAN dari teks mentah!
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    reader = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
    model_path = "yolo26m_doc_layout.pt" 
    model = YOLO(model_path)
    return client, reader, model

client, reader, model = load_models()

# ==========================================
# 2. TAMPILAN ANTARMUKA (UI) PREMIUM
# ==========================================
st.set_page_config(page_title="DocAI Project", page_icon="⚡", layout="wide")

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
if "analysis_

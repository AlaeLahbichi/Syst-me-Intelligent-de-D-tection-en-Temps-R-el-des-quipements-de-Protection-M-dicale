import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import time

try:
    import torch
except Exception:
    torch = None

try:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="MedGuard — Détection EPI Médicaux",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CSS — DARK MEDICAL THEME v2
# =========================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    .stApp {
        background: #03080f;
        color: #c8dff0;
        font-family: 'Inter', sans-serif;
    }

    /* ---- HIDE STREAMLIT CHROME ---- */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 0 2rem 3rem !important; max-width: 1400px; margin: 0 auto; }

    /* ---- SIDEBAR ---- */
    section[data-testid="stSidebar"] {
        background: #050c17 !important;
        border-right: 1px solid rgba(0,230,160,0.08);
        width: 260px !important;
    }
    section[data-testid="stSidebar"] > div { padding: 0 !important; }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label { color: #6a8aa8 !important; }

    .sb-logo {
        padding: 28px 24px 20px;
        border-bottom: 1px solid rgba(0,230,160,0.08);
        margin-bottom: 8px;
    }
    .sb-logo-badge {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        text-decoration: none;
    }
    .sb-icon {
        width: 38px; height: 38px;
        background: linear-gradient(135deg, #00e6a0 0%, #0060d4 100%);
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 18px;
        box-shadow: 0 0 24px rgba(0,230,160,0.25);
        flex-shrink: 0;
    }
    .sb-name { font-weight: 800; font-size: 18px; color: #fff; letter-spacing: -0.5px; }
    .sb-version {
        font-size: 10px; color: #00e6a0; letter-spacing: 1.5px;
        text-transform: uppercase; font-weight: 600;
        background: rgba(0,230,160,0.08);
        padding: 2px 8px; border-radius: 4px;
        margin-top: 4px; display: inline-block;
    }

    .sb-section {
        padding: 20px 24px 8px;
        font-size: 9px; font-weight: 700; letter-spacing: 2px;
        text-transform: uppercase; color: #2a4a62;
    }

    .sb-nav { padding: 0 12px 8px; }

    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        border: none;
        padding: 10px 16px;
        background: transparent;
        color: #6a8aa8;
        font-weight: 500;
        font-size: 13.5px;
        text-align: left;
        font-family: 'Inter', sans-serif;
        transition: all 0.15s;
        margin-bottom: 2px;
        cursor: pointer;
    }
    div.stButton > button:hover {
        background: rgba(0,230,160,0.07);
        color: #00e6a0;
    }

    .sb-divider {
        margin: 12px 24px;
        height: 1px;
        background: rgba(255,255,255,0.05);
    }

    .sb-footer {
        padding: 16px 24px;
        font-size: 11px;
        color: #1e3a50;
        line-height: 1.8;
    }

    /* ---- TOPBAR ---- */
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 20px 0 20px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 32px;
    }
    .topbar-left {
        display: flex; align-items: center; gap: 14px;
    }
    .topbar-icon {
        width: 46px; height: 46px;
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 22px;
        background: rgba(0,230,160,0.1);
        border: 1px solid rgba(0,230,160,0.2);
    }
    .topbar-icon.blue { background: rgba(0,96,212,0.12); border-color: rgba(0,96,212,0.25); }
    .topbar-icon.purple { background: rgba(120,60,220,0.12); border-color: rgba(120,60,220,0.25); }
    .topbar h1 {
        font-size: 24px !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        letter-spacing: -0.5px;
        line-height: 1.1 !important;
    }
    .topbar p {
        font-size: 13px;
        color: #3a5a70;
        margin-top: 3px;
    }
    .topbar-right { display: flex; align-items: center; gap: 8px; }

    .badge {
        display: inline-flex; align-items: center; gap: 5px;
        padding: 5px 12px; border-radius: 20px;
        font-size: 12px; font-weight: 600;
        white-space: nowrap;
    }
    .badge-green {
        background: rgba(0,230,160,0.1);
        border: 1px solid rgba(0,230,160,0.25);
        color: #00e6a0;
    }
    .badge-gray {
        background: rgba(100,130,160,0.1);
        border: 1px solid rgba(100,130,160,0.2);
        color: #7a9ab8;
    }
    .badge-amber {
        background: rgba(245,158,11,0.1);
        border: 1px solid rgba(245,158,11,0.25);
        color: #fbbf24;
    }

    /* ---- HERO ---- */
    .hero {
        position: relative;
        background: #050c17;
        border: 1px solid rgba(0,230,160,0.1);
        border-radius: 20px;
        padding: 48px 52px;
        overflow: hidden;
        margin-bottom: 28px;
    }
    .hero-glow-tl {
        position: absolute; top: -120px; left: -80px;
        width: 400px; height: 400px; border-radius: 50%;
        background: radial-gradient(circle, rgba(0,230,160,0.08) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero-glow-br {
        position: absolute; bottom: -100px; right: -60px;
        width: 350px; height: 350px; border-radius: 50%;
        background: radial-gradient(circle, rgba(0,96,212,0.1) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero-grid-lines {
        position: absolute; inset: 0;
        background-image:
            linear-gradient(rgba(0,230,160,0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,230,160,0.025) 1px, transparent 1px);
        background-size: 40px 40px;
        pointer-events: none;
    }
    .hero-content { position: relative; z-index: 1; display: flex; align-items: center; gap: 48px; }
    .hero-text { flex: 1; }
    .hero-eyebrow {
        display: inline-flex; align-items: center; gap: 8px;
        font-size: 11px; font-weight: 700; letter-spacing: 2px;
        text-transform: uppercase; color: #00e6a0;
        background: rgba(0,230,160,0.07); border: 1px solid rgba(0,230,160,0.2);
        padding: 5px 14px; border-radius: 20px; margin-bottom: 20px;
    }
    .hero-h1 {
        font-size: 46px; font-weight: 900; color: #fff;
        letter-spacing: -1.5px; line-height: 1.05; margin-bottom: 16px;
    }
    .hero-h1 span {
        background: linear-gradient(90deg, #00e6a0 0%, #0080ff 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-sub {
        font-size: 15px; color: #4a6a84; line-height: 1.75;
        max-width: 560px; margin-bottom: 28px;
    }
    .hero-tags { display: flex; flex-wrap: wrap; gap: 8px; }
    .hero-tag {
        padding: 6px 14px; border-radius: 8px;
        background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
        font-size: 12.5px; color: #7a9ab8; font-weight: 500;
    }

    /* SCANNER ANIMATION */
    .scan-wrap {
        width: 180px; height: 180px;
        position: relative; flex-shrink: 0;
    }
    .scan-ring {
        position: absolute; inset: 0;
        border-radius: 50%;
        border: 1.5px solid rgba(0,230,160,0.2);
        animation: scanPulse 3s ease-in-out infinite;
    }
    .scan-ring:nth-child(2) { inset: 22px; border-color: rgba(0,230,160,0.3); animation-delay: 0.75s; }
    .scan-ring:nth-child(3) { inset: 44px; border-color: rgba(0,230,160,0.45); animation-delay: 1.5s; }
    .scan-core {
        position: absolute; inset: 66px;
        border-radius: 50%;
        background: linear-gradient(135deg, rgba(0,230,160,0.15), rgba(0,96,212,0.15));
        border: 1.5px solid rgba(0,230,160,0.6);
        display: flex; align-items: center; justify-content: center;
        font-size: 28px;
        animation: coreGlow 3s ease-in-out infinite;
    }
    .scan-line {
        position: absolute;
        top: 10px; left: 50%; width: 2px;
        height: 160px;
        transform-origin: center 80px;
        animation: scanRotate 3s linear infinite;
        background: linear-gradient(to bottom, transparent, rgba(0,230,160,0.8), transparent);
    }
    @keyframes scanPulse { 0%,100%{transform:scale(1);opacity:.3} 50%{transform:scale(1.07);opacity:1} }
    @keyframes coreGlow { 0%,100%{box-shadow:0 0 12px rgba(0,230,160,.25)} 50%{box-shadow:0 0 32px rgba(0,230,160,.65)} }
    @keyframes scanRotate { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }

    /* ---- STATS BAR ---- */
    .stats-bar {
        display: grid; grid-template-columns: repeat(4,1fr);
        gap: 12px; margin-bottom: 28px;
    }
    .stat-card {
        padding: 20px 22px;
        border-radius: 14px;
        background: #050c17;
        border: 1px solid rgba(255,255,255,0.06);
        position: relative; overflow: hidden;
    }
    .stat-card::after {
        content: '';
        position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
    }
    .stat-card.g::after { background: linear-gradient(90deg,#00e6a0,transparent); }
    .stat-card.b::after { background: linear-gradient(90deg,#0070f3,transparent); }
    .stat-card.a::after { background: linear-gradient(90deg,#f59e0b,transparent); }
    .stat-card.r::after { background: linear-gradient(90deg,#ef4444,transparent); }
    .stat-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 38px; font-weight: 700; line-height: 1;
        margin-bottom: 6px;
    }
    .stat-num.g{color:#00e6a0} .stat-num.b{color:#3b8ff7} .stat-num.a{color:#fbbf24} .stat-num.r{color:#f87171}
    .stat-label { font-size: 12px; color: #3a5a70; font-weight: 500; }

    /* ---- SECTION TITLE ---- */
    .sec-title {
        font-size: 13px; font-weight: 700; letter-spacing: 1.5px;
        text-transform: uppercase; color: #2a4a60;
        display: flex; align-items: center; gap: 12px;
        margin: 28px 0 16px;
    }
    .sec-title::after { content:''; flex:1; height:1px; background:rgba(255,255,255,0.05); }
    .sec-title-accent { color: #00e6a0; }

    /* ---- LEVEL CARDS ---- */
    .level-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:28px; }
    .level-card {
        padding: 24px;
        border-radius: 14px;
        background: #050c17;
        border: 1px solid rgba(255,255,255,0.06);
        position: relative; overflow: hidden;
    }
    .level-card::before {
        content: ''; position: absolute; top:0; left:0; right:0; height:3px;
    }
    .level-card.ok::before { background: linear-gradient(90deg,#00e6a0,#00c070); }
    .level-card.warn::before { background: linear-gradient(90deg,#f59e0b,#f77f00); }
    .level-card.danger::before { background: linear-gradient(90deg,#ef4444,#c00); }
    .level-icon {
        font-size: 28px; margin-bottom: 14px;
        display: inline-block;
        filter: drop-shadow(0 0 10px rgba(0,230,160,0.4));
    }
    .level-card.warn .level-icon { filter: drop-shadow(0 0 10px rgba(245,158,11,0.4)); }
    .level-card.danger .level-icon { filter: drop-shadow(0 0 10px rgba(239,68,68,0.4)); }
    .level-title { font-size: 15px; font-weight: 700; margin-bottom: 8px; }
    .level-card.ok .level-title { color: #00e6a0; }
    .level-card.warn .level-title { color: #fbbf24; }
    .level-card.danger .level-title { color: #f87171; }
    .level-desc { font-size: 13px; color: #3a5a70; line-height: 1.65; }

    /* CLASS TAGS */
    .ctag-row { display:flex; flex-wrap:wrap; gap:7px; margin-top:10px; }
    .ctag {
        display:inline-flex; align-items:center; gap:5px;
        padding:5px 12px; border-radius:8px;
        font-size:12.5px; font-weight:600;
    }
    .ctag.ok { background:rgba(0,230,160,0.08); border:1px solid rgba(0,230,160,0.2); color:#00e6a0; }
    .ctag.no { background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.2); color:#f87171; }
    .ctag.warn { background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.2); color:#fbbf24; }
    .ctag.neutral { background:rgba(100,130,160,0.08); border:1px solid rgba(100,130,160,0.15); color:#7a9ab8; }

    /* ---- UPLOAD ZONE ---- */
    .stFileUploader {
        background: rgba(0,230,160,0.02) !important;
        border: 1.5px dashed rgba(0,230,160,0.2) !important;
        border-radius: 14px !important;
        transition: border-color 0.2s;
    }
    .stFileUploader:hover {
        border-color: rgba(0,230,160,0.4) !important;
    }
    .upload-hint {
        padding: 14px 18px;
        border-radius: 10px;
        background: rgba(0,96,212,0.05);
        border: 1px solid rgba(0,96,212,0.15);
        color: #4a6e96;
        font-size: 13px;
        margin-top: 10px;
        line-height: 1.6;
    }

    /* ---- IMAGE PANEL ---- */
    .img-panel {
        padding: 16px;
        border-radius: 14px;
        background: #050c17;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .img-panel-label {
        font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
        text-transform: uppercase; color: #2a4a60;
        margin-bottom: 10px; padding-bottom: 8px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }

    /* ---- DECISION BOX ---- */
    .dec-box {
        padding: 28px 32px;
        border-radius: 16px;
        margin: 20px 0;
        position: relative; overflow: hidden;
    }
    .dec-box::before {
        content:''; position:absolute; left:0;top:0;bottom:0; width:4px; border-radius:4px 0 0 4px;
    }
    .dec-box.ok { background:rgba(0,230,160,0.04); border:1px solid rgba(0,230,160,0.2); }
    .dec-box.ok::before { background:#00e6a0; }
    .dec-box.warn { background:rgba(245,158,11,0.04); border:1px solid rgba(245,158,11,0.25); }
    .dec-box.warn::before { background:#f59e0b; }
    .dec-box.danger { background:rgba(239,68,68,0.04); border:1px solid rgba(239,68,68,0.25); }
    .dec-box.danger::before { background:#ef4444; }
    .dec-label {
        font-size: 11px; font-weight: 700; letter-spacing: 1.5px;
        text-transform: uppercase; margin-bottom: 6px;
    }
    .dec-box.ok .dec-label { color:#00e6a0; }
    .dec-box.warn .dec-label { color:#fbbf24; }
    .dec-box.danger .dec-label { color:#f87171; }
    .dec-title { font-size: 26px; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 6px; }
    .dec-box.ok .dec-title { color:#fff; }
    .dec-box.warn .dec-title { color:#fff; }
    .dec-box.danger .dec-title { color:#fff; }
    .dec-msg { font-size: 14px; color: #3a5a70; margin-bottom: 20px; }
    .score-row { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:8px; }
    .score-label { font-size: 12px; color: #2a4a60; }
    .score-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 20px; font-weight: 700;
    }
    .dec-box.ok .score-val { color:#00e6a0; }
    .dec-box.warn .score-val { color:#fbbf24; }
    .dec-box.danger .score-val { color:#f87171; }
    .score-track { height:6px; border-radius:3px; background:rgba(255,255,255,0.06); }
    .score-fill { height:100%; border-radius:3px; transition:width .5s; }
    .score-fill.ok { background:linear-gradient(90deg,#00e6a0,#00c070); }
    .score-fill.warn { background:linear-gradient(90deg,#f59e0b,#f77f00); }
    .score-fill.danger { background:linear-gradient(90deg,#ef4444,#c00); }

    /* RESULT PANELS */
    .rp {
        padding: 18px 20px;
        border-radius: 12px;
        background: #050c17;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .rp-title {
        font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
        text-transform: uppercase; padding-bottom: 10px;
        border-bottom: 1px solid rgba(255,255,255,0.05); margin-bottom: 10px;
    }
    .rp.ok .rp-title { color:#00e6a0; }
    .rp.miss .rp-title { color:#fbbf24; }
    .rp.bad .rp-title { color:#f87171; }
    .rp-item {
        display:flex; align-items:center; gap:8px;
        font-size:13.5px; color:#6a8aa8;
        padding: 7px 0;
        border-bottom: 1px solid rgba(255,255,255,0.03);
    }
    .rp-item:last-child { border-bottom: none; }
    .dot { width:6px; height:6px; border-radius:50%; flex-shrink:0; }
    .dot.ok{background:#00e6a0} .dot.miss{background:#fbbf24} .dot.bad{background:#f87171}

    /* DET TABLE */
    .det-table {
        width:100%; border-collapse:collapse; font-size:13px; margin-top:10px;
    }
    .det-table th {
        text-align:left; padding:9px 14px;
        font-size:10px; font-weight:700; letter-spacing:1.5px;
        text-transform:uppercase; color:#2a4a60;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .det-table td {
        padding:9px 14px; border-bottom:1px solid rgba(255,255,255,0.03); color:#6a8aa8;
    }
    .det-table tr:last-child td { border-bottom:none; }
    .det-table tr:hover td { background:rgba(0,230,160,0.02); }
    .cbadge {
        display:inline-block; padding:3px 9px; border-radius:6px;
        font-family:'JetBrains Mono',monospace; font-size:11.5px; font-weight:600;
    }
    .ch { background:rgba(0,230,160,0.08); color:#00e6a0; }
    .cm { background:rgba(245,158,11,0.08); color:#fbbf24; }
    .cl { background:rgba(239,68,68,0.08); color:#f87171; }

    /* INFO BOX */
    .info-box {
        padding: 18px 22px;
        border-radius: 12px;
        background: #050c17;
        border: 1px solid rgba(0,230,160,0.1);
        margin-top: 16px;
    }
    .info-box h4 { font-size:14px; font-weight:700; color:#00e6a0; margin-bottom:8px; }
    .info-box p { font-size:13px; color:#3a5a70; line-height:1.7; }

    /* SLIDERS */
    .stSlider [data-baseweb="slider"] { padding: 0 !important; }

    /* SPINNER */
    .stSpinner > div { border-top-color: #00e6a0 !important; }

    /* SELECTBOX */
    .stSelectbox [data-baseweb="select"] > div {
        background: #050c17 !important;
        border-color: rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        color: #c8dff0 !important;
    }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid rgba(255,255,255,0.06) !important;
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #3a5a70 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 12px 20px !important;
        border-radius: 0 !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.15s;
    }
    .stTabs [aria-selected="true"] {
        color: #00e6a0 !important;
        border-bottom-color: #00e6a0 !important;
    }
    .stTabs [data-baseweb="tab-panel"] { padding: 24px 0 0 !important; }

    h1,h2,h3,h4,h5 { color: #e8f4ff !important; font-family:'Inter',sans-serif !important; }
    .stMarkdown p { color: #6a8aa8; }
    .stAlert { border-radius:12px !important; }
    div[data-testid="stProgress"] > div { background:rgba(0,230,160,0.15) !important; }
    div[data-testid="stProgress"] > div > div { background:#00e6a0 !important; }

    /* VIDEO PROGRESS */
    .video-progress-wrap {
        padding: 20px 24px;
        background: #050c17;
        border: 1px solid rgba(0,230,160,0.1);
        border-radius: 12px;
        margin: 16px 0;
    }
    .video-frame-label {
        display:flex; justify-content:space-between; margin-bottom:10px;
        font-size:12px; color:#3a5a70; font-family:'JetBrains Mono',monospace;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# DEVICE
# =========================================================

def get_best_device():
    if torch is not None:
        try:
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return 0
        except Exception:
            pass
    return "cpu"

DEVICE = get_best_device()


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Erreur de chargement du modèle : {e}")
        return None

MODEL_PATH = "/Users/alaethelegend/Documents/Projet_PFA/runs/detect/Lab/weights/best.pt"   
model = load_model(MODEL_PATH)


# =========================================================
# CLASSES & RULES
# =========================================================

POSITIVE_CLASSES = {
    "mask": "Masque de protection",
    "gloves": "Gants",
    "head mask": "Charlotte / Couvre-chef",
    "lab coat": "Blouse médicale",
    "googles": "Lunettes de protection",
    "goggles": "Lunettes de protection"
}
NEGATIVE_CLASSES = {
    "no mask": "Absence de masque",
    "no gloves": "Absence de gants",
    "no head mask": "Absence de couvre-chef",
    "no lab coat": "Absence de blouse médicale",
    "no labcoat": "Absence de blouse médicale",
    "no googles": "Absence de lunettes",
    "no goggles": "Absence de lunettes"
}
DANGER_CLASSES = {
    "eating": "Comportement interdit — manger",
    "drinking": "Comportement interdit — boire"
}
IGNORED_CLASSES = {"emsi": "Logo EMSI"}
REQUIRED_EQUIPMENT = {
    "Masque de protection", "Gants", "Charlotte / Couvre-chef",
    "Blouse médicale", "Lunettes de protection"
}


# =========================================================
# UTILS
# =========================================================

def normalize_label(label):
    return label.strip().lower().replace("_", " ").replace("-", " ")


def predict_yolo(source, conf=0.25, imgsz=1280):
    if model is None:
        return None
    try:
        return model.predict(source=source, conf=conf, imgsz=imgsz, device=DEVICE, verbose=False)
    except Exception:
        return model.predict(source=source, conf=conf, imgsz=imgsz, device="cpu", verbose=False)


def extract_detections(result, mdl):
    detections, labels = [], []
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return detections, labels
    for box in result.boxes:
        class_id = int(box.cls[0])
        label = mdl.names[class_id]
        confidence = float(box.conf[0])
        labels.append(label)
        detections.append({"Classe": label, "Confiance": round(confidence, 2)})
    return detections, labels


def evaluate_protection(labels):
    detected_positive, detected_negative, detected_danger, ignored = set(), set(), set(), set()
    for label in labels:
        clean = normalize_label(label)
        if clean in POSITIVE_CLASSES:
            detected_positive.add(POSITIVE_CLASSES[clean])
        elif clean in NEGATIVE_CLASSES:
            detected_negative.add(NEGATIVE_CLASSES[clean])
        elif clean in DANGER_CLASSES:
            detected_danger.add(DANGER_CLASSES[clean])
        elif clean in IGNORED_CLASSES:
            ignored.add(IGNORED_CLASSES[clean])

    missing = REQUIRED_EQUIPMENT - detected_positive
    score = int((len(detected_positive) / len(REQUIRED_EQUIPMENT)) * 100)

    base = {
        "score": score,
        "positive": sorted(detected_positive),
        "negative": sorted(detected_negative),
        "missing": sorted(missing),
        "ignored": sorted(ignored),
    }

    if detected_danger:
        return {**base, "status": "danger",
                "title": "Accès Impossible",
                "emoji": "⛔",
                "message": "Comportement interdit détecté dans la zone médicale stérile.",
                "details": sorted(detected_danger)}
    if detected_negative:
        return {**base, "status": "danger",
                "title": "Accès Refusé",
                "emoji": "🚨",
                "message": "Conditions de protection sanitaire non respectées.",
                "details": sorted(detected_negative)}
    if len(missing) == 0:
        return {**base, "status": "ok", "score": 100,
                "title": "Accès Autorisé",
                "emoji": "✅",
                "message": "Tous les équipements de protection individuelle sont conformes.",
                "details": sorted(detected_positive)}
    return {**base, "status": "warn",
            "title": "Protection Incomplète",
            "emoji": "⚠️",
            "message": "Certains équipements obligatoires ne sont pas confirmés.",
            "details": sorted(detected_positive)}


def render_decision(decision):
    s = decision["status"]
    score = decision["score"]
    css_class = {"ok": "ok", "warn": "warn", "danger": "danger"}[s]
    score_color = {"ok": "#00e6a0", "warn": "#fbbf24", "danger": "#f87171"}[s]

    st.markdown(f"""
    <div class="dec-box {css_class}">
        <div class="dec-label">Décision Sanitaire</div>
        <div class="dec-title">{decision['emoji']} &nbsp;{decision['title']}</div>
        <div class="dec-msg">{decision['message']}</div>
        <div class="score-row">
            <span class="score-label">NIVEAU DE PROTECTION EPI</span>
            <span class="score-val" style="color:{score_color}">{score}%</span>
        </div>
        <div class="score-track">
            <div class="score-fill {css_class}" style="width:{score}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        items_html = "".join(
            f'<div class="rp-item"><span class="dot ok"></span>{i}</div>'
            for i in decision["positive"]
        ) or '<div class="rp-item" style="color:#2a4a60">Aucun EPI détecté</div>'
        st.markdown(f'<div class="rp ok"><div class="rp-title">✓ Équipements détectés</div>{items_html}</div>', unsafe_allow_html=True)

    with c2:
        items_html = "".join(
            f'<div class="rp-item"><span class="dot miss"></span>{i}</div>'
            for i in decision["missing"]
        ) or '<div class="rp-item" style="color:#2a4a60">Aucun équipement manquant</div>'
        st.markdown(f'<div class="rp miss"><div class="rp-title">⚠ Équipements manquants</div>{items_html}</div>', unsafe_allow_html=True)

    if decision["negative"]:
        items_html = "".join(
            f'<div class="rp-item"><span class="dot bad"></span>{i}</div>'
            for i in decision["negative"]
        )
        st.markdown(f'<div class="rp bad" style="margin-top:12px"><div class="rp-title">✕ Non-conformités</div>{items_html}</div>', unsafe_allow_html=True)


def draw_overlay(frame, decision):
    color_map = {"ok": (0, 230, 160), "warn": (0, 165, 255), "danger": (60, 60, 240)}
    text_map = {
        "ok": f"ACCES AUTORISE  |  PROTECTION {decision['score']}%",
        "warn": f"PROTECTION INCOMPLETE  |  {decision['score']}%",
        "danger": "ACCES REFUSE  |  ALERTE SANITAIRE"
    }
    color = color_map[decision["status"]]
    text = text_map[decision["status"]]
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (4, 10, 22), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (0, 76), (w, 80), color, -1)
    cv2.putText(frame, text, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.05, color, 2, cv2.LINE_AA)
    y = 114
    for item in decision["positive"]:
        cv2.putText(frame, f"  {item}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 160), 1, cv2.LINE_AA)
        y += 22
    for item in decision["missing"]:
        cv2.putText(frame, f"  MANQUE: {item}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
        y += 22
    for item in decision.get("negative", []):
        cv2.putText(frame, f"  ALERTE: {item}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 240), 1, cv2.LINE_AA)
        y += 22
    return frame


# =========================================================
# SIDEBAR
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "Accueil"

with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-badge">
            <div class="sb-icon">🛡️</div>
            <div>
                <div class="sb-name">MedGuard</div>
                <div class="sb-version">v2.0 — PFA 2025</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Navigation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-nav">', unsafe_allow_html=True)

    pages = [
        ("🏠", "Accueil"),
        ("🖼️", "Image"),
        ("🎬", "Vidéo"),
        ("🎥", "Temps Réel"),
    ]
    for icon, name in pages:
        label = f"{icon}  {name}"
        if st.button(label, key=f"nav_{name}"):
            st.session_state.page = name
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Configuration</div>', unsafe_allow_html=True)
    st.markdown('<div style="padding:0 14px 8px;">', unsafe_allow_html=True)
    conf_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.25, 0.05, key="conf_global")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">Accélération</div>', unsafe_allow_html=True)
    st.markdown('<div style="padding:0 14px 12px;">', unsafe_allow_html=True)

    if DEVICE == "mps":
        st.markdown('<span class="badge badge-green">⚡ Apple Silicon MPS</span>', unsafe_allow_html=True)
    elif DEVICE == "cpu":
        st.markdown('<span class="badge badge-amber">🔲 Mode CPU</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-green">⚡ GPU CUDA</span>', unsafe_allow_html=True)

    st.markdown(f'<p style="font-size:11px;color:#1e3a50;margin-top:8px;">Device : <code style="color:#00c070">{DEVICE}</code></p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-footer">
        MedGuard v2.0<br>
        Projet PFA — EMSI 2025<br>
        Détection EPI médicaux YOLO
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# PAGE ACCUEIL
# =========================================================

def home_page():
    st.markdown("""
    <div class="hero">
        <div class="hero-glow-tl"></div>
        <div class="hero-glow-br"></div>
        <div class="hero-grid-lines"></div>
        <div class="hero-content">
            <div class="hero-text">
                <div class="hero-eyebrow">🧬 Intelligence Artificielle Médicale</div>
                <div class="hero-h1">Détection <span>EPI Médicaux</span><br>en Temps Réel</div>
                <div class="hero-sub">
                    Système de vision par ordinateur basé sur YOLOv8 pour la détection
                    automatique des équipements de protection individuelle dans les zones médicales.
                    Prévention active des risques sanitaires.
                </div>
                <div class="hero-tags">
                    <span class="hero-tag">🔬 YOLOv8</span>
                    <span class="hero-tag">⚡ Temps réel</span>
                    <span class="hero-tag">🛡️ 13 classes EPI</span>
                    <span class="hero-tag">🚨 Alertes automatiques</span>
                    <span class="hero-tag">📊 Score de protection</span>
                    <span class="hero-tag">🎬 Analyse vidéo</span>
                </div>
            </div>
            <div class="scan-wrap">
                <div class="scan-ring"></div>
                <div class="scan-ring"></div>
                <div class="scan-ring"></div>
                <div class="scan-line"></div>
                <div class="scan-core">🧑‍⚕️</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stats-bar">
        <div class="stat-card g">
            <div class="stat-num g">13</div>
            <div class="stat-label">Classes détectées</div>
        </div>
        <div class="stat-card b">
            <div class="stat-num b">5</div>
            <div class="stat-label">EPI obligatoires</div>
        </div>
        <div class="stat-card a">
            <div class="stat-num a">3</div>
            <div class="stat-label">Modes d'analyse</div>
        </div>
        <div class="stat-card r">
            <div class="stat-num r">3</div>
            <div class="stat-label">Niveaux d'alerte</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-title"><span class="sec-title-accent">01</span> Niveaux de décision</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="level-grid">
        <div class="level-card ok">
            <div class="level-icon">✅</div>
            <div class="level-title">Accès Autorisé</div>
            <div class="level-desc">La personne porte l'ensemble des EPI : masque, gants, blouse médicale, charlotte et lunettes. Score de protection : 100%.</div>
        </div>
        <div class="level-card warn">
            <div class="level-icon">⚠️</div>
            <div class="level-title">Protection Incomplète</div>
            <div class="level-desc">Certains équipements manquants. Le système affiche les EPI non confirmés avec un score de protection partiel.</div>
        </div>
        <div class="level-card danger">
            <div class="level-icon">🚨</div>
            <div class="level-title">Accès Refusé</div>
            <div class="level-desc">Absence confirmée d'EPI ou comportement interdit (manger, boire). Accès bloqué et alerte critique déclenchée.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-title"><span class="sec-title-accent">02</span> Classes détectées</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("""
        <div class="rp ok" style="padding:22px">
            <div class="rp-title">✓ EPI Conformes</div>
            <div class="ctag-row">
                <span class="ctag ok">😷 Mask</span>
                <span class="ctag ok">🧤 Gloves</span>
                <span class="ctag ok">👒 Head Mask</span>
                <span class="ctag ok">🥼 Lab Coat</span>
                <span class="ctag ok">🥽 Goggles</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="rp bad" style="padding:22px">
            <div class="rp-title">✕ EPI Non-conformes & Comportements</div>
            <div class="ctag-row">
                <span class="ctag no">❌ No Mask</span>
                <span class="ctag no">❌ No Gloves</span>
                <span class="ctag no">❌ No Head Mask</span>
                <span class="ctag no">❌ No Lab Coat</span>
                <span class="ctag no">❌ No Goggles</span>
                <span class="ctag warn">🍽️ Eating</span>
                <span class="ctag warn">🥤 Drinking</span>
                <span class="ctag neutral">🏢 EMSI</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sec-title"><span class="sec-title-accent">03</span> Modes d\'analyse</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3, gap="medium")
    for col, icon, title, desc in [
        (m1, "🖼️", "Analyse par image", "Chargez une photo JPG/PNG pour une analyse instantanée du niveau de protection EPI avec visualisation des détections."),
        (m2, "🎬", "Analyse vidéo", "Importez une vidéo MP4 pour traiter chaque frame et exporter un rapport de conformité avec overlay de détections."),
        (m3, "🎥", "Temps réel", "Flux caméra en direct avec overlay YOLO instantané, idéal pour un contrôle d'accès automatisé aux zones médicales.")
    ]:
        with col:
            col.markdown(f"""
            <div class="info-box" style="text-align:center; padding:24px;">
                <div style="font-size:32px;margin-bottom:12px;">{icon}</div>
                <h4 style="color:#fff!important;font-size:14px;margin-bottom:8px;">{title}</h4>
                <p style="font-size:12.5px;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)


# =========================================================
# DETECTION TABLE
# =========================================================

def render_detection_table(detections):
    if not detections:
        st.markdown("""
        <div class="info-box" style="border-color:rgba(245,158,11,0.2);">
            <h4 style="color:#fbbf24!important">Aucun objet détecté</h4>
            <p>Essayez de réduire le seuil de confiance ou d'utiliser une image de meilleure résolution.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    rows_html = ""
    for d in detections:
        c = d["Confiance"]
        badge = "ch" if c >= 0.7 else ("cm" if c >= 0.45 else "cl")
        clean = normalize_label(d["Classe"])
        if clean in POSITIVE_CLASSES:
            tag, sym = "ok", "✓"
        elif clean in NEGATIVE_CLASSES:
            tag, sym = "no", "✕"
        elif clean in DANGER_CLASSES:
            tag, sym = "warn", "⚠"
        else:
            tag, sym = "neutral", "—"
        rows_html += f"""
        <tr>
            <td><span class="ctag {tag}" style="padding:3px 10px;font-size:12px;">{sym} {d['Classe']}</span></td>
            <td><span class="cbadge {badge}">{c:.2f}</span></td>
        </tr>
        """

    counts = {}
    for d in detections:
        counts[d["Classe"]] = counts.get(d["Classe"], 0) + 1
    count_rows = "".join(
        f'<div class="rp-item"><span class="dot ok"></span>{lbl}&nbsp;<span style="color:#1e3a50;font-family:JetBrains Mono,monospace">×{cnt}</span></div>'
        for lbl, cnt in sorted(counts.items())
    )

    ca, cb = st.columns([3, 2], gap="medium")
    with ca:
        st.markdown(f"""
        <div class="rp ok" style="padding:20px">
            <div class="rp-title">Détail des détections</div>
            <table class="det-table">
                <thead><tr><th>Classe</th><th>Confiance</th></tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)
    with cb:
        st.markdown(f"""
        <div class="rp ok" style="padding:20px">
            <div class="rp-title">Résumé par classe</div>
            {count_rows}
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# PAGE IMAGE
# =========================================================

def image_detection_page():
    st.markdown("""
    <div class="topbar">
        <div class="topbar-left">
            <div class="topbar-icon">🖼️</div>
            <div>
                <h1>Détection par Image</h1>
                <p>Analysez une photo pour évaluer la conformité EPI et obtenir une décision sanitaire.</p>
            </div>
        </div>
        <div class="topbar-right">
            <span class="badge badge-green">● YOLO Actif</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("Modèle YOLO introuvable. Vérifiez le chemin `best.pt`.")
        return

    uploaded_file = st.file_uploader(
        "Déposez votre image ici",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="visible"
    )
    st.markdown("""
    <div class="upload-hint">
        📌 <strong>Formats acceptés :</strong> JPG, JPEG, PNG, WEBP
        &nbsp;·&nbsp; Résolution recommandée ≥ 640×640 px
        &nbsp;·&nbsp; Taille max : 200 MB
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file is None:
        return

    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="img-panel"><div class="img-panel-label">Image originale</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀  Lancer l'analyse EPI", use_container_width=True):
        with st.spinner("Analyse YOLO en cours…"):
            image_np = np.array(image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            results = predict_yolo(source=image_bgr, conf=conf_threshold, imgsz=1280)

            if results is None:
                st.error("Erreur lors de l'exécution du modèle.")
                return

            result = results[0]
            annotated = result.plot()
            detections, labels = extract_detections(result, model)
            decision = evaluate_protection(labels)

        with col2:
            st.markdown('<div class="img-panel"><div class="img-panel-label">Résultat — Détections YOLO</div>', unsafe_allow_html=True)
            st.image(annotated, channels="BGR", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sec-title"><span class="sec-title-accent">→</span> Décision Sanitaire</div>', unsafe_allow_html=True)
        render_decision(decision)

        st.markdown('<div class="sec-title"><span class="sec-title-accent">→</span> Objets Détectés</div>', unsafe_allow_html=True)
        render_detection_table(detections)


# =========================================================
# PAGE VIDEO
# =========================================================

def video_detection_page():
    st.markdown("""
    <div class="topbar">
        <div class="topbar-left">
            <div class="topbar-icon blue">🎬</div>
            <div>
                <h1>Analyse Vidéo</h1>
                <p>Importez une vidéo MP4/AVI pour analyser la conformité EPI frame par frame.</p>
            </div>
        </div>
        <div class="topbar-right">
            <span class="badge badge-green">● YOLO Actif</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("Modèle YOLO introuvable. Vérifiez le chemin `best.pt`.")
        return

    uploaded_video = st.file_uploader(
        "Déposez votre vidéo ici",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="visible"
    )
    st.markdown("""
    <div class="upload-hint">
        🎬 <strong>Formats acceptés :</strong> MP4, AVI, MOV, MKV
        &nbsp;·&nbsp; La vidéo sera traitée frame par frame avec overlay de détection
        &nbsp;·&nbsp; Le résultat peut être téléchargé
    </div>
    """, unsafe_allow_html=True)

    if uploaded_video is None:
        return

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3, gap="medium")
    with col_cfg1:
        video_conf = st.slider("Seuil de confiance", 0.0, 1.0, conf_threshold, 0.05, key="vid_conf")
    with col_cfg2:
        frame_skip = st.selectbox(
            "Fréquence d'analyse",
            ["Toutes les frames", "1 frame / 2", "1 frame / 3", "1 frame / 5"],
            index=1
        )
    with col_cfg3:
        output_res = st.selectbox(
            "Résolution sortie",
            ["Originale", "1280×720 (HD)", "640×480 (SD)"],
            index=1
        )

    skip_map = {"Toutes les frames": 1, "1 frame / 2": 2, "1 frame / 3": 3, "1 frame / 5": 5}
    frame_step = skip_map[frame_skip]

    if not st.button("🎬  Lancer l'analyse vidéo", use_container_width=True):
        return

    # Write uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_res == "1280×720 (HD)":
        out_w, out_h = 1280, 720
    elif output_res == "640×480 (SD)":
        out_w, out_h = 640, 480
    else:
        out_w, out_h = orig_w, orig_h

    out_path = tmp_path.replace(".mp4", "_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps / frame_step, (out_w, out_h))

    # UI placeholders
    col_prev, col_info = st.columns([3, 2], gap="large")
    with col_prev:
        frame_placeholder = st.empty()
    with col_info:
        decision_placeholder = st.empty()

    st.markdown('<div class="video-progress-wrap">', unsafe_allow_html=True)
    progress_label = st.empty()
    progress_bar = st.progress(0)
    st.markdown('</div>', unsafe_allow_html=True)

    # Stats accumulators
    frame_idx = 0
    processed = 0
    status_counts = {"ok": 0, "warn": 0, "danger": 0}
    last_decision = None
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % frame_step != 0:
            continue

        # Resize
        if (out_w, out_h) != (orig_w, orig_h):
            frame = cv2.resize(frame, (out_w, out_h))

        results = predict_yolo(source=frame, conf=video_conf, imgsz=960)
        if results:
            result = results[0]
            annotated_frame = result.plot()
            _, labels = extract_detections(result, model)
            decision = evaluate_protection(labels)
            last_decision = decision
            annotated_frame = draw_overlay(annotated_frame, decision)
            status_counts[decision["status"]] += 1
        else:
            annotated_frame = frame.copy()

        writer.write(annotated_frame)
        processed += 1

        # Update preview every 5 processed frames
        if processed % 5 == 0 or frame_idx <= frame_step:
            preview_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(preview_rgb, use_container_width=True, caption=f"Frame {frame_idx}/{total_frames}")

            elapsed = time.time() - t_start
            fps_proc = processed / max(elapsed, 0.01)
            pct = min(frame_idx / max(total_frames, 1), 1.0)
            progress_bar.progress(pct)
            progress_label.markdown(
                f'<div class="video-frame-label">'
                f'<span>Frame {frame_idx} / {total_frames}</span>'
                f'<span>{fps_proc:.1f} fps · {elapsed:.0f}s</span>'
                f'</div>',
                unsafe_allow_html=True
            )

            if last_decision:
                s = last_decision["status"]
                em = {"ok": "✅", "warn": "⚠️", "danger": "🚨"}[s]
                col = {"ok": "#00e6a0", "warn": "#fbbf24", "danger": "#f87171"}[s]
                decision_placeholder.markdown(f"""
                <div class="rp ok" style="padding:20px;border-color:rgba(0,230,160,.15)">
                    <div class="rp-title">Décision en cours</div>
                    <div style="font-size:22px;font-weight:800;color:{col};margin-bottom:10px">{em} {last_decision['title']}</div>
                    <div class="rp-title" style="margin-top:14px">Résumé</div>
                    <div class="rp-item"><span class="dot ok"></span>✅ Autorisé : {status_counts['ok']}</div>
                    <div class="rp-item"><span class="dot miss"></span>⚠️ Incomplet : {status_counts['warn']}</div>
                    <div class="rp-item"><span class="dot bad"></span>🚨 Refusé : {status_counts['danger']}</div>
                </div>
                """, unsafe_allow_html=True)

    cap.release()
    writer.release()

    progress_bar.progress(1.0)
    progress_label.markdown(
        f'<div class="video-frame-label"><span>✅ Traitement terminé — {processed} frames analysées</span><span>{time.time()-t_start:.1f}s</span></div>',
        unsafe_allow_html=True
    )

    # Final stats
    total_proc = sum(status_counts.values()) or 1
    st.markdown('<div class="sec-title"><span class="sec-title-accent">→</span> Rapport Final</div>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3, gap="medium")
    for col, key, em, color, label in [
        (r1, "ok", "✅", "#00e6a0", "Frames — Accès Autorisé"),
        (r2, "warn", "⚠️", "#fbbf24", "Frames — Protection Incomplète"),
        (r3, "danger", "🚨", "#f87171", "Frames — Accès Refusé"),
    ]:
        cnt = status_counts[key]
        pct = int(cnt / total_proc * 100)
        with col:
            col.markdown(f"""
            <div class="stat-card" style="text-align:center">
                <div class="stat-num" style="color:{color}">{cnt}</div>
                <div class="stat-label">{label}</div>
                <div style="margin-top:12px;height:4px;border-radius:2px;background:rgba(255,255,255,.05)">
                    <div style="width:{pct}%;height:100%;border-radius:2px;background:{color}"></div>
                </div>
                <div style="font-size:11px;color:#2a4a60;margin-top:6px;font-family:JetBrains Mono,monospace">{pct}% des frames</div>
            </div>
            """, unsafe_allow_html=True)

    # Download
    with open(out_path, "rb") as f:
        st.download_button(
            "⬇️  Télécharger la vidéo annotée",
            data=f,
            file_name="medguard_detection.mp4",
            mime="video/mp4",
            use_container_width=True
        )

    os.unlink(tmp_path)
    try:
        os.unlink(out_path)
    except Exception:
        pass


# =========================================================
# PAGE TEMPS RÉEL
# =========================================================

class YOLOProcessor(VideoProcessorBase):
    def __init__(self):
        self.conf_threshold = 0.25
        self.imgsz = 960
        self.frame_step = 1
        self.frame_count = 0
        self.last_frame = None
        self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.model is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        self.frame_count += 1
        if self.frame_step > 1 and self.frame_count % self.frame_step != 0 and self.last_frame is not None:
            return av.VideoFrame.from_ndarray(self.last_frame, format="bgr24")
        try:
            results = self.model.predict(source=img, conf=self.conf_threshold, imgsz=self.imgsz, device=DEVICE, verbose=False)
        except Exception:
            results = self.model.predict(source=img, conf=self.conf_threshold, imgsz=self.imgsz, device="cpu", verbose=False)
        result = results[0]
        annotated = result.plot()
        _, labels = extract_detections(result, self.model)
        decision = evaluate_protection(labels)
        annotated = draw_overlay(annotated, decision)
        self.last_frame = annotated
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def realtime_page():
    st.markdown("""
    <div class="topbar">
        <div class="topbar-left">
            <div class="topbar-icon purple">🎥</div>
            <div>
                <h1>Détection Temps Réel</h1>
                <p>Flux caméra en direct avec analyse EPI continue et overlay de décision sanitaire.</p>
            </div>
        </div>
        <div class="topbar-right">
            <span class="badge badge-green">● WebRTC</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("Modèle YOLO introuvable.")
        return

    if not WEBRTC_AVAILABLE:
        st.error("Modules manquants. Installez :")
        st.code("pip install streamlit-webrtc av")
        return

    st.markdown("""
    <div class="upload-hint">
        🎥 <strong>Instruction :</strong> Cliquez sur <em>START</em> et autorisez l'accès à la caméra.
        Les détections EPI et la décision sanitaire s'affichent directement en overlay sur le flux vidéo.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        live_conf = st.slider("Confiance", 0.0, 1.0, 0.25, 0.05, key="live_conf")
    with col2:
        quality = st.selectbox("Résolution", ["Full HD 1080p", "HD 720p", "SD 480p"], index=1)
    with col3:
        mode = st.selectbox("Mode", ["Qualité max (1fps/frame)", "Équilibré (1/2)", "Fluide (1/3)"], index=1)

    res_map = {"Full HD 1080p": (1920, 1080, 1280), "HD 720p": (1280, 720, 960), "SD 480p": (854, 480, 640)}
    vid_w, vid_h, yolo_imgsz = res_map[quality]
    step_map = {"Qualité max (1fps/frame)": 1, "Équilibré (1/2)": 2, "Fluide (1/3)": 3}
    step = step_map[mode]

    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    ctx = webrtc_streamer(
        key="medguard-rt",
        video_processor_factory=YOLOProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={
            "video": {"width": {"ideal": vid_w}, "height": {"ideal": vid_h}, "frameRate": {"ideal": 30}},
            "audio": False
        },
        async_processing=True,
    )
    if ctx.video_processor:
        ctx.video_processor.conf_threshold = live_conf
        ctx.video_processor.imgsz = yolo_imgsz
        ctx.video_processor.frame_step = step

    st.markdown("""
    <div class="info-box" style="margin-top:20px">
        <h4>📌 Overlay vidéo — Légende</h4>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px">
            <div>
                <div class="rp-item"><span class="dot ok"></span><span style="color:#00e6a0;font-weight:600">Bandeau vert</span> — Accès autorisé</div>
                <div class="rp-item"><span class="dot miss"></span><span style="color:#fbbf24;font-weight:600">Bandeau orange</span> — Protection incomplète</div>
                <div class="rp-item"><span class="dot bad"></span><span style="color:#f87171;font-weight:600">Bandeau rouge</span> — Accès refusé</div>
            </div>
            <div>
                <div class="rp-item" style="color:#3a5a70">✓ EPI détectés en texte vert sur la vidéo</div>
                <div class="rp-item" style="color:#3a5a70">⚠ Équipements manquants en orange</div>
                <div class="rp-item" style="color:#3a5a70">✕ Alertes critiques en rouge vif</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("""
        <div class="info-box">
            <h4>💻 Optimisation Apple Silicon</h4>
            <p>Mode MPS activé automatiquement sur Mac M1/M2/M3. Si la vidéo est saccadée, passez en mode "Équilibré" ou "Fluide".</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="info-box">
            <h4>📡 Connexion WebRTC</h4>
            <p>La détection s'effectue côté serveur via WebRTC. Autorisez l'accès caméra dans le navigateur et vérifiez que les ports STUN ne sont pas bloqués.</p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# ROUTER
# =========================================================

if st.session_state.page == "Accueil":
    home_page()
elif st.session_state.page == "Image":
    image_detection_page()
elif st.session_state.page == "Vidéo":
    video_detection_page()
elif st.session_state.page == "Temps Réel":
    realtime_page()
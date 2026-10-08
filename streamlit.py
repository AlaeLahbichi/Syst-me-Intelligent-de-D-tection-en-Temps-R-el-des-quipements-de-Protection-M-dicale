import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

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
# CSS GLOBAL — DARK MEDICAL THEME
# =========================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    * { box-sizing: border-box; }

    .stApp {
        background: #060d1a;
        color: #e8f0fe;
        font-family: 'Space Grotesk', sans-serif;
    }

    /* ---------- SIDEBAR ---------- */
    section[data-testid="stSidebar"] {
        background: #080f1f !important;
        border-right: 1px solid rgba(0,200,150,0.12);
    }

    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: #a8c0d6 !important;
    }

    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 18px 0 8px;
        border-bottom: 1px solid rgba(0,200,150,0.15);
        margin-bottom: 20px;
    }

    .sidebar-logo-icon {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #00c896, #0070f3);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        box-shadow: 0 0 20px rgba(0,200,150,0.3);
    }

    .sidebar-logo-text {
        font-weight: 700;
        font-size: 20px;
        color: #ffffff;
        letter-spacing: -0.3px;
    }

    .sidebar-logo-sub {
        font-size: 11px;
        color: #00c896;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .nav-section {
        font-size: 10px;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        color: #506070 !important;
        margin: 24px 0 8px;
        font-weight: 600;
    }

    /* ---------- NAV BUTTONS ---------- */
    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.07);
        padding: 11px 16px;
        background: transparent;
        color: #8aa4c0;
        font-weight: 500;
        font-size: 14px;
        text-align: left;
        font-family: 'Space Grotesk', sans-serif;
        transition: all 0.18s ease;
        margin-bottom: 4px;
    }

    div.stButton > button:hover {
        background: rgba(0,200,150,0.08);
        border-color: rgba(0,200,150,0.25);
        color: #00e8a8;
        transform: none;
    }

    div.stButton > button:active {
        background: rgba(0,200,150,0.15);
    }

    /* ---------- SLIDERS ---------- */
    .stSlider .stSlider > div > div > div > div {
        background: #00c896 !important;
    }

    /* ---------- FILE UPLOADER ---------- */
    .stFileUploader {
        background: rgba(255,255,255,0.02) !important;
        border: 1px dashed rgba(0,200,150,0.3) !important;
        border-radius: 16px !important;
    }

    /* ---------- HERO / ACCUEIL ---------- */
    .hero-wrap {
        position: relative;
        padding: 50px 48px 46px;
        border-radius: 24px;
        background: #0b1528;
        border: 1px solid rgba(0,200,150,0.15);
        overflow: hidden;
        margin-bottom: 32px;
    }

    .hero-wrap::before {
        content: '';
        position: absolute;
        top: -80px; left: -80px;
        width: 380px; height: 380px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(0,200,150,0.12) 0%, transparent 65%);
        pointer-events: none;
    }

    .hero-wrap::after {
        content: '';
        position: absolute;
        bottom: -60px; right: -60px;
        width: 280px; height: 280px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(0,112,243,0.12) 0%, transparent 65%);
        pointer-events: none;
    }

    .hero-tag {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        background: rgba(0,200,150,0.1);
        border: 1px solid rgba(0,200,150,0.3);
        color: #00c896;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 20px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.15;
        letter-spacing: -0.8px;
        margin-bottom: 16px;
    }

    .hero-title span {
        background: linear-gradient(90deg, #00c896, #0070f3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-subtitle {
        font-size: 16px;
        color: #7a9ab5;
        line-height: 1.75;
        max-width: 620px;
        margin-bottom: 24px;
    }

    .hero-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }

    .hero-pill {
        padding: 7px 16px;
        border-radius: 20px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        color: #a0bcd4;
        font-size: 13px;
        font-weight: 500;
    }

    /* SCANNER ANIMATION */
    .scanner-container {
        position: relative;
        width: 200px;
        height: 200px;
        margin: 0 auto;
    }

    .scanner-ring {
        position: absolute;
        inset: 0;
        border-radius: 50%;
        border: 2px solid rgba(0,200,150,0.25);
        animation: ring-pulse 2.4s ease-in-out infinite;
    }

    .scanner-ring:nth-child(2) {
        inset: 20px;
        border-color: rgba(0,200,150,0.35);
        animation-delay: 0.6s;
    }

    .scanner-ring:nth-child(3) {
        inset: 40px;
        border-color: rgba(0,200,150,0.55);
        animation-delay: 1.2s;
    }

    .scanner-core {
        position: absolute;
        inset: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg, rgba(0,200,150,0.2), rgba(0,112,243,0.2));
        border: 2px solid rgba(0,200,150,0.7);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 34px;
        animation: core-glow 2.4s ease-in-out infinite;
    }

    @keyframes ring-pulse {
        0%, 100% { transform: scale(1); opacity: 0.4; }
        50% { transform: scale(1.08); opacity: 1; }
    }

    @keyframes core-glow {
        0%, 100% { box-shadow: 0 0 15px rgba(0,200,150,0.3); }
        50% { box-shadow: 0 0 35px rgba(0,200,150,0.7); }
    }

    /* ---------- METRIC CARDS ---------- */
    .metrics-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 32px;
    }

    .metric-card {
        padding: 22px;
        border-radius: 16px;
        background: #0b1528;
        border: 1px solid rgba(255,255,255,0.07);
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 0 0 16px 16px;
    }

    .metric-card.green::before { background: linear-gradient(90deg, #00c896, transparent); }
    .metric-card.blue::before { background: linear-gradient(90deg, #0070f3, transparent); }
    .metric-card.amber::before { background: linear-gradient(90deg, #f59e0b, transparent); }
    .metric-card.red::before { background: linear-gradient(90deg, #ef4444, transparent); }

    .metric-num {
        font-size: 36px;
        font-weight: 700;
        color: #ffffff;
        line-height: 1;
        margin-bottom: 6px;
        font-family: 'DM Mono', monospace;
    }

    .metric-num.green { color: #00c896; }
    .metric-num.blue { color: #3b8ff7; }
    .metric-num.amber { color: #f59e0b; }
    .metric-num.red { color: #ef4444; }

    .metric-label {
        font-size: 13px;
        color: #5a7a96;
        font-weight: 500;
    }

    /* ---------- SECTION TITLE ---------- */
    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #e8f0fe;
        margin: 28px 0 16px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .section-title::after {
        content: '';
        flex: 1;
        height: 1px;
        background: rgba(255,255,255,0.06);
    }

    /* ---------- INFO CARDS ---------- */
    .info-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 28px;
    }

    .info-card {
        padding: 24px;
        border-radius: 16px;
        background: #0b1528;
        border: 1px solid rgba(255,255,255,0.07);
        position: relative;
        overflow: hidden;
    }

    .info-card .icon-wrap {
        width: 44px; height: 44px;
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 22px;
        margin-bottom: 16px;
    }

    .info-card.success .icon-wrap { background: rgba(0,200,150,0.12); }
    .info-card.warning .icon-wrap { background: rgba(245,158,11,0.12); }
    .info-card.danger .icon-wrap { background: rgba(239,68,68,0.12); }

    .info-card h4 {
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 8px;
        color: #ddeeff;
    }

    .info-card.success h4 { color: #00e8a8; }
    .info-card.warning h4 { color: #fbbf24; }
    .info-card.danger h4 { color: #f87171; }

    .info-card p {
        font-size: 13.5px;
        color: #5a7a96;
        line-height: 1.65;
    }

    /* CLASS TAGS */
    .class-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 12px 0;
    }

    .ctag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 500;
    }

    .ctag-ok {
        background: rgba(0,200,150,0.08);
        border: 1px solid rgba(0,200,150,0.25);
        color: #00e8a8;
    }

    .ctag-no {
        background: rgba(239,68,68,0.08);
        border: 1px solid rgba(239,68,68,0.25);
        color: #f87171;
    }

    .ctag-warn {
        background: rgba(245,158,11,0.08);
        border: 1px solid rgba(245,158,11,0.25);
        color: #fbbf24;
    }

    .ctag-neutral {
        background: rgba(148,163,184,0.08);
        border: 1px solid rgba(148,163,184,0.2);
        color: #94a3b8;
    }

    /* ---------- PAGE TITLE BAR ---------- */
    .page-header {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 28px 0 24px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 28px;
    }

    .page-header-icon {
        width: 52px; height: 52px;
        border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        font-size: 26px;
    }

    .page-header-icon.img { background: linear-gradient(135deg, rgba(0,200,150,0.2), rgba(0,200,150,0.05)); }
    .page-header-icon.vid { background: linear-gradient(135deg, rgba(0,112,243,0.2), rgba(0,112,243,0.05)); }

    .page-header h1 {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 4px;
        letter-spacing: -0.4px;
    }

    .page-header p {
        font-size: 14px;
        color: #5a7a96;
        margin: 0;
    }

    /* ---------- DECISION BOX ---------- */
    .decision-box {
        padding: 28px 32px;
        border-radius: 20px;
        margin: 24px 0;
        position: relative;
        overflow: hidden;
    }

    .decision-box::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 5px;
        border-radius: 4px 0 0 4px;
    }

    .decision-box.success {
        background: rgba(0,200,150,0.06);
        border: 1px solid rgba(0,200,150,0.25);
    }
    .decision-box.success::before { background: #00c896; }

    .decision-box.warning {
        background: rgba(245,158,11,0.06);
        border: 1px solid rgba(245,158,11,0.30);
    }
    .decision-box.warning::before { background: #f59e0b; }

    .decision-box.danger {
        background: rgba(239,68,68,0.06);
        border: 1px solid rgba(239,68,68,0.30);
    }
    .decision-box.danger::before { background: #ef4444; }

    .decision-title {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .decision-box.success .decision-title { color: #00e8a8; }
    .decision-box.warning .decision-title { color: #fbbf24; }
    .decision-box.danger .decision-title { color: #f87171; }

    .decision-msg {
        font-size: 14.5px;
        color: #7a9ab5;
        margin-bottom: 18px;
    }

    .score-bar-wrap {
        margin-top: 12px;
    }

    .score-bar-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
    }

    .score-bar-label span:first-child {
        font-size: 13px;
        color: #5a7a96;
    }

    .score-bar-label span:last-child {
        font-size: 15px;
        font-weight: 700;
        font-family: 'DM Mono', monospace;
    }

    .score-bar-bg {
        height: 8px;
        border-radius: 4px;
        background: rgba(255,255,255,0.06);
        overflow: hidden;
    }

    .score-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }

    .score-bar-fill.success { background: linear-gradient(90deg, #00c896, #00e8a8); }
    .score-bar-fill.warning { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .score-bar-fill.danger { background: linear-gradient(90deg, #ef4444, #f87171); }

    /* DETECTION RESULT GRID */
    .result-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        margin-top: 20px;
    }

    .result-panel {
        padding: 20px;
        border-radius: 16px;
        background: #0b1528;
        border: 1px solid rgba(255,255,255,0.07);
    }

    .result-panel-title {
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 14px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }

    .result-panel.ok .result-panel-title { color: #00c896; }
    .result-panel.miss .result-panel-title { color: #f59e0b; }
    .result-panel.bad .result-panel-title { color: #ef4444; }

    .result-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        font-size: 14px;
        color: #8aa4c0;
    }

    .result-item:last-child { border-bottom: none; }

    .result-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    .dot-ok { background: #00c896; }
    .dot-miss { background: #f59e0b; }
    .dot-bad { background: #ef4444; }

    /* DETECTION TABLE */
    .det-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13.5px;
        margin-top: 16px;
    }

    .det-table th {
        text-align: left;
        padding: 10px 14px;
        font-size: 11px;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #4a6a82;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }

    .det-table td {
        padding: 10px 14px;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        color: #8aa4c0;
    }

    .det-table tr:last-child td { border-bottom: none; }

    .det-table tr:hover td {
        background: rgba(255,255,255,0.02);
    }

    .conf-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 10px;
        font-family: 'DM Mono', monospace;
        font-size: 12px;
        font-weight: 500;
    }

    .conf-high { background: rgba(0,200,150,0.1); color: #00c896; }
    .conf-mid  { background: rgba(245,158,11,0.1); color: #f59e0b; }
    .conf-low  { background: rgba(239,68,68,0.1);  color: #ef4444; }

    /* UPLOAD ZONE */
    .upload-hint {
        padding: 16px;
        border-radius: 12px;
        background: rgba(0,112,243,0.06);
        border: 1px solid rgba(0,112,243,0.18);
        color: #7aa8d6;
        font-size: 13.5px;
        line-height: 1.6;
        margin-top: 10px;
    }

    /* VIDEO PAGE CARDS */
    .video-info-card {
        padding: 20px 24px;
        border-radius: 16px;
        background: #0b1528;
        border: 1px solid rgba(0,200,150,0.12);
        margin-top: 20px;
    }

    .video-info-card h4 {
        font-size: 15px;
        font-weight: 600;
        color: #00e8a8;
        margin-bottom: 8px;
    }

    .video-info-card p {
        font-size: 13.5px;
        color: #5a7a96;
        line-height: 1.65;
    }

    /* STATUS PILLS */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }

    .status-pill.ok {
        background: rgba(0,200,150,0.12);
        border: 1px solid rgba(0,200,150,0.3);
        color: #00e8a8;
    }

    .status-pill.warn {
        background: rgba(245,158,11,0.12);
        border: 1px solid rgba(245,158,11,0.3);
        color: #fbbf24;
    }

    .status-pill.cpu {
        background: rgba(148,163,184,0.1);
        border: 1px solid rgba(148,163,184,0.2);
        color: #94a3b8;
    }

    /* SUBHEADER */
    .stMarkdown h2 {
        color: #e8f0fe !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }

    /* SPINNER */
    .stSpinner > div {
        border-top-color: #00c896 !important;
    }

    /* ERROR / WARNING / SUCCESS */
    .stAlert {
        border-radius: 12px !important;
    }

    /* OVERRIDE DEFAULT TEXT COLORS */
    .stMarkdown p { color: #8aa4c0; }
    h1, h2, h3, h4 { color: #e8f0fe !important; }

    /* DATAFRAME */
    .stDataFrame { border-radius: 12px !important; overflow: hidden; }

    /* IMAGE CAPTION */
    .image-panel-title {
        font-size: 12px;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #4a6a82;
        font-weight: 600;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }

    .image-panel {
        padding: 16px;
        border-radius: 16px;
        background: #0b1528;
        border: 1px solid rgba(255,255,255,0.07);
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
# MODEL LOADING
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
    "head mask": "Charlotte / Couvre-chef médical",
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

IGNORED_CLASSES = {
    "emsi": "Logo EMSI"
}

REQUIRED_EQUIPMENT = {
    "Masque de protection",
    "Gants",
    "Charlotte / Couvre-chef médical",
    "Blouse médicale",
    "Lunettes de protection"
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
    detections = []
    labels = []
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
    detected_positive = set()
    detected_negative = set()
    detected_danger = set()
    ignored = set()

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

    missing_equipment = REQUIRED_EQUIPMENT - detected_positive
    protection_score = int((len(detected_positive) / len(REQUIRED_EQUIPMENT)) * 100)

    if detected_danger:
        return {
            "status": "danger",
            "title": "⛔  Accès impossible",
            "message": "Comportement interdit détecté dans une zone médicale stérile.",
            "score": protection_score,
            "details": sorted(list(detected_danger)),
            "positive": sorted(list(detected_positive)),
            "negative": sorted(list(detected_negative)),
            "missing": sorted(list(missing_equipment)),
            "ignored": sorted(list(ignored))
        }

    if detected_negative:
        return {
            "status": "danger",
            "title": "🚨  Accès refusé",
            "message": "La personne ne respecte pas les conditions de protection sanitaire obligatoires.",
            "score": protection_score,
            "details": sorted(list(detected_negative)),
            "positive": sorted(list(detected_positive)),
            "negative": sorted(list(detected_negative)),
            "missing": sorted(list(missing_equipment)),
            "ignored": sorted(list(ignored))
        }

    if len(missing_equipment) == 0:
        return {
            "status": "success",
            "title": "✅  Accès autorisé",
            "message": "Tous les équipements de protection individuelle sont détectés et conformes.",
            "score": 100,
            "details": sorted(list(detected_positive)),
            "positive": sorted(list(detected_positive)),
            "negative": [],
            "missing": [],
            "ignored": sorted(list(ignored))
        }

    return {
        "status": "warning",
        "title": "⚠️  Protection incomplète",
        "message": "Certains équipements obligatoires ne sont pas confirmés par le système.",
        "score": protection_score,
        "details": sorted(list(detected_positive)),
        "positive": sorted(list(detected_positive)),
        "negative": [],
        "missing": sorted(list(missing_equipment)),
        "ignored": sorted(list(ignored))
    }


def render_decision(decision):
    score = decision["score"]
    status = decision["status"]
    bar_color = status

    conf_badge_color = "#00c896" if status == "success" else ("#f59e0b" if status == "warning" else "#ef4444")

    st.markdown(f"""
    <div class="decision-box {status}">
        <div class="decision-title">{decision['title']}</div>
        <div class="decision-msg">{decision['message']}</div>
        <div class="score-bar-wrap">
            <div class="score-bar-label">
                <span>Niveau de protection EPI</span>
                <span style="color: {conf_badge_color}; font-family: 'DM Mono', monospace;">{score}%</span>
            </div>
            <div class="score-bar-bg">
                <div class="score-bar-fill {bar_color}" style="width: {score}%;"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="result-panel ok"><div class="result-panel-title">✓ Équipements détectés</div>', unsafe_allow_html=True)
        if decision["positive"]:
            for item in decision["positive"]:
                st.markdown(f'<div class="result-item"><span class="result-dot dot-ok"></span>{item}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-item" style="color:#4a6a82">Aucun équipement détecté</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="result-panel miss"><div class="result-panel-title">⚠ Équipements manquants</div>', unsafe_allow_html=True)
        if decision["missing"]:
            for item in decision["missing"]:
                st.markdown(f'<div class="result-item"><span class="result-dot dot-miss"></span>{item}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-item" style="color:#4a6a82">Aucun équipement manquant</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if decision["negative"]:
        st.markdown('<div class="result-panel bad" style="margin-top:16px"><div class="result-panel-title">✕ Non-conformités</div>', unsafe_allow_html=True)
        for item in decision["negative"]:
            st.markdown(f'<div class="result-item"><span class="result-dot dot-bad"></span>{item}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if decision.get("details") and decision["status"] == "danger":
        st.markdown('<div class="result-panel bad" style="margin-top:16px"><div class="result-panel-title">⛔ Alertes critiques</div>', unsafe_allow_html=True)
        for item in decision["details"]:
            st.markdown(f'<div class="result-item"><span class="result-dot dot-bad"></span>{item}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def draw_video_decision(frame, decision):
    color_map = {
        "success": (0, 210, 140),
        "warning": (0, 165, 255),
        "danger":  (0, 60, 240)
    }
    text_map = {
        "success": f"ACCES AUTORISE  |  PROTECTION {decision['score']}%",
        "warning": f"PROTECTION INCOMPLETE  |  {decision['score']}%",
        "danger":  "ACCES REFUSE  |  ALERTE SANITAIRE"
    }

    color = color_map[decision["status"]]
    text = text_map[decision["status"]]
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 76), (8, 14, 30), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (0, 72), (w, 76), color, -1)
    cv2.putText(frame, text, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.08, color, 2, cv2.LINE_AA)

    y = 110
    for item in decision["positive"]:
        cv2.putText(frame, f"  {item}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 210, 140), 1, cv2.LINE_AA)
        y += 24

    for item in decision["missing"]:
        cv2.putText(frame, f"  MANQUE: {item}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 165, 255), 1, cv2.LINE_AA)
        y += 24

    for item in decision.get("negative", []):
        cv2.putText(frame, f"  ALERTE: {item}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (60, 80, 240), 1, cv2.LINE_AA)
        y += 24

    return frame


# =========================================================
# SIDEBAR
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "Accueil"

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">🛡️</div>
        <div>
            <div class="sidebar-logo-text">MedGuard</div>
            <div class="sidebar-logo-sub">EPI Detection</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="nav-section">Navigation</p>', unsafe_allow_html=True)

    if st.button("🏠  Accueil"):
        st.session_state.page = "Accueil"
        st.rerun()

    if st.button("🖼️  Détection Image"):
        st.session_state.page = "Image"
        st.rerun()

    if st.button("🎥  Caméra Temps Réel"):
        st.session_state.page = "Video"
        st.rerun()

    st.divider()

    st.markdown('<p class="nav-section">Configuration</p>', unsafe_allow_html=True)

    conf_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.25, 0.05)

    st.divider()

    st.markdown('<p class="nav-section">Accélération</p>', unsafe_allow_html=True)

    if DEVICE == "mps":
        st.markdown('<span class="status-pill ok">⚡ Apple Silicon MPS</span>', unsafe_allow_html=True)
    elif DEVICE == "cpu":
        st.markdown('<span class="status-pill warn">🔲 Mode CPU</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill ok">⚡ GPU CUDA</span>', unsafe_allow_html=True)

    st.markdown(f'<p style="font-size:12px; color:#3a5a72; margin-top:10px;">Device : <code style="color:#4a9a72">{DEVICE}</code></p>', unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <p style="font-size: 11px; color: #2a4a62; line-height: 1.7;">
        MedGuard v1.0<br>
        Projet PFA — EMSI<br>
        Détection EPI médicaux YOLO
    </p>
    """, unsafe_allow_html=True)


# =========================================================
# PAGE ACCUEIL
# =========================================================

def home_page():
    # HERO
    st.markdown("""
    <div class="hero-wrap">
        <div style="display: flex; align-items: center; justify-content: space-between; gap: 40px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 280px;">
                <div class="hero-tag">🧬 Intelligence Artificielle Médicale</div>
                <div class="hero-title">
                    Détection <span>EPI Médicaux</span><br>en Temps Réel
                </div>
                <div class="hero-subtitle">
                    Système intelligent basé sur YOLO pour détecter automatiquement
                    les équipements de protection individuelle dans les environnements médicaux.
                    Prévention active de la propagation des épidémies.
                </div>
                <div class="hero-pills">
                    <span class="hero-pill">🔬 Détection YOLO</span>
                    <span class="hero-pill">⚡ Temps réel</span>
                    <span class="hero-pill">🛡️ 13 classes EPI</span>
                    <span class="hero-pill">🚨 Alertes automatiques</span>
                    <span class="hero-pill">📊 Score de protection</span>
                </div>
            </div>
            <div class="scanner-container">
                <div class="scanner-ring"></div>
                <div class="scanner-ring"></div>
                <div class="scanner-ring"></div>
                <div class="scanner-core">🧑‍⚕️</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # METRICS
    st.markdown("""
    <div class="metrics-row">
        <div class="metric-card green">
            <div class="metric-num green">13</div>
            <div class="metric-label">Classes détectées</div>
        </div>
        <div class="metric-card blue">
            <div class="metric-num blue">5</div>
            <div class="metric-label">EPI obligatoires</div>
        </div>
        <div class="metric-card amber">
            <div class="metric-num amber">2</div>
            <div class="metric-label">Modes de détection</div>
        </div>
        <div class="metric-card red">
            <div class="metric-num red">3</div>
            <div class="metric-label">Niveaux d'alerte</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # DECISION LEVELS
    st.markdown('<div class="section-title">🎯 Niveaux de décision</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-grid">
        <div class="info-card success">
            <div class="icon-wrap">✅</div>
            <h4>Accès autorisé</h4>
            <p>La personne porte l'ensemble des EPI obligatoires : masque, gants, blouse médicale, charlotte et lunettes de protection.</p>
        </div>
        <div class="info-card warning">
            <div class="icon-wrap">⚠️</div>
            <h4>Protection incomplète</h4>
            <p>Certains équipements obligatoires ne sont pas confirmés. Un avertissement est affiché avec les équipements manquants.</p>
        </div>
        <div class="info-card danger">
            <div class="icon-wrap">🚨</div>
            <h4>Accès refusé / Alerte</h4>
            <p>Absence confirmée d'EPI ou comportement interdit détecté (manger, boire). Accès bloqué à la zone médicale.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CLASSES
    st.markdown('<div class="section-title">🧪 Classes détectées</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="result-panel ok" style="padding: 22px;">
            <div class="result-panel-title">✓ EPI conformes</div>
            <div class="class-grid">
                <span class="ctag ctag-ok">😷 Mask</span>
                <span class="ctag ctag-ok">🧤 Gloves</span>
                <span class="ctag ctag-ok">👒 Head Mask</span>
                <span class="ctag ctag-ok">🥼 Lab Coat</span>
                <span class="ctag ctag-ok">🥽 Goggles</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="result-panel bad" style="padding: 22px;">
            <div class="result-panel-title">✕ EPI non-conformes</div>
            <div class="class-grid">
                <span class="ctag ctag-no">❌ No Mask</span>
                <span class="ctag ctag-no">❌ No Gloves</span>
                <span class="ctag ctag-no">❌ No Head Mask</span>
                <span class="ctag ctag-no">❌ No Lab Coat</span>
                <span class="ctag ctag-no">❌ No Goggles</span>
                <span class="ctag ctag-warn">🍽️ Eating</span>
                <span class="ctag ctag-warn">🥤 Drinking</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="result-panel" style="padding: 20px; margin-top: 16px; border-color: rgba(148,163,184,0.15);">
        <div class="result-panel-title" style="color: #4a6a82;">ℹ Classes ignorées dans la décision sanitaire</div>
        <div class="class-grid">
            <span class="ctag ctag-neutral">🏢 EMSI — logo d'établissement uniquement</span>
        </div>
        <p style="font-size: 13px; color: #3a5a72; margin: 10px 0 0;">
            La classe EMSI est présente dans le dataset mais n'intervient pas dans l'évaluation du niveau de protection. Elle correspond uniquement au logo de l'établissement.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # OBJECTIVE
    st.markdown('<div class="section-title">🎯 Objectif du projet</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="video-info-card">
        <h4>Prévention active de la propagation des épidémies</h4>
        <p>
            MedGuard analyse en temps réel la conformité sanitaire des personnels entrant dans des zones médicales sensibles.
            Le système vérifie automatiquement le port des équipements de protection individuelle (EPI) et décide de l'autorisation
            ou du refus d'accès. Il détecte également les comportements interdits comme manger ou boire dans des zones stériles.
            <br><br>
            Basé sur un modèle YOLO entraîné sur 13 classes spécifiques au domaine médical, le système offre une précision
            élevée pour la détection en conditions réelles.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# PAGE DETECTION IMAGE
# =========================================================

def image_detection_page():
    st.markdown("""
    <div class="page-header">
        <div class="page-header-icon img">🖼️</div>
        <div>
            <h1>Détection par image</h1>
            <p>Chargez une photo pour analyser le niveau de protection EPI et obtenir une décision sanitaire.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("Le modèle YOLO n'a pas pu être chargé. Vérifiez le chemin : `best.pt`.")
        return

    uploaded_file = st.file_uploader(
        "Déposez votre image ici",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="visible"
    )

    st.markdown("""
    <div class="upload-hint">
        📌 <strong>Formats acceptés :</strong> JPG, JPEG, PNG, WEBP — Résolution recommandée : ≥ 640×640 px pour une meilleure précision.
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file is None:
        return

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="image-panel"><div class="image-panel-title">Image originale</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀  Lancer la détection EPI"):
        with st.spinner("Analyse YOLO en cours..."):
            image_np = np.array(image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            results = predict_yolo(source=image_bgr, conf=conf_threshold, imgsz=1280)

            if results is None:
                st.error("Erreur lors de l'exécution du modèle.")
                return

            result = results[0]
            annotated_image = result.plot()

            detections, labels = extract_detections(result, model)
            decision = evaluate_protection(labels)

        with col2:
            st.markdown('<div class="image-panel"><div class="image-panel-title">Résultat — détections YOLO</div>', unsafe_allow_html=True)
            st.image(annotated_image, channels="BGR", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        st.markdown('<div class="section-title">🩺 Décision sanitaire</div>', unsafe_allow_html=True)
        render_decision(decision)

        st.markdown("---")

        st.markdown('<div class="section-title">📋 Objets détectés</div>', unsafe_allow_html=True)

        if detections:
            # Build table HTML
            rows_html = ""
            for d in detections:
                c = d["Confiance"]
                if c >= 0.7:
                    badge_class = "conf-high"
                elif c >= 0.45:
                    badge_class = "conf-mid"
                else:
                    badge_class = "conf-low"

                clean = normalize_label(d["Classe"])
                if clean in POSITIVE_CLASSES:
                    tag_class = "ctag-ok"
                    tag_prefix = "✓"
                elif clean in NEGATIVE_CLASSES:
                    tag_class = "ctag-no"
                    tag_prefix = "✕"
                elif clean in DANGER_CLASSES:
                    tag_class = "ctag-warn"
                    tag_prefix = "⚠"
                else:
                    tag_class = "ctag-neutral"
                    tag_prefix = "—"

                rows_html += f"""
                <tr>
                    <td><span class="ctag {tag_class}" style="padding:4px 10px; font-size:12px;">{tag_prefix} {d['Classe']}</span></td>
                    <td><span class="conf-badge {badge_class}">{c:.2f}</span></td>
                </tr>
                """

            counts = {}
            for d in detections:
                counts[d["Classe"]] = counts.get(d["Classe"], 0) + 1

            count_rows = "".join(
                f'<div class="result-item"><span class="result-dot dot-ok"></span>{label} &nbsp;<span style="color:#3a5a72; font-family: DM Mono, monospace;">×{cnt}</span></div>'
                for label, cnt in sorted(counts.items())
            )

            ca, cb = st.columns([2, 1], gap="large")

            with ca:
                st.markdown(f"""
                <div class="result-panel ok" style="padding: 20px;">
                    <div class="result-panel-title">Détail des détections</div>
                    <table class="det-table">
                        <thead>
                            <tr>
                                <th>Classe</th>
                                <th>Confiance</th>
                            </tr>
                        </thead>
                        <tbody>{rows_html}</tbody>
                    </table>
                </div>
                """, unsafe_allow_html=True)

            with cb:
                st.markdown(f"""
                <div class="result-panel ok" style="padding: 20px;">
                    <div class="result-panel-title">Résumé par classe</div>
                    {count_rows}
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="video-info-card" style="border-color: rgba(245,158,11,0.2);">
                <h4 style="color: #fbbf24;">Aucun objet détecté</h4>
                <p>Essayez de réduire le seuil de confiance dans la barre latérale, ou utilisez une image avec une meilleure résolution.</p>
            </div>
            """, unsafe_allow_html=True)


# =========================================================
# PAGE VIDEO
# =========================================================

class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.conf_threshold = 0.25
        self.imgsz = 1280
        self.process_every = 1
        self.frame_count = 0
        self.last_frame = None
        self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if self.model is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        self.frame_count += 1

        if self.process_every > 1 and self.frame_count % self.process_every != 0 and self.last_frame is not None:
            return av.VideoFrame.from_ndarray(self.last_frame, format="bgr24")

        try:
            results = self.model.predict(
                source=img, conf=self.conf_threshold,
                imgsz=self.imgsz, device=DEVICE, verbose=False
            )
        except Exception:
            results = self.model.predict(
                source=img, conf=self.conf_threshold,
                imgsz=self.imgsz, device="cpu", verbose=False
            )

        result = results[0]
        annotated_frame = result.plot()

        detections, labels = extract_detections(result, self.model)
        decision = evaluate_protection(labels)

        annotated_frame = draw_video_decision(annotated_frame, decision)
        self.last_frame = annotated_frame

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


def video_detection_page():
    st.markdown("""
    <div class="page-header">
        <div class="page-header-icon vid">🎥</div>
        <div>
            <h1>Caméra temps réel</h1>
            <p>Détection EPI en direct avec analyse sanitaire continue et affichage des alertes sur flux vidéo.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("Le modèle YOLO n'a pas pu être chargé.")
        return

    if not WEBRTC_AVAILABLE:
        st.error("Modules manquants : installez `streamlit-webrtc` et `av`.")
        st.code("pip install streamlit-webrtc av")
        return

    st.markdown("""
    <div class="upload-hint">
        🎥 <strong>Instruction :</strong> Cliquez sur <em>START</em> puis autorisez l'accès à la caméra depuis votre navigateur.
        Les détections EPI et la décision sanitaire s'affichent en overlay directement sur la vidéo.
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        live_conf = st.slider("Confiance (caméra)", 0.0, 1.0, 0.25, 0.05, key="live_conf")

    with col2:
        quality = st.selectbox(
            "Résolution vidéo",
            [
                "Full HD 1080p (recommandé)",
                "HD 720p (plus fluide)",
                "Ultra 1080p + imgsz 1536"
            ]
        )

    with col3:
        process_mode = st.selectbox(
            "Mode traitement",
            ["Qualité maximale", "Équilibre qualité/fluidité", "Fluidité maximale"]
        )

    # Video config
    if quality == "Full HD 1080p (recommandé)":
        video_width, video_height, yolo_imgsz = 1920, 1080, 1280
    elif quality == "HD 720p (plus fluide)":
        video_width, video_height, yolo_imgsz = 1280, 720, 960
    else:
        video_width, video_height, yolo_imgsz = 1920, 1080, 1536

    process_every = {"Qualité maximale": 1, "Équilibre qualité/fluidité": 2, "Fluidité maximale": 3}[process_mode]

    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    ctx = webrtc_streamer(
        key="medguard-realtime",
        video_processor_factory=YOLOVideoProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={
            "video": {
                "width":  {"min": 1280, "ideal": video_width, "max": 1920},
                "height": {"min": 720,  "ideal": video_height, "max": 1080},
                "frameRate": {"ideal": 30, "max": 60}
            },
            "audio": False
        },
        async_processing=True
    )

    if ctx.video_processor:
        ctx.video_processor.conf_threshold = live_conf
        ctx.video_processor.imgsz = yolo_imgsz
        ctx.video_processor.process_every = process_every

    # Legend
    st.markdown("""
    <div class="video-info-card" style="margin-top: 24px;">
        <h4>📌 Overlay vidéo — légende des indicateurs</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px;">
            <div>
                <div class="result-item"><span class="result-dot dot-ok"></span><span style="color:#00e8a8; font-weight:600">Bandeau vert</span> &nbsp;— Accès autorisé, tous les EPI détectés</div>
                <div class="result-item"><span class="result-dot dot-miss"></span><span style="color:#fbbf24; font-weight:600">Bandeau orange</span> &nbsp;— Protection incomplète, EPI manquants</div>
                <div class="result-item"><span class="result-dot dot-bad"></span><span style="color:#f87171; font-weight:600">Bandeau rouge</span> &nbsp;— Accès refusé ou comportement interdit</div>
            </div>
            <div>
                <div class="result-item" style="color: #5a7a96;">✓ Les EPI détectés s'affichent en texte vert sur la vidéo</div>
                <div class="result-item" style="color: #5a7a96;">⚠ Les équipements manquants s'affichent en orange</div>
                <div class="result-item" style="color: #5a7a96;">✕ Les alertes critiques s'affichent en rouge vif</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown("""
        <div class="video-info-card">
            <h4>💻 Recommandation MacBook M3</h4>
            <p>
                Utilisez <strong>Full HD 1080p</strong> avec <strong>Qualité maximale</strong>.
                Si la vidéo rame, passez en mode <em>Équilibre qualité/fluidité</em> (traitement 1 frame sur 2).
                Le mode MPS Apple Silicon est activé automatiquement.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="video-info-card">
            <h4>📡 Connexion WebRTC</h4>
            <p>
                La détection se fait côté serveur via WebRTC. Si la caméra ne démarre pas,
                vérifiez que votre navigateur autorise l'accès à la caméra pour localhost,
                et que les ports STUN/TURN ne sont pas bloqués par votre firewall.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# ROUTER
# =========================================================

if st.session_state.page == "Accueil":
    home_page()
elif st.session_state.page == "Image":
    image_detection_page()
elif st.session_state.page == "Video":
    video_detection_page()
# ============================================================
#  APP.PY — Professional Research-Grade Dashboard
#  Real-Time Multimodal Emotion & Stress Detection System
#  Run with: streamlit run app.py
# ============================================================

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import io

from face_module import detect_face_emotions, get_dominant_emotion, capture_photo
from voice_module import record_audio, analyze_voice_energy, extract_mfcc_features
from fusion import full_analysis

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroSense | Stress Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Professional CSS ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

:root {
    --navy:     #0A0F1E;
    --navy2:    #0F1729;
    --card:     #131D35;
    --card2:    #1A2440;
    --border:   #1E2D50;
    --accent:   #4F8EF7;
    --accent2:  #7C3AED;
    --green:    #10B981;
    --yellow:   #F59E0B;
    --red:      #EF4444;
    --text:     #E2E8F0;
    --muted:    #64748B;
    --mono:     'Space Mono', monospace;
    --sans:     'DM Sans', sans-serif;
}

/* Global */
html, body, [class*="css"] {
    font-family: var(--sans);
    background-color: var(--navy) !important;
    color: var(--text) !important;
}

.stApp {
    background: linear-gradient(135deg, #0A0F1E 0%, #0D1525 50%, #0A0F1E 100%) !important;
}

/* Hide Streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--navy2); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── TOP HEADER BAR ── */
.top-bar {
    background: linear-gradient(90deg, var(--navy2) 0%, var(--card) 100%);
    border-bottom: 1px solid var(--border);
    padding: 0.6rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: -1rem -1rem 2rem -1rem;
}
.top-bar-left {
    display: flex; align-items: center; gap: 0.8rem;
}
.top-bar-logo {
    font-family: var(--mono);
    font-size: 1rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 2px;
}
.top-bar-version {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--muted);
    background: var(--border);
    padding: 2px 8px;
    border-radius: 20px;
}
.top-bar-right {
    font-size: 0.75rem;
    color: var(--muted);
    font-family: var(--mono);
}
.live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: var(--green);
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── HERO TITLE ── */
.hero {
    text-align: center;
    padding: 1.5rem 0 2rem 0;
}
.hero-label {
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 4px;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.15;
    margin-bottom: 0.5rem;
}
.hero-title span { color: var(--accent); }
.hero-sub {
    font-size: 0.9rem;
    color: var(--muted);
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.6;
}
.hero-paper {
    margin-top: 1rem;
    display: inline-block;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.35rem 1rem;
    font-size: 0.75rem;
    color: var(--muted);
    font-family: var(--mono);
}

/* ── PIPELINE BAR ── */
.pipeline {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    margin: 1.5rem 0;
    flex-wrap: wrap;
}
.pipe-step {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    text-align: center;
    min-width: 110px;
}
.pipe-step-icon { font-size: 1.2rem; }
.pipe-step-label {
    font-size: 0.65rem;
    font-family: var(--mono);
    color: var(--accent);
    letter-spacing: 1px;
    margin-top: 2px;
}
.pipe-step-tech {
    font-size: 0.6rem;
    color: var(--muted);
}
.pipe-arrow {
    color: var(--border);
    font-size: 1.2rem;
    padding: 0 0.3rem;
}

/* ── CARDS ── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.card-title {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 3px;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── STRESS RESULT ── */
.stress-result {
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin: 1rem 0;
}
.stress-result-high {
    background: linear-gradient(135deg, #1a0a0a, #2d0f0f);
    border: 1px solid #EF444444;
}
.stress-result-moderate {
    background: linear-gradient(135deg, #1a150a, #2d1f0a);
    border: 1px solid #F59E0B44;
}
.stress-result-low {
    background: linear-gradient(135deg, #0a1a12, #0a2d1a);
    border: 1px solid #10B98144;
}
.stress-level-label {
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 4px;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.stress-level-value {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.stress-score-badge {
    display: inline-block;
    font-family: var(--mono);
    font-size: 0.9rem;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    margin-top: 0.5rem;
}

/* ── METRIC TILES ── */
.metric-tile {
    background: var(--card2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    margin-bottom: 0.6rem;
}
.metric-tile-label {
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 2px;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.metric-tile-value {
    font-family: var(--mono);
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
}

/* ── EMOTION BARS ── */
.emotion-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.6rem;
    gap: 0.8rem;
}
.emotion-name {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--muted);
    width: 65px;
    text-align: right;
    flex-shrink: 0;
}
.emotion-bar-bg {
    flex: 1;
    background: var(--border);
    border-radius: 3px;
    height: 8px;
    overflow: hidden;
}
.emotion-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 1s ease;
}
.emotion-pct {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--text);
    width: 38px;
    text-align: right;
    flex-shrink: 0;
}

/* ── SECTION DIVIDER ── */
.sec-divider {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 2rem 0 1.5rem 0;
}
.sec-divider-line {
    flex: 1;
    height: 1px;
    background: var(--border);
}
.sec-divider-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 3px;
    color: var(--muted);
    text-transform: uppercase;
    white-space: nowrap;
}

/* ── UPLOAD ZONE ── */
.upload-hint {
    background: var(--card);
    border: 1px dashed var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
}
.upload-hint-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.upload-hint-text {
    font-size: 0.85rem;
    color: var(--muted);
    line-height: 1.6;
}

/* ── RECOMMENDATION ── */
.rec-box {
    background: var(--card2);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    font-size: 0.9rem;
    color: var(--text);
    margin: 1rem 0;
}

/* ── FORMULA DISPLAY ── */
.formula-box {
    background: var(--card2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    font-family: var(--mono);
    font-size: 0.85rem;
    color: var(--accent);
    text-align: center;
    letter-spacing: 1px;
    margin: 0.8rem 0;
}

/* ── FOOTER ── */
.footer {
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
}
.footer-left {
    font-size: 0.75rem;
    color: var(--muted);
    font-family: var(--mono);
    line-height: 1.8;
}
.footer-right {
    font-size: 0.7rem;
    color: var(--border);
    font-family: var(--mono);
}

/* ── STREAMLIT OVERRIDES ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    letter-spacing: 2px !important;
    padding: 0.7rem 2rem !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(79,142,247,0.35) !important;
}
.stButton > button:disabled {
    background: var(--card2) !important;
    color: var(--muted) !important;
}

.stFileUploader {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.5rem !important;
}

.stRadio > div {
    background: var(--card) !important;
    border-radius: 8px !important;
    padding: 0.5rem !important;
}

[data-testid="stSidebar"] {
    background: var(--navy2) !important;
    border-right: 1px solid var(--border) !important;
}

.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important;
}

.stStatusWidget {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

div[data-testid="metric-container"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.8rem !important;
}

.stExpander {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

img { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem 0;">
        <div style="font-family:'Space Mono',monospace;font-size:1.1rem;
             font-weight:700;color:#4F8EF7;letter-spacing:3px;">
            NEUROSENSE
        </div>
        <div style="font-size:0.65rem;color:#64748B;font-family:'Space Mono',monospace;
             margin-top:4px;letter-spacing:2px;">
            v1.0 · RESEARCH EDITION
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
         letter-spacing:3px;color:#64748B;margin-bottom:0.8rem;">
         SYSTEM CONFIG
    </div>
    """, unsafe_allow_html=True)

    input_mode = st.radio(
        "Input Mode",
        ["📸 Upload a Photo", "📷 Take a Photo", "🎥 Live Webcam (OpenCV)"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("")
    face_weight = st.slider("Face Modality Weight", 0.0, 1.0, 0.6, 0.05)
    voice_weight = round(1.0 - face_weight, 2)
    st.caption(f"Voice Weight auto-set to **{voice_weight}**")

    record_duration = st.slider("Voice Capture (seconds)", 2, 6, 3)

    st.markdown("---")

    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
         letter-spacing:3px;color:#64748B;margin-bottom:0.8rem;">
         ARCHITECTURE
    </div>
    """, unsafe_allow_html=True)

    for module, tech in [
        ("Face Module", "CNN · FER-2013"),
        ("Voice Module", "LSTM · MFCC×40"),
        ("Fusion Engine", "Weighted · S=f(F,V,P)"),
        ("Classifier", "3-Level Stress"),
    ]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;
             padding:0.4rem 0;border-bottom:1px solid #1E2D50;
             font-size:0.72rem;">
            <span style="color:#E2E8F0">{module}</span>
            <span style="font-family:'Space Mono',monospace;
                  font-size:0.6rem;color:#64748B">{tech}</span>
        </div>
        """, unsafe_allow_html=True)




# ── TOP BAR ──────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
    <div class="top-bar-left">
        <span class="top-bar-logo">NEUROSENSE</span>
        <span class="top-bar-version">v1.0-research</span>
    </div>
    <div class="top-bar-right">
        <span class="live-dot"></span>SYSTEM ACTIVE · MULTIMODAL AI
    </div>
</div>
""", unsafe_allow_html=True)


# ── HERO ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-label">Real-Time AI System</div>
    <div class="hero-title">
        Multimodal <span>Emotion</span> &<br>Stress Detection
    </div>
    <div class="hero-sub">
        A deep learning system integrating Convolutional Neural Networks
        for facial analysis and MFCC-based speech processing, combined
        through weighted fusion to deliver real-time psychophysiological
        stress assessment.
    </div>
    <div class="hero-paper">
        📄 &nbsp;IEEE · Dronacharya Group of Institutions · Greater Noida
    </div>
</div>
""", unsafe_allow_html=True)


# ── PIPELINE ─────────────────────────────────────────────────
st.markdown("""
<div class="pipeline">
    <div class="pipe-step">
        <div class="pipe-step-icon">📷</div>
        <div class="pipe-step-label">FACE INPUT</div>
        <div class="pipe-step-tech">Camera / Upload</div>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step">
        <div class="pipe-step-icon">🔲</div>
        <div class="pipe-step-label">CNN MODEL</div>
        <div class="pipe-step-tech">FER-2013 Dataset</div>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step">
        <div class="pipe-step-icon">🎤</div>
        <div class="pipe-step-label">VOICE INPUT</div>
        <div class="pipe-step-tech">Microphone</div>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step">
        <div class="pipe-step-icon">〰️</div>
        <div class="pipe-step-label">MFCC · LSTM</div>
        <div class="pipe-step-tech">40 Coefficients</div>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step">
        <div class="pipe-step-icon">⚡</div>
        <div class="pipe-step-label">FUSION</div>
        <div class="pipe-step-tech">S = f(F, V, P)</div>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step" style="border-color:#4F8EF7;">
        <div class="pipe-step-icon">📊</div>
        <div class="pipe-step-label">STRESS LEVEL</div>
        <div class="pipe-step-tech">3-Class Output</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ── INPUT SECTION ─────────────────────────────────────────────
frame = None
face_emotions = None
uploaded_file = None

st.markdown("""
<div class="sec-divider">
    <div class="sec-divider-line"></div>
    <div class="sec-divider-label">01 · INPUT CONFIGURATION</div>
    <div class="sec-divider-line"></div>
</div>
""", unsafe_allow_html=True)

col_in1, col_in2 = st.columns([1.2, 1])

with col_in1:
    if input_mode == "📸 Upload a Photo":
        st.markdown("""
        <div class="card">
            <div class="card-title">📸 Face Modality — Photo Input</div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="upload-hint">
            <div class="upload-hint-icon">🖼️</div>
            <div class="upload-hint-text">
                Upload a clear frontal selfie · JPG or PNG<br>
                <span style="color:#4F8EF7">Face centered · Good lighting · No mask</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drop photo here",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            file_bytes = np.asarray(
                bytearray(uploaded_file.read()), dtype=np.uint8
            )
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with st.spinner("CNN analyzing facial features..."):
                face_emotions = detect_face_emotions(frame)

            if face_emotions:
                dom = get_dominant_emotion(face_emotions)
                st.markdown(f"""
                <div style="background:#0a1a12;border:1px solid #10B98144;
                     border-radius:8px;padding:0.8rem 1rem;
                     font-family:'Space Mono',monospace;font-size:0.75rem;">
                    <span style="color:#10B981">✓ FACE DETECTED</span>
                    <span style="color:#64748B;margin-left:1rem;">
                    DOMINANT → </span>
                    <span style="color:#E2E8F0">{dom.upper()}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:#1a0a0a;border:1px solid #EF444444;
                     border-radius:8px;padding:0.8rem 1rem;
                     font-family:'Space Mono',monospace;font-size:0.75rem;">
                    <span style="color:#EF4444">✗ NO FACE DETECTED</span>
                    <span style="color:#64748B;margin-left:1rem;">
                    Try a clearer, well-lit photo</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    elif input_mode == "📷 Take a Photo":
        st.markdown("""
        <div class="card">
            <div class="card-title">📷 Face Modality — Take a Photo</div>
        """, unsafe_allow_html=True)

        camera_snap = st.camera_input(
            "Point your face at the camera and click 📸 Take photo"
        )

        if camera_snap:
            file_bytes = np.asarray(bytearray(camera_snap.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with st.spinner("CNN analyzing facial features..."):
                face_emotions = detect_face_emotions(frame)

            if face_emotions:
                dom = get_dominant_emotion(face_emotions)
                st.markdown(f"""
                <div style="background:#0a1a12;border:1px solid #10B98144;
                     border-radius:8px;padding:0.8rem 1rem;
                     font-family:'Space Mono',monospace;font-size:0.75rem;">
                    <span style="color:#10B981">✓ FACE DETECTED</span>
                    <span style="color:#64748B;margin-left:1rem;">DOMINANT → </span>
                    <span style="color:#E2E8F0">{dom.upper()}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:#1a0a0a;border:1px solid #EF444444;
                     border-radius:8px;padding:0.8rem 1rem;
                     font-family:'Space Mono',monospace;font-size:0.75rem;">
                    <span style="color:#EF4444">✗ NO FACE DETECTED</span>
                    <span style="color:#64748B;margin-left:1rem;">Try better lighting</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    else:  # Live Webcam (OpenCV)
        st.markdown("""
        <div class="card">
            <div class="card-title">🎥 Face Modality — Live Webcam (OpenCV)</div>
            <div style="background:#131D35;border:1px solid #1E2D50;
                 border-radius:8px;padding:1.2rem;text-align:center;
                 font-size:0.85rem;color:#64748B;line-height:1.8;">
                🎯 Face your webcam directly<br>
                💡 Ensure good lighting<br>
                📵 Close Zoom, Teams, Meet
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_in2:
    st.markdown("""
    <div class="card">
        <div class="card-title">🎤 Voice Modality — MFCC Analysis</div>
        <div style="font-size:0.85rem;color:#64748B;line-height:1.8;margin-bottom:1rem;">
            Upon clicking Analyze, the system will capture
            your voice and extract 40 Mel-Frequency Cepstral
            Coefficients encoding pitch, tone and rhythm.
        </div>
        <div class="formula-box">S = f(F, V, P)</div>
        <div style="display:flex;gap:0.5rem;margin-top:0.8rem;">
    """, unsafe_allow_html=True)

    for label_t, val, col_t in [
        ("FACE WEIGHT", f"{face_weight:.0%}", "#4F8EF7"),
        ("VOICE WEIGHT", f"{voice_weight:.0%}", "#7C3AED"),
        ("MFCC DIMS", "40", "#10B981"),
        ("DURATION", f"{record_duration}s", "#F59E0B"),
    ]:
        st.markdown(f"""
        <div style="flex:1;background:#0F1729;border:1px solid #1E2D50;
             border-radius:8px;padding:0.6rem;text-align:center;">
            <div style="font-family:'Space Mono',monospace;font-size:0.55rem;
                 letter-spacing:1px;color:#64748B;">{label_t}</div>
            <div style="font-family:'Space Mono',monospace;font-size:1rem;
                 font-weight:700;color:{col_t};">{val}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)


# ── ANALYZE BUTTON ────────────────────────────────────────────
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

col_b1, col_b2, col_b3 = st.columns([1.5, 2, 1.5])
with col_b2:
    btn_disabled = (
        (input_mode == "📸 Upload a Photo" and face_emotions is None) or
        (input_mode == "📷 Take a Photo" and face_emotions is None)
    )
    analyze = st.button(
        "⚡ INITIATE ANALYSIS",
        use_container_width=True,
        disabled=btn_disabled
    )

if btn_disabled:
    st.markdown("""
    <div style="text-align:center;font-family:'Space Mono',monospace;
         font-size:0.65rem;color:#64748B;letter-spacing:2px;margin-top:0.5rem;">
         ⬆️ CAPTURE / UPLOAD YOUR PHOTO ABOVE TO ACTIVATE
    </div>
    """, unsafe_allow_html=True)


# ── ANALYSIS & RESULTS ───────────────────────────────────────
if analyze:

    # Safety guard — should not happen due to btn_disabled, but just in case
    if face_emotions is None and input_mode != "🎥 Live Webcam (OpenCV)":
        st.error("⚠️ No face data. Please upload or take a photo first, then click Analyze.")
        st.stop()

    if input_mode == "📷 Take a Photo":
        # face already captured via st.camera_input above
        if frame is None or face_emotions is None:
            st.error("Please take a photo first using the camera widget above.")
            st.stop()

    elif input_mode == "🎥 Live Webcam (OpenCV)":
        with st.status("📷 Initialising camera array...", expanded=True) as s:
            frame, ok = capture_photo()
            if not ok:
                st.error("Camera error — switch to Photo Upload or Take a Photo mode.")
                st.stop()
            face_emotions = detect_face_emotions(frame)
            if not face_emotions:
                st.error("Face not detected — try Take a Photo mode instead.")
                st.stop()
            s.update(label="✅ Face modality captured", state="complete")

    with st.status(
        f"🎤 Voice capture active — speak now ({record_duration}s)...",
        expanded=True
    ) as s:
        audio, sr = record_audio(duration=record_duration)
        voice_stress = analyze_voice_energy(audio)
        mfcc_features = extract_mfcc_features(audio, sr)
        s.update(label="✅ Voice modality processed", state="complete")

    with st.status("⚡ Running fusion engine: S = f(F, V, P)...",
                   expanded=True) as s:
        time.sleep(0.5)
        result = full_analysis(face_emotions, voice_stress)
        s.update(label="✅ Multimodal fusion complete", state="complete")

    # ── RESULTS HEADER ────────────────────────────────────────
    st.markdown("""
    <div class="sec-divider">
        <div class="sec-divider-line"></div>
        <div class="sec-divider-label">02 · ANALYSIS RESULTS</div>
        <div class="sec-divider-line"></div>
    </div>
    """, unsafe_allow_html=True)

    # ── STRESS RESULT BOX ─────────────────────────────────────
    label = result["stress_label"]
    score = result["stress_score"]
    color = result["stress_color"]

    if "HIGH" in label:
        box_class = "stress-result-high"
        score_color = "#EF4444"
        emoji = "⚠"
    elif "MODERATE" in label:
        box_class = "stress-result-moderate"
        score_color = "#F59E0B"
        emoji = "◈"
    else:
        box_class = "stress-result-low"
        score_color = "#10B981"
        emoji = "✓"

    st.markdown(f"""
    <div class="stress-result {box_class}">
        <div class="stress-level-label">PSYCHOPHYSIOLOGICAL ASSESSMENT</div>
        <div class="stress-level-value" style="color:{score_color};">
            {emoji} &nbsp;{label}
        </div>
        <div style="font-family:'Space Mono',monospace;font-size:0.75rem;
             color:#64748B;margin-top:0.3rem;">
            COMPOSITE STRESS INDEX
        </div>
        <div class="stress-score-badge" style="background:{score_color}22;
             color:{score_color};border:1px solid {score_color}44;">
            {score:.1%} · FUSION CONFIDENCE
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Recommendation
    st.markdown(f"""
    <div class="rec-box">
        <span style="font-family:'Space Mono',monospace;font-size:0.65rem;
             letter-spacing:2px;color:#64748B;">CLINICAL RECOMMENDATION &nbsp;·&nbsp;</span>
        {result['recommendation']}
    </div>
    """, unsafe_allow_html=True)

    # ── MAIN RESULTS GRID ─────────────────────────────────────
    col_r1, col_r2, col_r3 = st.columns([1.1, 1.2, 0.9])

    # Column 1: Face photo
    with col_r1:
        st.markdown("""
        <div class="card">
            <div class="card-title">📷 CAPTURED FRAME · CNN INPUT</div>
        """, unsafe_allow_html=True)
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, use_container_width=True)

        dom = result["dominant_emotion"]
        dom_score = face_emotions.get(dom, 0) if face_emotions else 0
        st.markdown(f"""
        <div style="margin-top:0.8rem;background:#0F1729;border:1px solid #1E2D50;
             border-radius:8px;padding:0.8rem;text-align:center;">
            <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
                 letter-spacing:2px;color:#64748B;">DOMINANT CLASSIFICATION</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.6rem;
                 color:#4F8EF7;margin-top:0.2rem;">{dom.upper()}</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.7rem;
                 color:#10B981;">{dom_score:.1%} confidence</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Column 2: Emotion bars
    with col_r2:
        st.markdown("""
        <div class="card">
            <div class="card-title">😐 CNN EMOTION DISTRIBUTION</div>
        """, unsafe_allow_html=True)

        emotion_colors = {
            'happy':    '#10B981',
            'neutral':  '#4F8EF7',
            'sad':      '#7C3AED',
            'angry':    '#EF4444',
            'fear':     '#F59E0B',
            'surprise': '#06B6D4',
            'disgust':  '#F97316',
        }

        sorted_em = dict(sorted(
            face_emotions.items(), key=lambda x: x[1], reverse=True
        ))

        bars_html = ""
        for em, sc in sorted_em.items():
            col_em = emotion_colors.get(em, '#4F8EF7')
            pct = int(sc * 100)
            bars_html += f"""
            <div class="emotion-row">
                <div class="emotion-name">{em}</div>
                <div class="emotion-bar-bg">
                    <div class="emotion-bar-fill"
                         style="width:{pct}%;background:{col_em};"></div>
                </div>
                <div class="emotion-pct">{pct}%</div>
            </div>
            """

        st.markdown(bars_html, unsafe_allow_html=True)

        # Fused scores
        st.markdown("""
        <div style="height:1px;background:#1E2D50;margin:1rem 0;"></div>
        <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
             letter-spacing:3px;color:#64748B;margin-bottom:0.8rem;">
             FUSED OUTPUT · S=f(F,V,P)
        </div>
        """, unsafe_allow_html=True)

        fused = result["fused_emotions"]
        fused_sorted = dict(sorted(
            fused.items(), key=lambda x: x[1], reverse=True
        ))
        for em, sc in list(fused_sorted.items())[:4]:
            col_em = emotion_colors.get(em, '#4F8EF7')
            pct = int(sc * 100)
            st.markdown(f"""
            <div class="emotion-row">
                <div class="emotion-name">{em}</div>
                <div class="emotion-bar-bg">
                    <div class="emotion-bar-fill"
                         style="width:{pct}%;
                         background:linear-gradient(90deg,{col_em},{col_em}88);">
                    </div>
                </div>
                <div class="emotion-pct">{pct}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Column 3: Metrics
    with col_r3:
        st.markdown("""
        <div class="card">
            <div class="card-title">📈 SYSTEM METRICS</div>
        """, unsafe_allow_html=True)

        for lbl, val, col_t in [
            ("STRESS INDEX",  f"{score:.1%}", score_color),
            ("VOICE ENERGY",  f"{voice_stress:.1%}", "#7C3AED"),
            ("MFCC DIMS",     "40",           "#10B981"),
            ("FACE WEIGHT",   f"{face_weight:.0%}", "#4F8EF7"),
            ("VOICE WEIGHT",  f"{voice_weight:.0%}", "#F59E0B"),
            ("MODALITIES",    "2",            "#06B6D4"),
        ]:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="metric-tile-label">{lbl}</div>
                <div class="metric-tile-value" style="color:{col_t};">{val}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── RESULTS FOOTER ────────────────────────────────────────
    st.markdown("""
    <div class="footer">
        <div class="footer-left">
            <b style="color:#4F8EF7;">NEUROSENSE</b> · Real-Time Multimodal Emotion & Stress Detection System<br>
            Simran Tomar · Ridhi Raj · Dronacharya Group of Institutions, Greater Noida<br>
            Department of Computer Science Engineering
        </div>
        <div class="footer-right">
            CNN · LSTM · MFCC · WEIGHTED FUSION<br>
            IEEE CONFERENCE FORMAT · 2024
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── DEFAULT STATE ─────────────────────────────────────────────
else:
    if not (input_mode == "📸 Photo Upload" and face_emotions is not None):
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;
             border:1px dashed #1E2D50;border-radius:16px;margin-top:1rem;">
            <div style="font-size:3rem;margin-bottom:1rem;">🧠</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.5rem;
                 color:#E2E8F0;margin-bottom:0.5rem;">
                System Ready
            </div>
            <div style="font-family:'Space Mono',monospace;font-size:0.7rem;
                 color:#64748B;letter-spacing:2px;">
                CONFIGURE INPUT · UPLOAD PHOTO · INITIATE ANALYSIS
            </div>
        </div>
        """, unsafe_allow_html=True)
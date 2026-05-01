# ============================================================
#  APP.PY — Multimodal Emotion & Stress Detection Dashboard
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
    page_title="Emotion & Stress Detector",
    page_icon="🧠",
    layout="wide"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem; font-weight: 800;
        text-align: center; color: #1E1E2E;
    }
    .subtitle {
        text-align: center; color: #888;
        font-size: 0.95rem; margin-bottom: 1.5rem;
    }
    .stress-box {
        border-radius: 16px; padding: 1.8rem;
        text-align: center; font-size: 1.8rem;
        font-weight: bold; color: white; margin: 1rem 0;
    }
    .metric-card {
        background: #F8F9FA; border-radius: 12px;
        padding: 1rem; text-align: center;
        border: 1px solid #E9ECEF; margin-bottom: 0.6rem;
    }
    .section-title {
        font-size: 1.1rem; font-weight: 700;
        color: #1E1E2E; margin: 1.2rem 0 0.6rem 0;
        border-left: 4px solid #7C3AED;
        padding-left: 0.7rem;
    }
    .step-box {
        background: #EEF2FF; border-radius: 10px;
        padding: 0.8rem 1rem; margin-bottom: 0.5rem;
        border-left: 4px solid #7C3AED;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    st.markdown("### 🎛️ Fusion Weights")
    face_weight = st.slider("Face Weight", 0.0, 1.0, 0.6, 0.05)
    voice_weight = round(1.0 - face_weight, 2)
    st.info(f"Voice Weight: **{voice_weight}** *(auto)*")

    st.markdown("---")
    st.markdown("### 🎤 Voice Settings")
    record_duration = st.slider("Recording Duration (sec)", 2, 6, 3)

    st.markdown("---")
    st.markdown("### 📥 Input Mode")
    input_mode = st.radio(
        "Choose how to provide your face:",
        ["📸 Upload a Photo", "📷 Use Webcam"],
        index=0
    )

    st.markdown("---")
    st.caption(
        "**Real-Time Multimodal Emotion & Stress Detection**\n\n"
        "Simran Tomar & Ridhi Raj\n"
        "Dronacharya Group of Institution"
    )

# ── Header ────────────────────────────────────────────────────
st.markdown('<div class="main-title">🧠 Emotion & Stress Detector</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multimodal AI — Face (CNN) + Voice (MFCC) → Stress Level</div>',
            unsafe_allow_html=True)
st.markdown("---")

# ── How It Works ──────────────────────────────────────────────
with st.expander("📚 How does this work? (click to expand)"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 📷 Step 1 — Face (CNN)")
        st.markdown("A Convolutional Neural Network reads your face and detects 7 emotions: happy, sad, angry, fear, disgust, surprise, neutral.")
    with c2:
        st.markdown("#### 🎤 Step 2 — Voice (MFCC)")
        st.markdown("Your voice is converted into 40 MFCC numbers capturing pitch, tone and rhythm — exactly as in the research paper.")
    with c3:
        st.markdown("#### 🔀 Step 3 — Fusion")
        st.markdown("Both signals combine using: **S = f(F, V, P)** — the weighted fusion formula from your paper.")

st.markdown("")

# ── INPUT SECTION ─────────────────────────────────────────────
frame = None
face_emotions = None

if input_mode == "📸 Upload a Photo":
    st.markdown('<div class="section-title">📸 Step 1 — Upload Your Photo</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="step-box">Take a selfie or any photo of your face '
        'and upload it below. Make sure your face is clearly visible.</div>',
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader(
        "Upload a clear photo of your face",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        # Convert uploaded file to OpenCV frame
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_pil = Image.open(io.BytesIO(file_bytes.tobytes()
                             if hasattr(file_bytes,'tobytes') else uploaded_file.getvalue()))

        col_prev1, col_prev2, col_prev3 = st.columns([1, 2, 1])
        with col_prev2:
            st.image(uploaded_file, caption="Your uploaded photo", use_container_width=True)

        # Detect emotions immediately on upload
        with st.spinner("🧠 CNN analyzing your face..."):
            face_emotions = detect_face_emotions(frame)

        if face_emotions:
            st.success(f"✅ Face detected! Dominant emotion: **{get_dominant_emotion(face_emotions).upper()}**")
        else:
            st.error("❌ No face found in this photo. Try a clearer, well-lit selfie.")
            st.info("💡 Tips: Face should be centered, well-lit, looking forward, no mask.")

else:  # Webcam mode
    st.markdown('<div class="section-title">📷 Step 1 — Webcam Capture</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="step-box">Make sure your face is visible and well-lit, '
        'then click Analyze below.</div>',
        unsafe_allow_html=True
    )
    st.info("💡 Look directly at your webcam lens. Sit in good lighting.")

# ── VOICE + ANALYZE SECTION ───────────────────────────────────
st.markdown("")
st.markdown('<div class="section-title">🎤 Step 2 — Voice Recording</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="step-box">When you click Analyze, the system will record '
    f'your voice for {record_duration} seconds. Speak naturally — '
    'say your name, count numbers, or say anything aloud.</div>',
    unsafe_allow_html=True
)

st.markdown("")
col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
with col_b2:
    analyze = st.button(
        "🔍 Analyze My Stress Level",
        use_container_width=True,
        type="primary",
        disabled=(input_mode == "📸 Upload a Photo" and face_emotions is None)
    )

if input_mode == "📸 Upload a Photo" and face_emotions is None:
    st.caption("⬆️ Upload a photo first, then the Analyze button will activate.")

# ── ANALYSIS ─────────────────────────────────────────────────
if analyze:

    # ── Get face if webcam mode ───────────────────────────────
    if input_mode == "📷 Use Webcam":
        with st.status("📷 Capturing face from webcam...", expanded=True) as status:
            frame, camera_ok = capture_photo()
            if not camera_ok or frame is None:
                st.error("❌ Camera failed. Switch to 'Upload a Photo' mode in the sidebar.")
                st.stop()
            face_emotions = detect_face_emotions(frame)
            if not face_emotions:
                st.error("❌ No face detected. Try Upload mode instead — go to sidebar.")
                st.stop()
            status.update(label="✅ Face analyzed!", state="complete")

    # ── Record voice ──────────────────────────────────────────
    with st.status(
        f"🎤 Recording {record_duration} sec — SPEAK NOW! Say anything aloud...",
        expanded=True
    ) as status:
        audio, sr = record_audio(duration=record_duration)
        voice_stress = analyze_voice_energy(audio)
        mfcc_features = extract_mfcc_features(audio, sr)
        status.update(label="✅ Voice recorded!", state="complete")

    # ── Fusion ────────────────────────────────────────────────
    with st.status("🧠 Running fusion analysis: S = f(F, V, P)...",
                   expanded=True) as status:
        time.sleep(0.4)
        result = full_analysis(face_emotions, voice_stress)
        status.update(label="✅ Analysis complete!", state="complete")

    st.markdown("---")

    # ── RESULTS ──────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Your Results</div>',
                unsafe_allow_html=True)

    # Main stress box
    label = result["stress_label"]
    score = result["stress_score"]
    color = result["stress_color"]
    emoji = "🔴" if "HIGH" in label else "🟡" if "MODERATE" in label else "🟢"

    st.markdown(
        f'<div class="stress-box" style="background:{color};">'
        f'{emoji} {label} &nbsp;|&nbsp; {score:.0%}</div>',
        unsafe_allow_html=True
    )

    # Recommendation
    st.info(f"💡 **Recommendation:** {result['recommendation']}")

    st.markdown("")

    # ── Three columns ─────────────────────────────────────────
    col1, col2, col3 = st.columns([1.2, 1.3, 1])

    with col1:
        st.markdown('<div class="section-title">📷 Your Face</div>',
                    unsafe_allow_html=True)
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, use_container_width=True)
        st.markdown(
            f'<div class="metric-card">'
            f'<div style="color:#888;font-size:0.8rem">Dominant Emotion</div>'
            f'<b style="font-size:1.3rem">{result["dominant_emotion"].upper()}</b>'
            f'</div>', unsafe_allow_html=True
        )

    with col2:
        st.markdown('<div class="section-title">😊 Face Emotion Scores (CNN)</div>',
                    unsafe_allow_html=True)
        sorted_em = dict(sorted(
            face_emotions.items(), key=lambda x: x[1], reverse=True
        ))
        for emotion, sc in sorted_em.items():
            c_a, c_b = st.columns([1.4, 3])
            with c_a:
                st.markdown(
                    f'<div style="text-align:right;padding-top:6px;font-size:0.9rem">'
                    f'{emotion}</div>', unsafe_allow_html=True
                )
            with c_b:
                st.progress(float(sc), text=f"{sc:.0%}")

    with col3:
        st.markdown('<div class="section-title">📈 Metrics</div>',
                    unsafe_allow_html=True)
        for label_t, value in [
            ("🎤 Voice Stress", f"{voice_stress:.0%}"),
            ("⚖️ Face Weight", f"{face_weight:.0%}"),
            ("⚖️ Voice Weight", f"{voice_weight:.0%}"),
            ("🔢 MFCC Features", f"{len(mfcc_features)}"),
        ]:
            st.markdown(
                f'<div class="metric-card">'
                f'<div style="color:#888;font-size:0.8rem">{label_t}</div>'
                f'<b style="font-size:1.3rem">{value}</b>'
                f'</div>', unsafe_allow_html=True
            )

    # ── Fused scores ──────────────────────────────────────────
    st.markdown("")
    st.markdown(
        '<div class="section-title">🔀 Fused Emotion Scores — S = f(F, V, P)</div>',
        unsafe_allow_html=True
    )
    fused_sorted = dict(sorted(
        result["fused_emotions"].items(), key=lambda x: x[1], reverse=True
    ))
    cols_f = st.columns(len(fused_sorted))
    for i, (em, sc) in enumerate(fused_sorted.items()):
        with cols_f[i]:
            st.metric(em.capitalize(), f"{sc:.0%}")

    # ── Paper reference ───────────────────────────────────────
    st.markdown("---")
    st.caption(
        "📄 Based on: *Real-Time Multimodal Emotion and Stress Detection System* "
    )

# ── Default state ─────────────────────────────────────────────
elif not (input_mode == "📸 Upload a Photo" and face_emotions is not None):
    if not (input_mode == "📸 Upload a Photo" and uploaded_file if 'uploaded_file' in dir() else False):
        st.markdown("")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown("""
            <div style="text-align:center;padding:2.5rem 1rem;
                 border:2px dashed #DDD;border-radius:16px;color:#999;">
                <div style="font-size:3.5rem">🧠</div>
                <div style="font-size:1.1rem;margin-top:1rem;">
                    <b>Choose input mode from sidebar</b><br>
                    then upload a photo or use webcam
                </div>
                <div style="font-size:0.85rem;margin-top:0.5rem;">
                    📸 Upload mode is recommended for best results
                </div>
            </div>
            """, unsafe_allow_html=True)

import os
os.environ.setdefault("DEEPFACE_HOME", "D:\\deepface_models")

import cv2
import streamlit as st
from deepface import DeepFace

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
EMOTION_COLORS = {
    'angry':    '#e74c3c',
    'disgust':  '#8e44ad',
    'fear':     '#e67e22',
    'happy':    '#f1c40f',
    'sad':      '#3498db',
    'surprise': '#1abc9c',
    'neutral':  '#95a5a6',
}

def draw_face_box(frame, region, dominant_emotion):
    """Draw bounding box and emotion label on frame."""
    if not region or not isinstance(region, dict):
        return frame
    x = region.get('x', 0)
    y = region.get('y', 0)
    w = region.get('w', 0)
    h = region.get('h', 0)
    if w > 10 and h > 10:
        color = (0, 220, 100)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = dominant_emotion.capitalize()
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 6, y), color, -1)
        cv2.putText(frame, label, (x + 3, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)
    return frame

EMOTION_EMOJI = {
    'angry': '😠', 'disgust': '🤢', 'fear': '😨',
    'happy': '😄', 'sad': '😢', 'surprise': '😲', 'neutral': '😐'
}

def make_emotion_card(dominant: str, confidence: float) -> str:
    """Return an HTML card showing only the dominant emotion."""
    col   = EMOTION_COLORS.get(dominant, '#00d4aa')
    emoji = EMOTION_EMOJI.get(dominant, '🙂')
    pct   = min(confidence, 100.0)
    return f"""
    <div style="
        background:{col}18; border:2px solid {col}55; border-radius:16px;
        padding:28px 20px; text-align:center; margin-top:8px;
    ">
        <div style="font-size:4rem; line-height:1.1">{emoji}</div>
        <div style="font-size:1.8rem; font-weight:800; color:{col}; margin:10px 0 4px">
            {dominant.capitalize()}
        </div>
        <div style="font-size:0.9rem; color:#aaa; margin-bottom:10px">Confidence</div>
        <div style="
            background:#ffffff18; border-radius:30px; height:14px;
            width:100%; overflow:hidden; margin-bottom:8px;
        ">
            <div style="
                background:{col}; height:100%; width:{pct:.1f}%;
                border-radius:30px; transition:width 0.3s ease;
            "></div>
        </div>
        <div style="font-size:1.4rem; font-weight:700; color:{col}">{pct:.1f}%</div>
    </div>
    """

def main():
    st.set_page_config(page_title="AI Emotion Analytics", layout="wide", page_icon="🧠")

    st.markdown("""
    <style>
    .main-title {font-size:2rem; font-weight:700; color:#00d4aa; margin-bottom:0;}
    .sub-title  {font-size:1rem; color:#aaaaaa; margin-bottom:1rem;}
    .emotion-badge {
        display:inline-block; padding:6px 16px; border-radius:20px;
        font-size:1.1rem; font-weight:700; margin-top:8px;
        background:#00d4aa22; color:#00d4aa; border:1px solid #00d4aa55;
    }
    .status-box {
        background:#1a1a2e; border-radius:8px; padding:10px 16px;
        color:#aaaaaa; font-size:0.9rem; margin-top:8px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="main-title">🧠 Real-Time AI Emotion Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Live facial expression recognition powered by DeepFace + TensorFlow</p>', unsafe_allow_html=True)

    st.sidebar.header("⚙️ Controls")
    run = st.sidebar.checkbox('▶ Start Camera', value=False)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Detects 7 emotions:**")
    for e in EMOTIONS:
        st.sidebar.markdown(f"<span style='color:{EMOTION_COLORS[e]}'>● {e.capitalize()}</span>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("📷 Live Video Stream")
        frame_window = st.empty()
    with col2:
        st.subheader("🎭 Detected Emotion")
        chart_window  = st.empty()
        status_slot   = st.empty()

    if not run:
        st.info("☝️ Check **▶ Start Camera** in the sidebar to begin emotion analytics.")
        return

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        st.error("❌ Cannot open webcam. Make sure it is connected and not in use.")
        return

    status_slot.markdown('<div class="status-box">🔄 Initialising — stand by…</div>', unsafe_allow_html=True)
    frame_skip = 0  # analyse every 3rd frame to keep UI responsive

    while True:
        success, frame = camera.read()
        if not success:
            st.error("❌ Lost webcam connection.")
            break

        frame_skip += 1
        dominant = "detecting…"
        emotions  = {e: 0.0 for e in EMOTIONS}

        if frame_skip % 3 == 0:
            try:
                results = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True,
                )
                res = results[0] if isinstance(results, list) else results
                emotions_raw = res.get('emotion', {})
                # normalise keys to lowercase
                emotions = {k.lower(): v for k, v in emotions_raw.items()}
                dominant = res.get('dominant_emotion', 'neutral').lower()
                region   = res.get('region', {})
                frame    = draw_face_box(frame, region, dominant)

                # Update emotion card — dominant emotion only
                confidence = emotions.get(dominant, 0.0)
                chart_window.markdown(
                    make_emotion_card(dominant, confidence),
                    unsafe_allow_html=True
                )
                status_slot.markdown('<div class="status-box">✅ Face detected — analysing emotions</div>', unsafe_allow_html=True)

            except Exception as e:
                status_slot.markdown(f'<div class="status-box">⚠️ {str(e)[:120]}</div>', unsafe_allow_html=True)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb, channels="RGB", use_container_width=True)

    camera.release()

if __name__ == '__main__':
    main()
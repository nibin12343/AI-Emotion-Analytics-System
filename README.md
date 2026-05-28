# 🧠 Real-Time AI Emotion Analytics Dashboard

An intelligent monitoring tool that captures **live video** and uses **Deep Learning** to detect and analyze human facial emotions in real-time — powered by DeepFace, OpenCV, and Streamlit.

---

## 📌 Project Overview

The AI Emotion Analytics System demonstrates how AI can "see" and interpret human sentiment directly from a webcam feed. Potential applications include:

- 🛍️ Customer feedback & sentiment monitoring
- 🧘 Mental health awareness tools
- 🎓 Smart & adaptive education platforms
- 📊 UX research and human-computer interaction

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend UI | [Streamlit](https://streamlit.io/) |
| Computer Vision | [OpenCV](https://opencv.org/) (`cv2`) |
| Deep Learning Engine | [DeepFace](https://github.com/serengil/deepface) + TensorFlow |
| Data Visualization | Streamlit Native + Custom HTML/CSS Cards |

---

## 🎭 Detectable Emotions

| Emotion | Color |
|---|---|
| 😠 Angry | Red |
| 🤢 Disgust | Purple |
| 😨 Fear | Orange |
| 😄 Happy | Yellow |
| 😢 Sad | Blue |
| 😲 Surprise | Teal |
| 😐 Neutral | Grey |

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-emotion-analytics.git
cd ai-emotion-analytics
```

### 2. (Optional) Set DeepFace model directory

By default, models are saved to `D:\deepface_models`. To change this, update the path at the top of `app.py`:

```python
os.environ.setdefault("DEEPFACE_HOME", "/your/custom/path")
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`** should include:

```
streamlit
opencv-python
deepface
tensorflow
pandas
```

### 4. Run the application

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser, check **▶ Start Camera** in the sidebar, and the system begins analysing your expressions live.

---

## ⚙️ How It Works

```
Webcam Feed
    │
    ▼
┌─────────────────────────┐
│  OpenCV Frame Capture   │  — Reads frames in real time
└────────────┬────────────┘
             │ every 3rd frame
             ▼
┌─────────────────────────┐
│  DeepFace.analyze()     │  — Runs CNN-based emotion model
└────────────┬────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
Face Region       Emotion Scores
(x, y, w, h)   (7 probabilities)
    │                 │
    ▼                 ▼
Bounding Box    Dominant Emotion
drawn on frame  + Confidence Card
```

1. **Live Video Stream** — Webcam frames are captured continuously via OpenCV.
2. **Facial Landmark Detection** — DeepFace identifies the face bounding box `(x, y, w, h)` and overlays it on the video feed.
3. **Emotion Classification** — The deep learning model scores all 7 emotions and returns the dominant one with a confidence percentage.
4. **Live Analytics Card** — A dynamic confidence card updates in real time, colour-coded per emotion.

> **Performance note:** Analysis runs on every 3rd frame (`frame_skip % 3`) to keep the UI responsive while maintaining smooth video playback.

---

## 📁 Project Structure

```
ai-emotion-analytics/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is not intended for use in production systems involving surveillance, profiling, or any application without the informed consent of participants.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [DeepFace](https://github.com/serengil/deepface) by Sefik Ilkin Serengil
- [OpenCV](https://opencv.org/) community
- [Streamlit](https://streamlit.io/) for the rapid UI framework

# 🧠 AI Emotion Analytics System

A real-time computer vision application that captures live webcam footage and analyzes facial expressions using DeepFace, OpenCV, and Streamlit. The app detects dominant emotions in real time and presents them on an interactive dashboard.

This project is designed for educational and research use and showcases how AI can interpret human sentiment from visual cues in applications such as HCI, sentiment monitoring, learning analytics, and user experience research.

## ✨ Features

- Real-time webcam emotion detection
- Live video stream with face bounding box overlay
- Dominant emotion classification from seven categories
- Confidence scoring for detected emotions
- Streamlit-based interactive dashboard
- Lightweight and easy to run locally

### Supported emotions

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## 🧩 Tech Stack

| Component | Technology |
| --- | --- |
| UI | Streamlit |
| Computer Vision | OpenCV |
| AI / Emotion Recognition | DeepFace + TensorFlow |
| Visualization | Streamlit HTML/CSS cards |

## 🏗️ Project Structure

```bash
AI-Emotion-Analytics-System/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── emotion_analytics.zip  # Archived project assets / export
└── LICENSE                # License file (if present in the repo)
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/nibin12343/AI-Emotion-Analytics-System.git
cd AI-Emotion-Analytics-System
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, typically:

```text
http://localhost:8501
```

In the Streamlit sidebar, enable the camera by checking the Start Camera option.

## ⚙️ Model Configuration

The project sets a default DeepFace model directory in `app.py`:

```python
os.environ.setdefault("DEEPFACE_HOME", "D:\deepface_models")
```

If you want to store the models elsewhere, update that path to a folder you prefer.

Example:

```python
os.environ.setdefault("DEEPFACE_HOME", "/your/custom/path")
```

## 🧠 How It Works

1. The webcam captures live frames using OpenCV.
2. Each selected frame is analyzed by DeepFace for facial expressions.
3. The app extracts the emotion probabilities and detects the dominant emotion.
4. The system updates the UI with a live confidence card and overlays the detected face region on the image.
5. The result is displayed in real time through the Streamlit dashboard.

## 🧪 Example Workflow

```text
Webcam Feed
   │
   ▼
OpenCV Frame Capture
   │
   ▼
DeepFace.analyze(action='emotion')
   │
   ▼
Emotion probabilities + dominant emotion
   │
   ▼
Live dashboard + face bounding box overlay
```

## 🛠️ Troubleshooting

### Webcam not opening

- Make sure your webcam is connected and not already in use by another app.
- Try restarting the app after granting camera access.
- If using a remote environment, ensure the system supports local camera access.

### DeepFace model download issues

- Ensure you have a stable internet connection for the first model download.
- Verify that the `DEEPFACE_HOME` path is writable.
- If the app fails to load models, try creating the target folder manually before running the app.

### Dependency errors

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## ⚠️ Disclaimer

This project is intended for educational, research, and demonstration purposes only. It is not meant for surveillance, profiling, or decision-making systems without explicit consent and proper ethical review.

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgements

- [DeepFace](https://github.com/serengil/deepface)
- [OpenCV](https://opencv.org/)
- [Streamlit](https://streamlit.io/)
- TensorFlow community

## 📌 Notes

- The app analyzes frames every few iterations to balance performance and responsiveness.
- The UI is built for local demonstration and quick experimentation rather than large-scale production deployment.

If you want, I can also help you turn this into a more polished GitHub-style README with badges, a screenshot section, and a demo GIF placeholder.

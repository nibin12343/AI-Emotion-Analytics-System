# Real-Time AI Emotion Analytics Dashboard

## Project Overview 
The AI Emotion Analytics System is an intelligent monitoring tool that captures live video and uses Deep Learning to detect and analyze human emotions in real-time. This project demonstrates how AI can "see" and interpret human sentiment, which has applications in customer feedback, mental health monitoring, and smart education.

## Technical Stack 
* **Frontend UI**: Streamlit
* **Computer Vision**: OpenCV (cv2)
* **Deep Learning Engine**: DeepFace
* **Data Visualization**: Pandas & Streamlit Native Charts

## Quick Start
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

## How It Works
1. **Live Video Stream**: Real-time capture of the user's face via the webcam. 
2. **Facial Landmark Detection**: Automatically identifies the face region ($x, y, w, h$) to focus the analysis. 
3. **Emotion Classification**: Categorizes facial expressions into 7 key emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral. 
4. **Live Analytics Chart**: A dynamic bar chart that shows the AI's "Confidence Level" for every emotion simultaneously. 

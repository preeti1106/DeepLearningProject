# 🎵 MoodTunes AI

MoodTunes AI is a deep learning based emotion recognition and music recommendation system.  
The application detects a user's facial emotion from an uploaded image and suggests suitable songs along with motivational quotes.

## 🚀 Features

- Facial emotion detection using Deep Learning (CNN)
- Supports 4 emotions:
  - Happy 😊
  - Sad 😔
  - Angry 😠
  - Neutral 😐
- Emotion-based music recommendations
- Motivational quotes based on mood
- Attractive Streamlit web interface
- Real-time confidence score display

## 🧠 Model Used

- Custom Convolutional Neural Network (CNN)
- Trained on **FER2013 Dataset**
- Image Size: **48x48 grayscale**
- Achieved good classification accuracy

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- PIL (Image Processing)

## 📂 Dataset

FER2013 Facial Emotion Recognition Dataset

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
How It Works
Upload a face image
Model predicts emotion
App shows:
Detected Emotion
Confidence Score
Recommended Songs
Motivational Quote
🌟 Future Improvements
Real-time webcam emotion detection
Spotify API integration
More emotion classes
Better pretrained models.... Made by Preeti Singh

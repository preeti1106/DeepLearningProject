import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="MoodTunes AI",
    page_icon="🎵",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: white;
}

.block-container {
    padding-top: 2rem;
    max-width: 1100px;
}

.title {
    text-align:center;
    font-size:58px;
    font-weight:800;
    color:#d8b4fe;
    margin-bottom:0px;
    text-shadow: 0 0 15px rgba(216,180,254,0.4);
}

.subtitle {
    text-align:center;
    font-size:22px;
    color:#d1d5db;
    margin-bottom:35px;
}

.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
    border:1px solid rgba(255,255,255,0.12);
    padding:25px;
    border-radius:22px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    margin-top:18px;
    color:white;
}

.songcard {
    background: rgba(255,255,255,0.06);
    padding:14px;
    border-radius:14px;
    margin:10px 0;
    font-size:18px;
    font-weight:600;
}

.songcard a {
    color:#e9d5ff;
    text-decoration:none;
}

.songcard a:hover {
    color:#ffffff;
}

.quote {
    font-size:24px;
    line-height:1.7;
    font-style:italic;
    color:#f3e8ff;
}

.small {
    font-size:20px;
    color:#ddd6fe;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(48,48,1)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(4,activation='softmax'))

model.load_weights("weights.weights.h5")

labels = ['Angry','Happy','Sad','Neutral']

# ---------------- MUSIC DATA ----------------
songs = {
    "Happy":[
        ("Shape of You", "https://www.youtube.com/watch?v=JGwWNGJdvx8"),
        ("Brown Munde", "https://www.youtube.com/watch?v=VNs_cCtdbPc"),
        ("Blinding Lights", "https://www.youtube.com/watch?v=4NRXx6U8ABQ")
    ],
    "Sad":[
        ("Channa Mereya", "https://www.youtube.com/watch?v=284Ov7ysmfA"),
        ("Let Her Go", "https://www.youtube.com/watch?v=RBumgq5yVrA"),
        ("Agar Tum Saath Ho", "https://www.youtube.com/results?search_query=agar+tum+saath+ho")
    ],
    "Angry":[
        ("Relaxing Piano", "https://www.youtube.com/results?search_query=relaxing+piano"),
        ("Lo-fi Beats", "https://www.youtube.com/watch?v=jfKfPfyJRdk"),
        ("Waheguru Simran", "https://www.youtube.com/results?search_query=waheguru+simran")
    ],
    "Neutral":[
        ("Kesariya", "https://www.youtube.com/watch?v=BddP6PYo2gs"),
        ("Excuses", "https://www.youtube.com/results?search_query=excuses+ap+dhillon"),
        ("Top Hits", "https://www.youtube.com/results?search_query=top+hits")
    ]
}

quotes = {
    "Happy":"Your smile can light up any room. Keep glowing. ✨",
    "Sad":"Even the darkest night ends with sunrise. 🌅",
    "Angry":"Peace begins the moment you choose calm. 🌿",
    "Neutral":"Consistency creates success. Keep moving forward. 🚀"
}

# ---------------- UI ----------------
st.markdown("<div class='title'>🎵 MoodTunes AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Detect your vibe. Hear your mood.</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload Face Image", type=["jpg","jpeg","png"])

if uploaded_file:

    img = Image.open(uploaded_file).convert("L")
    img = img.resize((48,48))

    st.image(uploaded_file, caption="Uploaded Image", width=320)

    img_array = np.array(img)/255.0
    img_array = img_array.reshape(1,48,48,1)

    pred = model.predict(img_array, verbose=0)
    idx = np.argmax(pred)

    emotion = labels[idx]
    confidence = float(np.max(pred)*100)

    # RESULT CARD
    st.markdown(f"""
    <div class='card'>
        <h2>🧠 Detected Emotion: {emotion}</h2>
        <div class='small'>Confidence Score: {confidence:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # SONGS
    st.markdown("""
    <div class='card'>
        <h2>🎶 Recommended Songs</h2>
    </div>
    """, unsafe_allow_html=True)

    for song, link in songs[emotion]:
        st.markdown(f"""
        <div class='songcard'>
            ▶ <a href="{link}" target="_blank">{song}</a>
        </div>
        """, unsafe_allow_html=True)

    # QUOTE
    st.markdown(f"""
    <div class='card'>
        <h2>💬 Motivation For You</h2>
        <div class='quote'>"{quotes[emotion]}"</div>
    </div>
    """, unsafe_allow_html=True)
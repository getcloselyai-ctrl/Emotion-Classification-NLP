
import streamlit as st  # type: ignore
import joblib
import re
import os
import nltk
from nltk.corpus import stopwords

# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="😊",
    layout="centered"
)

# -------------------- LOAD MODEL FILES --------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "emotion_model.pkl"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    emotion_mapping = joblib.load(os.path.join(MODEL_DIR, "emotion_mapping.pkl"))
    return model, vectorizer, emotion_mapping

try:
    model, vectorizer, emotion_mapping = load_artifacts()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

# -------------------- STOPWORDS --------------------

@st.cache_resource
def load_stopwords():
    try:
        return set(stopwords.words("english")) - {"not", "no", "nor"}
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english")) - {"not", "no", "nor"}

stop_words = load_stopwords()

# -------------------- TEXT PREPROCESSING --------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# -------------------- EMOTION LABELS --------------------

# Supports either label-to-number or number-to-label mappings
def get_emotion_label(prediction):
    if prediction in emotion_mapping:
        return str(emotion_mapping[prediction])

    for label, number in emotion_mapping.items():
        if number == prediction:
            return str(label)

    return str(prediction)

emotion_emojis = {
    "happy": "😊",
    "happiness": "😊",
    "sadness": "😢",
    "anger": "😡",
    "love": "❤️",
    "fear": "😨",
    "surprise": "😲"
}

# -------------------- STREAMLIT INTERFACE --------------------

st.title("😊 Emotion Classification")
st.write("Discover the emotion behind your text using Machine Learning.")

st.divider()

user_text = st.text_area(
    "Enter your text:",
    placeholder="Example: I am feeling really happy today!",
    height=150
)

if st.button("Predict Emotion", type="primary", use_container_width=True):

    if not user_text.strip():
        st.warning("Please enter some text first.")

    else:
        cleaned_text = clean_text(user_text)

        if not cleaned_text.strip():
            st.warning("Your text is empty after preprocessing. Please try another sentence.")

        else:
            transformed_text = vectorizer.transform([cleaned_text])
            prediction = model.predict(transformed_text)[0]

            emotion = get_emotion_label(prediction)
            emoji = emotion_emojis.get(emotion.lower(), "✨")

            st.divider()
            st.subheader("Prediction Result")

            st.markdown(
                f"## {emoji} {emotion.capitalize()}"
            )

            st.success("Emotion successfully predicted!")

            with st.expander("View processed text"):
                st.write(cleaned_text)

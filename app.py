
import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="EmotiSense",
    page_icon="🟠",
    layout="centered"
)

# ---------------- CUSTOM THEME ----------------
st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background: #080808;
        color: #ffffff;
    }

    /* Content width and spacing */
    .block-container {
        max-width: 750px;
        padding-top: 4rem;
        padding-bottom: 3rem;
    }

    /* Main title */
    h1 {
        color: #ff8c00 !important;
        text-align: center;
        font-size: 44px !important;
        font-weight: 800 !important;
    }

    /* Headings and text */
    h2, h3, p, label {
        color: #ffffff !important;
    }

    /* Input area */
    [data-testid="stTextArea"] textarea {
        background-color: #ffffff !important;
        color: #111111 !important;
        border: 2px solid #ff8c00 !important;
        border-radius: 12px !important;
        font-size: 14px !important;
    }

    [data-testid="stTextArea"] textarea::placeholder {
        color: #777777 !important;
        font-size: 12px !important;
    }

    /* Analyze button */
    .stButton button {
        background-color: #ff8c00 !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 0.65rem !important;
    }

    .stButton button:hover {
        background-color: #ffa726 !important;
        color: #000000 !important;
        border: none !important;
    }

    /* Divider */
    hr {
        border-color: #ff8c00 !important;
    }

    /* Success and warning messages */
    [data-testid="stAlert"] {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/emotion_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    emotion_mapping = joblib.load("models/emotion_mapping.pkl")

    return model, vectorizer, emotion_mapping


model, vectorizer, emotion_mapping = load_artifacts()


# ---------------- LOAD STOPWORDS ----------------
@st.cache_resource
def load_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


stop_words = load_stopwords()


# ---------------- TEXT PREPROCESSING ----------------
def preprocess_text(text):
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove non-ASCII characters
    text = text.encode("ascii", "ignore").decode("ascii")

    # Remove stopwords
    words = text.split()
    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# ---------------- EMOTION MAPPING ----------------
id_to_emotion = {
    int(value): str(key)
    for key, value in emotion_mapping.items()
}

emotion_emoji = {
    "joy": "😊",
    "happiness": "😊",
    "sadness": "😢",
    "anger": "😠",
    "love": "❤️",
    "fear": "😨",
    "surprise": "😲",
    "neutral": "😐"
}


# ---------------- APPLICATION UI ----------------
st.title("EmotiSense")

st.markdown(
    "<p style='text-align:center; color:#cccccc;'>"
    "Understand the emotion behind your words."
    "</p>",
    unsafe_allow_html=True
)

st.write("")

sentence = st.text_area(
    "Enter your sentence",
    placeholder="Example: I feel so happy and excited today!",
    height=110,
    max_chars=1000
)

st.write("")

if st.button("ANALYZE EMOTION", use_container_width=True):

    if not sentence.strip():
        st.warning("Please enter a sentence first.")

    else:
        cleaned_text = preprocess_text(sentence)

        if not cleaned_text:
            st.warning("Please enter a meaningful sentence.")

        else:
            vector = vectorizer.transform([cleaned_text])

            if vector.nnz == 0:
                st.warning(
                    "Unable to recognize these words. "
                    "Please try another sentence."
                )

            else:
                prediction = model.predict(vector)[0]

                emotion = id_to_emotion.get(
                    int(prediction),
                    str(prediction)
                )

                emoji = emotion_emoji.get(
                    emotion.lower(),
                    "✨"
                )

                st.divider()
                st.subheader("DETECTED EMOTION")

                st.markdown(
                    f"<h1 style='color:#ff8c00; text-align:center;'>"
                    f"{emoji} {emotion.title()}"
                    f"</h1>",
                    unsafe_allow_html=True
                )


import os
import re
import nltk
import joblib
import streamlit as st

from nltk.corpus import stopwords


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="😊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 2. LOAD NLTK STOPWORDS
# ============================================================

@st.cache_resource
def get_stopwords():
    """
    Download and load English stopwords.
    The same stopword-removal approach must be used during training.
    """

    nltk_data_path = os.path.join(
        os.path.dirname(BASE_DIR),
        "nltk_data"
    )

    os.makedirs(nltk_data_path, exist_ok=True)

    if nltk_data_path not in nltk.data.path:
        nltk.data.path.append(nltk_data_path)

    try:
        return set(stopwords.words("english"))

    except LookupError:
        downloaded = nltk.download(
            "stopwords",
            download_dir=nltk_data_path,
            quiet=True
        )

        if not downloaded:
            raise RuntimeError(
                "Unable to download NLTK stopwords."
            )

        return set(stopwords.words("english"))


# ============================================================
# 3. LOAD MODEL, VECTORIZER AND LABEL MAPPING
# ============================================================

@st.cache_resource
def load_artifacts():

    files = {
        "model": os.path.join(BASE_DIR, "emotion_model.pkl"),
        "vectorizer": os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"),
        "mapping": os.path.join(BASE_DIR, "emotion_mapping.pkl")
    }

    for name, path in files.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing {name} file: {path}"
            )

    model = joblib.load(files["model"])
    vectorizer = joblib.load(files["vectorizer"])
    emotion_mapping = joblib.load(files["mapping"])

    return model, vectorizer, emotion_mapping


try:
    STOP_WORDS = get_stopwords()
    model, vectorizer, emotion_mapping = load_artifacts()

except Exception as error:
    st.error("The application could not initialize.")
    st.exception(error)
    st.stop()


# ============================================================
# 4. TEXT PREPROCESSING
# ============================================================

def clean_text(text: str) -> str:
    """
    Preprocessing must match the notebook's training pipeline:
    - Convert to lowercase
    - Remove non-ASCII characters, punctuation and digits
    - Remove English stopwords
    - Remove extra whitespace
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove punctuation, numbers, emojis and non-ASCII characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    words = [
        word for word in text.split()
        if word not in STOP_WORDS
    ]

    return " ".join(words)


# ============================================================
# 5. CONVERT MODEL OUTPUT INTO EMOTION LABEL
# ============================================================

def get_emotion_label(prediction):
    """
    Supports both mapping formats:

    {'sadness': 0, 'anger': 1, ...}

    {0: 'sadness', 1: 'anger', ...}
    """

    if hasattr(prediction, "item"):
        prediction = prediction.item()

    # Mapping: emotion -> integer
    for label, number in emotion_mapping.items():
        if str(number) == str(prediction):
            return str(label).capitalize()

    # Mapping: integer -> emotion
    for number, label in emotion_mapping.items():
        if str(number) == str(prediction):
            return str(label).capitalize()

    # Prediction may already be a label
    return str(prediction).capitalize()


# ============================================================
# 6. EMOTION DISPLAY CONFIGURATION
# ============================================================

EMOTION_EMOJIS = {
    "Happiness": "😊",
    "Sadness": "😢",
    "Anger": "😡",
    "Fear": "😨",
    "Love": "❤️",
    "Surprise": "😲"
}


EMOTION_DESCRIPTIONS = {
    "Happiness": "The text expresses positive feelings or joy.",
    "Sadness": "The text expresses sadness or disappointment.",
    "Anger": "The text expresses frustration or anger.",
    "Fear": "The text expresses worry, anxiety or fear.",
    "Love": "The text expresses affection or love.",
    "Surprise": "The text expresses unexpected feelings or surprise."
}


# ============================================================
# 7. PREDICTION FUNCTION
# ============================================================

def predict_emotion(text: str):

    cleaned_text = clean_text(text)

    if not cleaned_text:
        raise ValueError(
            "No usable words remain after preprocessing. "
            "Please enter a longer or different sentence."
        )

    # Convert input into TF-IDF features
    transformed_text = vectorizer.transform([cleaned_text])

    # Predict emotion
    prediction = model.predict(transformed_text)[0]

    emotion = get_emotion_label(prediction)

    # Get probability for the predicted class, if supported
    confidence = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            transformed_text
        )[0]

        classes = list(model.classes_)

        predicted_index = next(
            (
                index for index, class_label in enumerate(classes)
                if str(class_label) == str(prediction)
            ),
            None
        )

        if predicted_index is not None:
            confidence = float(
                probabilities[predicted_index]
            ) * 100

    return emotion, confidence, cleaned_text


# ============================================================
# 8. APPLICATION DESIGN
# ============================================================

st.title("😊 Emotion Classification")

st.markdown(
    """
    Discover the emotion behind your words using
    **Natural Language Processing and Machine Learning**.

    Enter a sentence and let the model identify the emotion.
    """
)

st.divider()

st.subheader("Analyze your text")

user_input = st.text_area(
    "Enter your sentence",
    placeholder="Example: I feel so happy and excited today!",
    height=130,
    max_chars=5000
)

st.caption(
    "Supported emotions: Happiness, Sadness, Anger, Fear, Love and Surprise."
)

predict_button = st.button(
    "✨ Predict Emotion",
    type="primary",
    use_container_width=True
)


# ============================================================
# 9. DISPLAY PREDICTION
# ============================================================

if predict_button:

    if not user_input.strip():

        st.warning("Please enter some text before predicting.")

    else:

        with st.spinner("Analyzing your text..."):

            try:

                emotion, confidence, cleaned_text = predict_emotion(
                    user_input
                )

                emoji = EMOTION_EMOJIS.get(emotion, "💬")

                description = EMOTION_DESCRIPTIONS.get(
                    emotion,
                    "The model has identified an emotion in your text."
                )

                st.divider()

                st.subheader("Prediction Result")

                st.markdown(
                    f"<div style='text-align:center; padding:25px; "
                    f"border:1px solid rgba(128,128,128,0.3); "
                    f"border-radius:15px;'>"
                    f"<div style='font-size:65px;'>{emoji}</div>"
                    f"<h1>{emotion}</h1>"
                    f"<p>{description}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                if confidence is not None:

                    st.subheader("Model Confidence")

                    st.progress(
                        min(max(confidence / 100, 0.0), 1.0)
                    )

                    st.write(
                        f"**{confidence:.2f}%**"
                    )

                    st.caption(
                        "This is the model's predicted probability, "
                        "not a guarantee of correctness."
                    )

                with st.expander("View processed text"):
                    st.write(cleaned_text)

            except Exception as error:

                st.error("Prediction failed.")
                st.exception(error)


# ============================================================
# 10. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; opacity:0.7;">
        Built with Python · NLP · TF-IDF · Scikit-learn · Streamlit
        <br><br>
        Emotion Classification Project
    </div>
    """,
    unsafe_allow_html=True
)

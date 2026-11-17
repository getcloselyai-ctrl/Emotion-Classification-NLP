BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    try:
        model = joblib.load(
            os.path.join(BASE_DIR, "emotion_model.pkl")
        )

        vectorizer = joblib.load(
            os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
        )

        emotion_mapping = joblib.load(
            os.path.join(BASE_DIR, "emotion_mapping.pkl")
        )

        return model, vectorizer, emotion_mapping

    except Exception as e:
        st.error(f"Could not load model files: {e}")
        st.stop()


model, vectorizer, emotion_mapping = load_models()

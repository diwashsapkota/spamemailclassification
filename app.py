import streamlit as st
import joblib

# Load models and vectorizer
@st.cache_resource
def load_model_and_vectorizer(model_name):
    model = joblib.load(f"{model_name}")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

# UI layout
st.set_page_config(page_title="Spam Classifier", layout="centered")
st.title("📧 Spam Email Classifier")
st.write("Enter your email message below and choose a model to predict if it's spam or ham.")

email_input = st.text_area("Email Message", height=200)
model_option = st.selectbox("Choose a Classifier", [
    "logistic_regression_model.pkl",
    "naive_bayes_model.pkl",
    "random_forest_model.pkl"
])

if st.button("Classify"):
    if email_input.strip():
        model, vectorizer = load_model_and_vectorizer(model_option)
        transformed_input = vectorizer.transform([email_input])
        prediction = model.predict(transformed_input)
        result = "🚫 Spam" if prediction[0] == 1 else "✅ Ham"
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter an email message to classify.")


# To Run:
# python3 -m streamlit run app.py